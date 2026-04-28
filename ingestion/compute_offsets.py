#!/usr/bin/env python3
"""
compute_offsets.py

One-time (and re-runnable) script that computes youtube_offset_ms for existing
episodes by aligning DB utterance timestamps against downloaded YouTube transcripts.

How it works:
  1. Scans youtube_transcripts/ for .json files (downloaded by download_youtube_transcripts.py)
  2. For each JSON file, tries to find the matching episode in the DB by title
  3. Loads that episode's utterances from the DB
  4. Aligns the two transcripts to find the consistent time delta
  5. Stores the result in episodes.youtube_offset_ms

Run after bulk-downloading YouTube transcripts:
    python compute_offsets.py
    python compute_offsets.py --dry-run          # print offsets without writing
    python compute_offsets.py --episode "Kristen Bell"   # single episode
    python compute_offsets.py --min-matches 5    # require more matches for confidence

Supports transcript files in these formats (auto-detected by extension):
    .json  — from our download_youtube_transcripts.py script
    .srt   — from yt-bulk-subtitles-downloader or yt-dlp
    .txt   — tactiq.io style (HH:MM:SS.mmm text...)

The script skips episodes where youtube_offset_ms is already non-zero.
Use --force to recompute everything.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

TRANSCRIPT_DIR = Path(__file__).parent / "youtube_transcripts"


# ──────────────────────────────────────────────
# Title matching
# ──────────────────────────────────────────────
def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def match_episode(json_path: Path, db_episodes: list[dict]) -> dict | None:
    """Find the best-matching DB episode for a YouTube transcript JSON file."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    yt_title = data.get("title", "")

    # Strip common suffix
    yt_clean = re.sub(r"\s*\|\s*armchair expert.*$", "", yt_title, flags=re.IGNORECASE).strip()
    yt_clean = re.sub(r"^\d+[\s\-–:]+", "", yt_clean).strip()

    best_ep, best_score = None, 0.0
    for ep in db_episodes:
        score = jaccard(yt_clean, ep["file_stem"])
        if score > best_score:
            best_score, best_ep = score, ep

    if best_score >= 0.4:
        return best_ep
    return None


# ──────────────────────────────────────────────
# Offset computation
# ──────────────────────────────────────────────
def load_yt_segments(json_path: Path) -> list[dict]:
    """Load segments from a JSON transcript (downloaded by our script)."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data.get("segments", [])


def load_yt_segments_from_txt(txt_path: Path) -> list[dict]:
    """Load segments from a tactiq-style .txt transcript (HH:MM:SS.mmm text...)."""
    segments = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\d{2}):(\d{2}):(\d{2})\.(\d+)\s+(.*)", line)
        if m:
            h, mi, s, ms_str, text = m.groups()
            total_ms = (int(h) * 3600 + int(mi) * 60 + int(s)) * 1000 + int(ms_str[:3].ljust(3, "0"))
            segments.append({"start": total_ms / 1000, "text": text.strip()})
    return segments


def _srt_timestamp_to_ms(ts: str) -> int:
    """Convert SRT timestamp '00:01:23,456' or '00:01:23.456' to milliseconds."""
    ts = ts.strip().replace(",", ".")
    m = re.match(r"(\d+):(\d{2}):(\d{2})\.(\d+)", ts)
    if not m:
        return 0
    h, mi, s, ms_str = m.groups()
    return (int(h) * 3600 + int(mi) * 60 + int(s)) * 1000 + int(ms_str[:3].ljust(3, "0"))


def load_yt_segments_from_srt(srt_path: Path) -> list[dict]:
    """Load segments from an SRT subtitle file.

    SRT format:
        1
        00:00:01,000 --> 00:00:03,500
        Hello and welcome to Armchair Expert

        2
        00:00:03,600 --> 00:00:06,000
        I am Dax Shepard
    """
    segments = []
    content = srt_path.read_text(encoding="utf-8", errors="replace")
    # Split on blank lines separating subtitle blocks
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        # Line 0: index number, Line 1: timestamps, Line 2+: text
        ts_line = lines[1]
        m = re.match(r"([\d:,\.]+)\s*-->\s*([\d:,\.]+)", ts_line)
        if not m:
            continue
        start_ms = _srt_timestamp_to_ms(m.group(1))
        text = " ".join(lines[2:])
        # Strip HTML tags that some SRT files include
        text = re.sub(r"<[^>]+>", "", text).strip()
        if text:
            segments.append({"start": start_ms / 1000, "text": text})
    return segments


def load_segments(fpath: Path) -> list[dict]:
    """Auto-detect format and load YouTube transcript segments."""
    suffix = fpath.suffix.lower()
    if suffix == ".json":
        return load_yt_segments(fpath)
    elif suffix == ".srt":
        return load_yt_segments_from_srt(fpath)
    elif suffix == ".txt":
        return load_yt_segments_from_txt(fpath)
    return []


def compute_offset(utterances: list[dict], yt_segments: list[dict],
                   min_matches: int = 3) -> int | None:
    """
    Returns offset_ms such that: youtube_time_ms ≈ db_time_ms + offset_ms

    Matches long DB utterances against nearby YouTube segments using key-word overlap,
    computes the median delta, filters outliers, and returns the final offset.
    """
    # Build a lookup: for each YouTube segment, record (start_ms, text_lower)
    yt_lookup = [(int(s["start"] * 1000), s["text"].lower()) for s in yt_segments]

    deltas = []
    for utt in utterances:
        text = (utt.get("text") or "").strip()
        if len(text) < 80:
            continue

        db_ms = utt.get("start_ms") or 0
        words = [w for w in text.lower().split() if len(w) > 4][:6]
        if len(words) < 3:
            continue

        # Search YouTube segments within a ±5 minute window first, then full scan
        best_ms = None
        best_hits = 0

        search_window = [(ms, t) for ms, t in yt_lookup if abs(ms - db_ms) < 7_200_000]
        if not search_window:
            search_window = yt_lookup

        for i in range(len(search_window)):
            # Combine a few consecutive segments to handle phrase splits
            combined = " ".join(t for _, t in search_window[i:i+5])
            hits = sum(1 for w in words if w in combined)
            if hits > best_hits:
                best_hits = hits
                best_ms = search_window[i][0]

        if best_hits >= 3 and best_ms is not None:
            delta = best_ms - db_ms
            if abs(delta) < 7_200_000:  # within 2 hours
                deltas.append(delta)

    if len(deltas) < min_matches:
        return None

    median_delta = int(statistics.median(deltas))
    mad = statistics.median([abs(d - median_delta) for d in deltas]) or 1000

    # Filter outliers: keep only deltas within 3× MAD of median
    clean = [d for d in deltas if abs(d - median_delta) <= max(mad * 3, 10_000)]
    if len(clean) < min_matches:
        return None

    return int(statistics.median(clean))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compute YouTube timing offsets for all episodes")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print computed offsets without writing to DB")
    parser.add_argument("--episode", default="",
                        help="Process only episodes whose name contains this string")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even for episodes that already have a non-zero offset")
    parser.add_argument("--min-matches", type=int, default=3,
                        help="Minimum phrase matches required to trust the offset (default: 3)")
    parser.add_argument("--transcript-dir", default=str(TRANSCRIPT_DIR),
                        help="Directory containing YouTube transcript JSON/TXT files")
    args = parser.parse_args()

    transcript_dir = Path(args.transcript_dir)
    if not transcript_dir.exists():
        print(f"ERROR: Transcript directory not found: {transcript_dir}")
        sys.exit(1)

    json_files = list(transcript_dir.glob("*.json"))
    srt_files = list(transcript_dir.glob("*.srt"))
    txt_files = [f for f in transcript_dir.glob("*.txt") if f.stem != "manifest"]
    print(f"Found {len(json_files)} JSON + {len(srt_files)} SRT + {len(txt_files)} TXT transcripts in {transcript_dir}")

    # Connect to DB
    conn = psycopg2.connect(os.environ["DATABASE_URL"])

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT id, file_stem, youtube_offset_ms FROM episodes ORDER BY file_stem")
        db_episodes = [dict(r) for r in cur.fetchall()]

    print(f"Found {len(db_episodes)} episodes in DB")

    if args.episode:
        db_episodes = [e for e in db_episodes
                       if args.episode.lower() in e["file_stem"].lower()]
        print(f"Filtered to {len(db_episodes)} matching episode(s)")

    # Process JSON files
    processed = 0
    skipped_existing = 0
    failed_match = 0
    failed_compute = 0
    written = 0

    # Collect all transcript files (json, srt, txt), deduplicate by stem
    # (prefer srt > json > txt if multiple formats exist for same episode)
    file_map: dict[str, Path] = {}
    for ext_priority, pattern in enumerate(["*.txt", "*.json", "*.srt"]):
        for fpath in transcript_dir.glob(pattern):
            if fpath.stem in ("manifest",):
                continue
            file_map[fpath.stem] = fpath  # later (higher priority) overwrites earlier

    all_files = list(file_map.values())
    print(f"Found {len(all_files)} unique transcript files to process")

    for fpath in all_files:
        # Find matching DB episode
        ep = None
        if fpath.suffix.lower() == ".json":
            ep = match_episode(fpath, db_episodes)
        if ep is None:
            # Fallback: match by slug — strip numbering and channel suffix
            stem_clean = re.sub(r"^\d{4}-", "", fpath.stem)
            stem_clean = re.sub(r"-armchair-expert.*$", "", stem_clean).replace("-", " ")
            for db_ep in db_episodes:
                if jaccard(stem_clean, db_ep["file_stem"]) >= 0.4:
                    ep = db_ep
                    break

        if ep is None:
            print(f"  ✗ No DB match: {fpath.name}")
            failed_match += 1
            continue

        if not args.force and (ep.get("youtube_offset_ms") or 0) != 0:
            skipped_existing += 1
            continue

        processed += 1
        print(f"\n  Processing: {ep['file_stem']}")
        print(f"    Transcript: {fpath.name}")

        # Load YouTube segments (auto-detects format)
        try:
            yt_segments = load_segments(fpath)
        except Exception as e:
            print(f"    ✗ Failed to load transcript: {e}")
            failed_compute += 1
            continue

        if not yt_segments:
            print(f"    ✗ Empty transcript")
            failed_compute += 1
            continue

        # Load DB utterances
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT speaker, start_ms, end_ms, text FROM utterances "
                "WHERE episode_id = %s ORDER BY start_ms",
                (ep["id"],)
            )
            utterances = [dict(r) for r in cur.fetchall()]

        if not utterances:
            print(f"    ✗ No utterances in DB")
            failed_compute += 1
            continue

        # Compute offset
        offset_ms = compute_offset(utterances, yt_segments, min_matches=args.min_matches)

        if offset_ms is None:
            print(f"    ✗ Could not compute reliable offset")
            failed_compute += 1
            continue

        print(f"    ✓ Offset: {offset_ms:+d}ms ({offset_ms/1000:+.1f}s)")

        if args.dry_run:
            print(f"    [dry-run] Would write youtube_offset_ms = {offset_ms} to DB")
        else:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE episodes SET youtube_offset_ms = %s WHERE id = %s",
                    (offset_ms, ep["id"])
                )
            conn.commit()
            print(f"    ✓ Written to DB")
            written += 1

    conn.close()

    print(f"\n{'=' * 50}")
    print(f"Done.")
    print(f"  Processed      : {processed}")
    print(f"  Written to DB  : {written}")
    print(f"  Already set    : {skipped_existing} (use --force to recompute)")
    print(f"  No DB match    : {failed_match}")
    print(f"  Compute failed : {failed_compute}")


if __name__ == "__main__":
    main()
