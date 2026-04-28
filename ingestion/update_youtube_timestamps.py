#!/usr/bin/env python3
"""
update_youtube_timestamps.py

For every quote in mentions_llm, find the matching text in the episode's YouTube
SRT transcript and store the exact YouTube timestamp as youtube_start_ms / youtube_end_ms.

This is more accurate than a single per-episode offset because each quote gets its
own timestamp matched directly from the YouTube auto-captions.

Usage:
    python update_youtube_timestamps.py              # process all episodes
    python update_youtube_timestamps.py --dry-run    # print matches without writing
    python update_youtube_timestamps.py --episode "Jennifer Aniston"
    python update_youtube_timestamps.py --force      # recompute already-set timestamps
    python update_youtube_timestamps.py --srt path/to/file.srt --episode "Name"

How it works:
    1. For each episode that has a matching SRT file in youtube_transcripts/,
       load all its mentions from mentions_llm.
    2. For each mention quote, extract the first ~8 distinctive keywords.
    3. Search through the SRT segments (combining consecutive ones) for the
       best keyword match.
    4. Store (srt_start_ms - PADDING_MS) and (srt_end_ms + PADDING_MS) as the
       YouTube clip window.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from dotenv import load_dotenv

load_dotenv()

TRANSCRIPT_DIR = Path(__file__).parent / "youtube_transcripts"
PADDING_MS = 15_000          # ±15 seconds around matched quote
MIN_KEYWORD_HITS = 3         # minimum word matches to accept a SRT position
SEARCH_WINDOW_MS = 600_000   # ±10 min from DB timestamp when searching SRT


# ──────────────────────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────────────────────

def get_conn():
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        if "sslmode=" not in db_url:
            db_url += ("&" if "?" in db_url else "?") + "sslmode=require"
        return psycopg2.connect(db_url)
    raise RuntimeError("Missing DATABASE_URL in .env")


def ensure_columns(conn):
    """Add youtube_start_ms / youtube_end_ms to mentions_llm if not present."""
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE mentions_llm
            ADD COLUMN IF NOT EXISTS youtube_start_ms INTEGER,
            ADD COLUMN IF NOT EXISTS youtube_end_ms   INTEGER
        """)
    conn.commit()


def fetch_episodes(conn, episode_filter: str | None = None) -> list[dict]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if episode_filter:
            cur.execute(
                "SELECT id, file_stem FROM episodes WHERE file_stem ILIKE %s ORDER BY file_stem",
                (f"%{episode_filter}%",)
            )
        else:
            cur.execute("SELECT id, file_stem FROM episodes ORDER BY file_stem")
        return cur.fetchall()


def fetch_mentions(conn, episode_id: int, force: bool = False) -> list[dict]:
    """Fetch mentions for an episode. If not force, skip ones already stamped."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if force:
            cur.execute(
                "SELECT id, quote, start_ms, end_ms FROM mentions_llm WHERE episode_id = %s",
                (episode_id,)
            )
        else:
            cur.execute(
                """SELECT id, quote, start_ms, end_ms FROM mentions_llm
                   WHERE episode_id = %s AND youtube_start_ms IS NULL""",
                (episode_id,)
            )
        return cur.fetchall()


def write_timestamps(conn, updates: list[tuple[int, int, int]], dry_run: bool):
    """Write (youtube_start_ms, youtube_end_ms, mention_id) tuples to DB."""
    if dry_run or not updates:
        return
    with conn.cursor() as cur:
        execute_values(cur, """
            UPDATE mentions_llm AS m SET
                youtube_start_ms = v.ys,
                youtube_end_ms   = v.ye
            FROM (VALUES %s) AS v(ys, ye, id)
            WHERE m.id = v.id
        """, updates, template="(%s, %s, %s)")
    conn.commit()


# ──────────────────────────────────────────────────────────────
# SRT loading
# ──────────────────────────────────────────────────────────────

def _srt_ts_to_ms(ts: str) -> int:
    ts = ts.strip().replace(",", ".")
    m = re.match(r"(\d+):(\d{2}):(\d{2})\.(\d+)", ts)
    if not m:
        return 0
    h, mi, s, frac = m.groups()
    return (int(h) * 3600 + int(mi) * 60 + int(s)) * 1000 + int(frac[:3].ljust(3, "0"))


def load_srt(path: Path) -> list[dict]:
    """Return list of {start_ms, end_ms, text} dicts."""
    segments = []
    content = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        m = re.match(r"([\d:,\.]+)\s*-->\s*([\d:,\.]+)", lines[1])
        if not m:
            continue
        start_ms = _srt_ts_to_ms(m.group(1))
        end_ms   = _srt_ts_to_ms(m.group(2))
        text = " ".join(lines[2:])
        text = re.sub(r"<[^>]+>", "", text).strip()
        if text:
            segments.append({"start_ms": start_ms, "end_ms": end_ms, "text": text})
    return segments


# ──────────────────────────────────────────────────────────────
# Title matching (SRT filename → DB episode)
# ──────────────────────────────────────────────────────────────

def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def match_srt_to_episode(srt_path: Path, episodes: list[dict]) -> dict | None:
    """Find best DB episode for an SRT filename."""
    # SRT filename example: "Jennifer Aniston Armchair Expert with Dax Shepard.srt"
    name = srt_path.stem
    # Strip common suffix
    name = re.sub(r"\s+armchair\s+expert.*$", "", name, flags=re.IGNORECASE).strip()
    name = re.sub(r"^\d+[\s\-–:]+", "", name).strip()

    best_ep, best_score = None, 0.0
    for ep in episodes:
        score = jaccard(name, ep["file_stem"])
        if score > best_score:
            best_score, best_ep = score, ep

    return best_ep if best_score >= 0.35 else None


# ──────────────────────────────────────────────────────────────
# Quote → SRT timestamp matching
# ──────────────────────────────────────────────────────────────

def _keywords(text: str, n: int = 8) -> list[str]:
    """Extract the first N distinctive words (length > 4) from a quote."""
    words = re.findall(r"[a-z']+", text.lower())
    stop = {"that", "this", "with", "from", "they", "their", "there", "would",
            "could", "should", "about", "really", "think", "going", "because",
            "which", "where", "these", "those", "have", "just", "like", "know",
            "dont", "wasnt", "didnt", "youre", "were", "when", "what", "into"}
    return [w for w in words if len(w) > 4 and w not in stop][:n]


def find_quote_in_srt(
    quote: str,
    start_ms: int,
    segments: list[dict],
    padding_ms: int = PADDING_MS,
    min_hits: int = MIN_KEYWORD_HITS,
    window_ms: int = SEARCH_WINDOW_MS,
) -> tuple[int, int] | None:
    """
    Find where a quote appears in the SRT and return (youtube_start_ms, youtube_end_ms).
    Returns None if no reliable match found.
    """
    keywords = _keywords(quote)
    if len(keywords) < min_hits:
        return None

    # Build a search window around the DB timestamp (if available)
    if start_ms and start_ms > 0:
        candidates = [s for s in segments if abs(s["start_ms"] - start_ms) < window_ms]
        if not candidates:
            candidates = segments  # full scan fallback
    else:
        candidates = segments

    best_hits = 0
    best_start = None
    best_end = None

    # Slide a window of ~5 consecutive SRT segments
    for i in range(len(candidates)):
        window_segs = candidates[i:i + 5]
        combined = " ".join(s["text"] for s in window_segs).lower()
        hits = sum(1 for kw in keywords if kw in combined)
        if hits > best_hits:
            best_hits = hits
            best_start = window_segs[0]["start_ms"]
            best_end   = window_segs[-1]["end_ms"]

    if best_hits < min_hits or best_start is None:
        return None

    yt_start = max(0, best_start - padding_ms)
    yt_end   = best_end + padding_ms
    return yt_start, yt_end


# ──────────────────────────────────────────────────────────────
# Main processing
# ──────────────────────────────────────────────────────────────

def process_episode(
    conn,
    episode: dict,
    srt_path: Path,
    dry_run: bool,
    force: bool,
    verbose: bool = True,
) -> tuple[int, int]:
    """Process one episode. Returns (matched, total) mention counts."""
    mentions = fetch_mentions(conn, episode["id"], force=force)
    if not mentions:
        return 0, 0

    segments = load_srt(srt_path)
    if not segments:
        if verbose:
            print(f"    ✗ Empty SRT: {srt_path.name}")
        return 0, len(mentions)

    updates = []
    matched = 0
    for mention in mentions:
        result = find_quote_in_srt(
            mention["quote"],
            mention["start_ms"] or 0,
            segments,
        )
        if result:
            yt_start, yt_end = result
            updates.append((yt_start, yt_end, mention["id"]))
            matched += 1

    write_timestamps(conn, updates, dry_run)
    return matched, len(mentions)


def run(args):
    conn = get_conn()
    ensure_columns(conn)

    episodes = fetch_episodes(conn, args.episode)
    if not episodes:
        print("No episodes found.")
        return

    # Build episode lookup by file_stem (lowercase)
    ep_by_stem = {ep["file_stem"].lower(): ep for ep in episodes}

    # Find all SRT files
    srt_files = sorted(TRANSCRIPT_DIR.glob("*.srt"))
    if not srt_files:
        print(f"No SRT files found in {TRANSCRIPT_DIR}")
        return

    total_matched = 0
    total_mentions = 0
    processed_eps = 0
    no_srt = 0

    for ep in episodes:
        # Find matching SRT for this episode
        best_srt = None
        best_score = 0.0
        for srt in srt_files:
            srt_name = re.sub(r"\s+armchair\s+expert.*$", "", srt.stem, flags=re.IGNORECASE).strip()
            srt_name = re.sub(r"^\d+[\s\-–:]+", "", srt_name).strip()
            score = jaccard(srt_name, ep["file_stem"])
            if score > best_score:
                best_score, best_srt = score, srt

        if best_score < 0.35 or best_srt is None:
            no_srt += 1
            continue

        if args.verbose:
            print(f"\n  {ep['file_stem'][:55]}")
            print(f"    SRT: {best_srt.name[:55]}")

        matched, total = process_episode(conn, ep, best_srt, args.dry_run, args.force, args.verbose)
        if total == 0:
            continue

        processed_eps += 1
        total_matched += matched
        total_mentions += total

        status = f"✓ {matched}/{total} quotes matched"
        if args.dry_run:
            status += " [dry-run]"
        if args.verbose:
            print(f"    {status}")
        else:
            print(f"  {ep['file_stem'][:50]}: {status}")

    conn.close()
    print(f"\n{'='*50}")
    print(f"Episodes processed : {processed_eps}")
    print(f"Episodes without SRT: {no_srt}")
    print(f"Quotes matched     : {total_matched}/{total_mentions}")
    if args.dry_run:
        print("DRY RUN — nothing written to DB")


def main():
    parser = argparse.ArgumentParser(description="Match mention quotes to YouTube SRT timestamps")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing to DB")
    parser.add_argument("--force", action="store_true", help="Recompute already-set timestamps")
    parser.add_argument("--episode", type=str, default="", help="Filter to specific episode name")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    args = parser.parse_args()
    if args.quiet:
        args.verbose = False
    run(args)


if __name__ == "__main__":
    main()
