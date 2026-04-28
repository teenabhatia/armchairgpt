#!/usr/bin/env python3
"""
weekly_update.py

Weekly automation script for Armchair Expert game maintenance.

What it does:
  1. Reads the RSS feed to find episodes published in the last N days
  2. Skips episodes already in the database, Fact Checks, Armchair Anonymous
  3. Transcribes new episodes via AssemblyAI (audio URL from RSS, no download)
  4. Extracts guest-mention quotes via Gemini → inserts into mentions_llm
  5. Downloads the YouTube transcript for the new episode (one video, no rate limit)
  6. Computes the youtube_offset_ms and stores it in the episodes table

Usage:
    python weekly_update.py                      # process episodes from last 8 days
    python weekly_update.py --days 14            # look back further
    python weekly_update.py --dry-run            # show what would be processed, no DB writes
    python weekly_update.py --episode "Cher"     # force-process a specific episode by name

Secrets needed (in .env or GitHub Actions secrets):
    ASSEMBLYAI_API_KEY
    GEMINI_API_KEY
    DATABASE_URL
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
RSS_URL = "https://rss.art19.com/armchair-expert"
SKIP_TITLES_RE = re.compile(
    r"^\s*(armchair\s+anonymous|fact\s+check|best\s+of|rerelease)\b",
    re.IGNORECASE,
)
YOUTUBE_TRANSCRIPT_DIR = Path(__file__).parent / "youtube_transcripts"


# ──────────────────────────────────────────────
# RSS helpers
# ──────────────────────────────────────────────
def fetch_rss() -> ET.Element:
    req = urllib.request.Request(RSS_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return ET.fromstring(resp.read())


def rss_episodes(root: ET.Element, since: datetime) -> list[dict]:
    """Return episodes published after `since`, newest-first, skipping noise titles."""
    ns = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}
    items = []
    for item in root.iter("item"):
        title_el = item.find("title")
        title = (title_el.text or "").strip() if title_el is not None else ""
        if SKIP_TITLES_RE.search(title):
            continue

        pub_el = item.find("pubDate")
        if pub_el is None or not pub_el.text:
            continue
        try:
            pub_date = parsedate_to_datetime(pub_el.text.strip())
            if pub_date.tzinfo is None:
                pub_date = pub_date.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if pub_date < since:
            continue

        enclosure = item.find("enclosure")
        audio_url = enclosure.get("url", "") if enclosure is not None else ""
        if not audio_url:
            continue

        duration_el = item.find("itunes:duration", ns)
        duration_str = (duration_el.text or "").strip() if duration_el is not None else ""

        items.append({
            "title": title,
            "pub_date": pub_date,
            "audio_url": audio_url,
            "duration": duration_str,
        })

    items.sort(key=lambda x: x["pub_date"], reverse=True)
    return items


def safe_stem(title: str) -> str:
    """Convert RSS title to a clean file_stem for DB lookup."""
    # Strip episode numbers and suffixes like "| Armchair Expert with Dax Shepard"
    s = re.sub(r"\s*\|\s*armchair expert.*$", "", title, flags=re.IGNORECASE).strip()
    s = re.sub(r"^\d+[\s\-–:]+", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s or title


# ──────────────────────────────────────────────
# YouTube transcript helpers
# ──────────────────────────────────────────────
def fetch_youtube_transcript_for_episode(video_id: str) -> Optional[list[dict]]:
    """Fetch transcript for a single YouTube video using youtube-transcript-api.
    One video at a time = no rate limiting / IP blocks."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=["en"])
        rows = []
        for item in transcript:
            rows.append({
                "text": item.text.replace("\n", " ").strip(),
                "start": float(item.start),
                "duration": float(item.duration),
            })
        return rows
    except Exception as e:
        print(f"  [YouTube] Could not fetch transcript for {video_id}: {e}")
        return None


def find_youtube_video_id(episode_title: str, pub_date: datetime) -> Optional[str]:
    """Search YouTube channel for a video matching this episode title.
    Uses yt-dlp flat-playlist search."""
    try:
        import subprocess, json as _json, shutil

        yt = shutil.which("yt-dlp") or "yt-dlp"
        # Search channel videos published around the pub_date
        result = subprocess.run(
            [yt, "--flat-playlist", "--dump-single-json", "--playlist-end", "20",
             "https://www.youtube.com/@armchairexpertpod/videos"],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            return None

        data = _json.loads(result.stdout)
        entries = data.get("entries", [])

        # Fuzzy match title
        title_words = set(re.findall(r"\w+", episode_title.lower()))
        best_id, best_score = None, 0
        for entry in entries:
            yt_title = entry.get("title", "").lower()
            yt_words = set(re.findall(r"\w+", yt_title))
            overlap = len(title_words & yt_words)
            score = overlap / max(len(title_words), 1)
            if score > best_score and score >= 0.5:
                best_score = score
                best_id = entry.get("id")

        return best_id
    except Exception as e:
        print(f"  [YouTube] Could not find video ID: {e}")
        return None


def compute_offset_from_transcript(db_utterances: list[dict],
                                   yt_segments: list[dict]) -> Optional[int]:
    """
    Align DB utterance timestamps against YouTube transcript timestamps.

    Strategy: find 5+ long utterances whose text appears in the YouTube transcript,
    compute the time delta for each, take the median.

    Returns offset_ms such that: youtube_time_ms = db_time_ms + offset_ms
    """
    import statistics

    # Build searchable YouTube text with timestamps
    # Each segment: {"text": ..., "start": seconds, "duration": seconds}
    yt_text_lower = " ".join(s["text"].lower() for s in yt_segments)

    deltas = []
    for utt in db_utterances:
        text = (utt.get("text") or "").strip()
        if len(text) < 60:
            continue  # too short to match reliably

        # Extract key phrase: first 8 words, skip common words
        words = [w for w in text.lower().split() if len(w) > 4][:6]
        if len(words) < 3:
            continue
        phrase = " ".join(words[:4])

        # Find phrase in YouTube segments
        for i, seg in enumerate(yt_segments):
            combined = " ".join(s["text"].lower() for s in yt_segments[i:i+4])
            if all(w in combined for w in words[:3]):
                db_ms = utt.get("start_ms") or 0
                yt_ms = int(seg["start"] * 1000)
                delta = yt_ms - db_ms
                # Sanity: offset should be consistent, reject wild outliers
                if abs(delta) < 600_000:  # within 10 minutes
                    deltas.append(delta)
                break

    if len(deltas) < 3:
        print(f"  [offset] Too few matches ({len(deltas)}) to compute reliable offset")
        return None

    median_delta = int(statistics.median(deltas))
    mad = statistics.median([abs(d - median_delta) for d in deltas])
    # Filter outliers (>2x MAD from median)
    clean = [d for d in deltas if abs(d - median_delta) <= max(mad * 2, 5000)]
    if len(clean) < 3:
        print(f"  [offset] Too many outliers, skipping")
        return None

    final_offset = int(statistics.median(clean))
    print(f"  [offset] Computed from {len(clean)} matches: {final_offset}ms ({final_offset/1000:.1f}s)")
    return final_offset


def store_youtube_offset(episode_id: int, offset_ms: int, db_conn) -> None:
    with db_conn.cursor() as cur:
        cur.execute(
            "UPDATE episodes SET youtube_offset_ms = %s WHERE id = %s",
            (offset_ms, episode_id)
        )
    db_conn.commit()


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────
def process_new_episode(episode: dict, dry_run: bool, conn) -> bool:
    """Full pipeline for one new episode. Returns True if successfully processed."""
    title = episode["title"]
    audio_url = episode["audio_url"]
    stem = safe_stem(title)

    print(f"\n{'═' * 60}")
    print(f"New episode: {title}")
    print(f"  Stem    : {stem}")
    print(f"  Audio   : {audio_url[:80]}...")
    print(f"  PubDate : {episode['pub_date'].strftime('%Y-%m-%d')}")

    if dry_run:
        print("  [dry-run] Would transcribe and extract quotes.")
        return True

    # ── Step 1: Transcribe via AssemblyAI ──
    print("\n  Step 1/4: Transcribing via AssemblyAI...")
    try:
        from transcribe_to_pgvector import (
            process_one_from_url, build_catalog_from_db, _dedupe_names
        )
        global_catalog = _dedupe_names(build_catalog_from_db())
        result = process_one_from_url(
            stem=stem,
            audio_url=audio_url,
            global_guest_catalog=global_catalog,
        )
        if result.get("skipped"):
            print(f"  Skipped: {result.get('reason')}")
            return False
        episode_id = result["episode_id"]
        print(f"  ✓ Transcribed: {result['utterances']} utterances, episode_id={episode_id}")
    except Exception as e:
        print(f"  ✗ Transcription failed: {e}")
        return False

    # ── Step 2: Extract quotes via Gemini ──
    print("\n  Step 2/4: Extracting quotes via Gemini...")
    try:
        from psycopg2.extras import RealDictCursor
        import extract_quotes_llm as eq

        eq.GUEST_CATALOG = eq.load_guest_catalog(eq.GUEST_LIST_PATH)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, file_stem, guests FROM episodes WHERE id = %s",
                (episode_id,)
            )
            ep_row = cur.fetchone()

        if ep_row:
            eq.process_episode(conn, dict(ep_row), provider="gemini",
                               model_name="gemini-2.5-flash", dry_run=False)
            print(f"  ✓ Quote extraction complete")
        else:
            print(f"  ✗ Episode row not found after transcription")
    except Exception as e:
        print(f"  ✗ Quote extraction failed: {e}")
        # Non-fatal — transcription succeeded, quotes can be re-run manually

    # ── Step 3: Download YouTube subtitle file via yt-dlp ──
    print("\n  Step 3/4: Downloading YouTube subtitles for clip timing...")
    srt_path = None
    video_id = find_youtube_video_id(title, episode["pub_date"])
    if video_id:
        print(f"  Found YouTube video: {video_id}")
        srt_path = _download_srt_for_video(video_id, stem)
        if srt_path:
            print(f"  ✓ Downloaded subtitles: {srt_path.name}")
        else:
            print("  Could not download subtitles — clip timing will use DB timestamps")
    else:
        print("  Could not find YouTube video — clip timing will use DB timestamps")

    # ── Step 4: Match quotes to YouTube timestamps ──
    if srt_path and srt_path.exists():
        print("\n  Step 4/4: Matching quotes to YouTube timestamps...")
        try:
            from update_youtube_timestamps import load_srt, find_quote_in_srt, ensure_columns, write_timestamps
            from psycopg2.extras import RealDictCursor

            ensure_columns(conn)
            segments = load_srt(srt_path)

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, quote, start_ms, end_ms FROM mentions_llm WHERE episode_id = %s",
                    (episode_id,)
                )
                mentions = cur.fetchall()

            updates = []
            for m in mentions:
                result = find_quote_in_srt(m["quote"], m["start_ms"] or 0, segments)
                if result:
                    updates.append((result[0], result[1], m["id"]))

            write_timestamps(conn, updates, dry_run=False)
            print(f"  ✓ Matched {len(updates)}/{len(mentions)} quotes to YouTube timestamps")
        except Exception as e:
            print(f"  ✗ YouTube timestamp matching failed: {e}")
    else:
        print("\n  Step 4/4: Skipped (no YouTube subtitles)")

    return True


def _download_srt_for_video(video_id: str, stem: str) -> Optional[Path]:
    """Download auto-generated subtitles for a single YouTube video using yt-dlp.
    Returns the path to the downloaded SRT file, or None on failure."""
    import subprocess
    import tempfile

    out_dir = Path(__file__).parent / "youtube_transcripts"
    out_dir.mkdir(exist_ok=True)

    # Clean stem for filename
    safe = re.sub(r"[^\w\s\-]", "", stem).strip()[:80]
    out_template = str(out_dir / f"{safe}.%(ext)s")

    try:
        result = subprocess.run([
            "yt-dlp",
            f"https://www.youtube.com/watch?v={video_id}",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--sub-format", "srt",
            "--skip-download",
            "--convert-subs", "srt",
            "-o", out_template,
            "--quiet",
        ], capture_output=True, text=True, timeout=60)

        # yt-dlp names the file like "stem.en.srt"
        candidates = list(out_dir.glob(f"{safe}*.srt"))
        return candidates[0] if candidates else None
    except Exception as e:
        print(f"  yt-dlp error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Weekly Armchair Expert game update")
    parser.add_argument("--days", type=int, default=8,
                        help="Look back this many days for new episodes (default: 8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without writing to DB")
    parser.add_argument("--episode", type=str, default="",
                        help="Force-process a specific episode by title substring")
    args = parser.parse_args()

    print("=" * 60)
    print("Armchair Expert — Weekly Update")
    print(f"Looking back {args.days} days | dry_run={args.dry_run}")
    print("=" * 60)

    # ── Check required env vars ──
    missing = [k for k in ("ASSEMBLYAI_API_KEY", "GEMINI_API_KEY", "DATABASE_URL")
               if not os.environ.get(k)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    # ── Fetch RSS ──
    print("\nFetching RSS feed...")
    try:
        root = fetch_rss()
    except Exception as e:
        print(f"ERROR: Could not fetch RSS: {e}")
        sys.exit(1)

    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    episodes = rss_episodes(root, since)

    if args.episode:
        episodes = [e for e in episodes
                    if args.episode.lower() in e["title"].lower()]

    if not episodes:
        print(f"No new episodes found in the last {args.days} days.")
        return

    print(f"Found {len(episodes)} new episode(s):")
    for ep in episodes:
        print(f"  • {ep['title']} ({ep['pub_date'].strftime('%Y-%m-%d')})")

    if args.dry_run:
        print("\n[dry-run] Stopping here. Remove --dry-run to process.")
        return

    # ── Connect to DB ──
    import psycopg2
    conn = psycopg2.connect(os.environ["DATABASE_URL"])

    # ── Check which episodes are already in DB ──
    with conn.cursor() as cur:
        cur.execute("SELECT file_stem FROM episodes")
        existing_stems = {r[0].lower() for r in cur.fetchall()}

    new_episodes = []
    for ep in episodes:
        stem = safe_stem(ep["title"])
        if stem.lower() in existing_stems:
            print(f"\nSkipping (already in DB): {ep['title']}")
        else:
            new_episodes.append(ep)

    if not new_episodes:
        print("\nAll found episodes already in database. Nothing to do.")
        conn.close()
        return

    # ── Process each new episode ──
    success, failed = 0, 0
    for ep in new_episodes:
        ok = process_new_episode(ep, dry_run=args.dry_run, conn=conn)
        if ok:
            success += 1
        else:
            failed += 1

    conn.close()

    print(f"\n{'=' * 60}")
    print(f"Done. {success} episode(s) processed, {failed} failed.")
    if failed:
        sys.exit(1)  # non-zero exit triggers GitHub Actions failure notification


if __name__ == "__main__":
    main()
