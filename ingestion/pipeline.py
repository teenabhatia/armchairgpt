#!/usr/bin/env python3
"""
pipeline.py — Armchair Expert Game Update Tool
================================================
Simple command-line tool for maintaining the game database.
Designed to be run by a non-technical person.

Commands:
    python pipeline.py weekly          Run the full weekly update (new episodes → DB → game)
    python pipeline.py backfill        Process all downloaded episodes not yet in DB
    python pipeline.py status          Show what's in the DB and what's pending
    python pipeline.py fix-timestamps  Re-match YouTube clip timestamps for all quotes
    python pipeline.py fix-quotes      Re-extract quotes for specific episodes

Run any command with --help for more options, e.g.:
    python pipeline.py weekly --help
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
DOWNLOADS_DIR = SCRIPT_DIR.parent / "rss" / "downloads"
VENV_PYTHON = SCRIPT_DIR / "venv" / "bin" / "python"


def _python():
    """Return the right python executable."""
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def _check_env():
    """Check required environment variables are set."""
    missing = []
    for key in ["ASSEMBLYAI_API_KEY", "GEMINI_API_KEY", "DATABASE_URL"]:
        if not os.environ.get(key):
            missing.append(key)
    if missing:
        print(f"\n❌  Missing required settings in .env file: {', '.join(missing)}")
        print("    Open .env and fill in the missing values, then try again.")
        sys.exit(1)


def cmd_status(args):
    """Show current database status."""
    import psycopg2
    from psycopg2.extras import RealDictCursor

    db_url = os.environ.get("DATABASE_URL", "")
    if "sslmode=" not in db_url:
        db_url += ("&" if "?" in db_url else "?") + "sslmode=require"

    try:
        conn = psycopg2.connect(db_url)
    except Exception as e:
        print(f"\n❌  Cannot connect to database: {e}")
        sys.exit(1)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM episodes")
        n_eps = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM mentions_llm")
        n_mentions = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM mentions_llm WHERE youtube_start_ms IS NOT NULL")
        n_yt_ts = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM mentions_llm WHERE youtube_start_ms IS NULL")
        n_no_ts = cur.fetchone()[0]

        cur.execute("SELECT file_stem FROM episodes ORDER BY id DESC LIMIT 5")
        recent = [r[0] for r in cur.fetchall()]

    conn.close()

    # Check downloads folder
    n_downloaded = 0
    if DOWNLOADS_DIR.exists():
        n_downloaded = len(list(DOWNLOADS_DIR.glob("*.mp3")) + list(DOWNLOADS_DIR.glob("*.m4a")))

    print("\n" + "=" * 55)
    print("  Armchair Expert Game — Database Status")
    print("=" * 55)
    print(f"  Episodes in database    : {n_eps:,}")
    print(f"  Audio files downloaded  : {n_downloaded:,}")
    print(f"  Quotes (total)          : {n_mentions:,}")
    print(f"  Quotes with YT timing   : {n_yt_ts:,}  ✓")
    print(f"  Quotes without YT timing: {n_no_ts:,}  (use 'fix-timestamps' to update)")
    print()
    print("  Most recently added episodes:")
    for ep in recent:
        print(f"    • {ep}")
    print("=" * 55 + "\n")


def cmd_weekly(args):
    """Run the full weekly update pipeline."""
    _check_env()

    print("\n" + "=" * 55)
    print("  Weekly Update — Starting")
    print("=" * 55)
    print("\nStep 1/3: Downloading new episodes from RSS feed...")

    # Download new RSS episodes
    rss_script = SCRIPT_DIR.parent / "rss" / "setup.py"
    if rss_script.exists():
        result = subprocess.run(
            [_python(), str(rss_script)],
            cwd=str(rss_script.parent)
        )
        if result.returncode != 0:
            print("⚠️  RSS download had errors — continuing anyway")
    else:
        print(f"⚠️  RSS script not found at {rss_script} — skipping download step")

    print("\nStep 2/3: Transcribing new episodes and extracting quotes...")
    result = subprocess.run([
        _python(), str(SCRIPT_DIR / "weekly_update.py"),
        "--days", str(args.days),
        *(["--dry-run"] if args.dry_run else []),
    ])

    print("\nStep 3/3: Updating YouTube clip timestamps...")
    subprocess.run([
        _python(), str(SCRIPT_DIR / "update_youtube_timestamps.py"),
        "--quiet",
        *(["--dry-run"] if args.dry_run else []),
    ])

    print("\n✅  Weekly update complete!")
    if args.dry_run:
        print("   (dry-run mode — nothing was written to the database)")


def cmd_backfill(args):
    """Process all downloaded episodes not yet in the database."""
    _check_env()

    if not DOWNLOADS_DIR.exists():
        print(f"\n❌  Downloads folder not found: {DOWNLOADS_DIR}")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("  Backfill — Processing unprocessed episodes")
    print("=" * 55)
    print(f"\nLooking in: {DOWNLOADS_DIR}")
    print("This may take many hours for a large backfill.\n")

    cmd = [
        _python(), str(SCRIPT_DIR / "transcribe_to_pgvector.py"),
        "--input-dir", str(DOWNLOADS_DIR),
        "--guest-csv", str(SCRIPT_DIR / "all_guests.csv"),
    ]
    if args.dry_run:
        print("DRY RUN — would run:\n  " + " ".join(cmd))
        return

    subprocess.run(cmd)

    print("\nUpdating YouTube clip timestamps for newly added episodes...")
    subprocess.run([
        _python(), str(SCRIPT_DIR / "update_youtube_timestamps.py"), "--quiet"
    ])

    print("\n✅  Backfill complete!")


def cmd_fix_timestamps(args):
    """Re-match YouTube clip timestamps for all (or specific) episodes."""
    _check_env()

    print("\n" + "=" * 55)
    print("  Fix YouTube Timestamps")
    print("=" * 55)

    cmd = [_python(), str(SCRIPT_DIR / "update_youtube_timestamps.py")]
    if args.episode:
        cmd += ["--episode", args.episode]
    if args.force:
        cmd += ["--force"]
    if args.dry_run:
        cmd += ["--dry-run"]

    subprocess.run(cmd)


def cmd_fix_quotes(args):
    """Re-run quote extraction for episodes (useful after prompt improvements)."""
    _check_env()

    print("\n" + "=" * 55)
    print("  Fix Quotes — Re-extracting")
    print("=" * 55)
    print("⚠️  This will delete existing quotes for the matched episodes and re-extract.")

    if not args.episode:
        confirm = input("\nThis will re-extract quotes for ALL episodes. Are you sure? (yes/no): ")
        if confirm.strip().lower() != "yes":
            print("Cancelled.")
            return

    # Use the transcribe script in reprocess mode for specific episodes
    cmd = [
        _python(), str(SCRIPT_DIR / "transcribe_to_pgvector.py"),
        "--input-dir", str(DOWNLOADS_DIR),
        "--guest-csv", str(SCRIPT_DIR / "all_guests.csv"),
        "--reprocess",
    ]
    if args.dry_run:
        print("DRY RUN — would run:\n  " + " ".join(cmd))
        return

    print("\nNote: Full reprocess re-transcribes audio which costs AssemblyAI credits.")
    print("For quote-only re-extraction, run extract_quotes_llm.py directly.")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Armchair Expert Game — Pipeline Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  status           Show database stats and recent episodes
  weekly           Run weekly update (RSS → transcribe → quotes → timestamps)
  backfill         Process all downloaded audio files not yet in the database
  fix-timestamps   Re-match YouTube clip timestamps for quotes
  fix-quotes       Re-extract quotes for episodes

Examples:
  python pipeline.py status
  python pipeline.py weekly --dry-run
  python pipeline.py weekly --days 14
  python pipeline.py backfill
  python pipeline.py fix-timestamps --episode "Jennifer Aniston"
  python pipeline.py fix-timestamps --force
        """
    )

    sub = parser.add_subparsers(dest="command")

    # status
    sub.add_parser("status", help="Show database stats")

    # weekly
    p_weekly = sub.add_parser("weekly", help="Run full weekly update")
    p_weekly.add_argument("--days", type=int, default=8, help="Look back N days for new episodes (default: 8)")
    p_weekly.add_argument("--dry-run", action="store_true", help="Show what would happen without making changes")

    # backfill
    p_back = sub.add_parser("backfill", help="Process unprocessed downloaded episodes")
    p_back.add_argument("--dry-run", action="store_true", help="Show what would happen without making changes")

    # fix-timestamps
    p_ts = sub.add_parser("fix-timestamps", help="Update YouTube clip timestamps")
    p_ts.add_argument("--episode", type=str, default="", help="Only fix a specific episode (partial name match)")
    p_ts.add_argument("--force", action="store_true", help="Recompute timestamps even if already set")
    p_ts.add_argument("--dry-run", action="store_true", help="Show matches without writing to DB")

    # fix-quotes
    p_quotes = sub.add_parser("fix-quotes", help="Re-extract quotes for episodes")
    p_quotes.add_argument("--episode", type=str, default="", help="Only fix a specific episode")
    p_quotes.add_argument("--dry-run", action="store_true", help="Show what would happen without making changes")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    load_dotenv()

    dispatch = {
        "status": cmd_status,
        "weekly": cmd_weekly,
        "backfill": cmd_backfill,
        "fix-timestamps": cmd_fix_timestamps,
        "fix-quotes": cmd_fix_quotes,
    }

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
