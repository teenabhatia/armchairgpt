#!/usr/bin/env python3
"""
Script to remove ad mentions from the mentions table.
Removes quotes containing phrases like "we are supported by" which are advertisements.
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    """Get database connection using environment variables."""
    db_url = os.environ.get('DATABASE_URL')
    if db_url:
        return psycopg2.connect(db_url)
    else:
        conn_params = {
            'host': os.environ.get('PGHOST'),
            'database': os.environ.get('PGDATABASE'),
            'user': os.environ.get('PGUSER'),
            'password': os.environ.get('PGPASSWORD'),
            'port': os.environ.get('PGPORT', 5432)
        }
        return psycopg2.connect(**conn_params)

def remove_ad_mentions(dry_run=True):
    """
    Remove ad mentions from the mentions table.

    Args:
        dry_run: If True, only show what would be deleted without making changes
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Patterns to identify ads
    ad_patterns = [
        '%we are supported by%',
        '%this episode is brought to you%',
        '%this podcast is sponsored%',
        '%brought to you by%',
        '%sponsored by%'
    ]

    print("=" * 80)
    print("Ad Mention Removal Tool")
    print("=" * 80)
    print(f"Mode: {'DRY RUN (no changes will be made)' if dry_run else 'LIVE DELETE'}")
    print()

    # Get total count before
    cur.execute("SELECT COUNT(*) FROM mentions")
    total_before = cur.fetchone()[0]
    print(f"Total mentions before: {total_before}")
    print()

    # Find all ad mentions
    all_ad_ids = set()
    print("Searching for ad patterns...")
    print("-" * 80)

    for pattern in ad_patterns:
        cur.execute("""
            SELECT id FROM mentions WHERE quote ILIKE %s
        """, (pattern,))
        ids = [row[0] for row in cur.fetchall()]
        if ids:
            all_ad_ids.update(ids)
            print(f"Pattern '{pattern}': {len(ids)} mentions")

    print("-" * 80)
    print(f"Total ad mentions to remove: {len(all_ad_ids)}")
    print()

    if len(all_ad_ids) == 0:
        print("✓ No ad mentions found!")
        cur.close()
        conn.close()
        return

    # Show some examples
    print("\nSample mentions to be removed:")
    print("-" * 80)
    cur.execute("""
        SELECT id, speaker, about_person, quote
        FROM mentions
        WHERE id IN %s
        LIMIT 5
    """, (tuple(all_ad_ids),))

    for mention_id, speaker, about_person, quote in cur.fetchall():
        print(f"ID: {mention_id}")
        print(f"Speaker: {speaker}")
        print(f"About: {about_person}")
        preview = quote[:150] + "..." if len(quote) > 150 else quote
        print(f"Quote: {preview}")
        print()

    if not dry_run:
        print("Deleting ad mentions...")
        cur.execute("""
            DELETE FROM mentions WHERE id IN %s
        """, (tuple(all_ad_ids),))
        conn.commit()

        # Get count after
        cur.execute("SELECT COUNT(*) FROM mentions")
        total_after = cur.fetchone()[0]

        print("=" * 80)
        print(f"✓ Successfully deleted {len(all_ad_ids)} ad mentions")
        print(f"Total mentions after: {total_after}")
        print(f"Remaining mentions: {total_after}")
        print("=" * 80)
    else:
        print("=" * 80)
        print("DRY RUN COMPLETE - No changes made to database")
        print(f"Would delete: {len(all_ad_ids)} mentions")
        print(f"Would remain: {total_before - len(all_ad_ids)} mentions")
        print("Run with --live flag to delete these mentions")
        print("=" * 80)

    cur.close()
    conn.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Remove ad mentions from mentions table')
    parser.add_argument('--live', action='store_true',
                       help='Actually delete from database (default is dry-run)')
    args = parser.parse_args()

    remove_ad_mentions(dry_run=not args.live)
