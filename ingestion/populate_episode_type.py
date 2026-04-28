#!/usr/bin/env python3
"""
Script to populate the episode_type column in the episodes table.
- 'wednesday' for expert episodes (EXPERTS ON EXPERT or with profession descriptors)
- 'monday' for celebrity episodes (simple names)
"""

import os
import re
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

def classify_episode(file_stem: str, file_path: str) -> str:
    """
    Classify episode as 'monday' (celebrity) or 'wednesday' (expert).

    Rules:
    1. If "EXPERTS ON EXPERT" appears in file_path or file_stem -> wednesday
    2. If file_stem contains parentheses (profession/topic descriptor) -> wednesday
    3. Otherwise -> monday
    """
    # Check for EXPERTS ON EXPERT
    if file_path and 'EXPERT' in file_path.upper() and file_path.upper().count('EXPERT') >= 2:
        return 'wednesday'

    if 'EXPERT' in file_stem.upper() and file_stem.upper().count('EXPERT') >= 2:
        return 'wednesday'

    # Check for parentheses indicating profession/topic
    if '(' in file_stem and ')' in file_stem:
        return 'wednesday'

    # Default to celebrity episode
    return 'monday'

def populate_episode_types(dry_run=True):
    """
    Populate the episode_type column for all episodes.

    Args:
        dry_run: If True, only show what would be updated without making changes
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Get all episodes
    cur.execute("SELECT id, file_stem, file_path FROM episodes")
    episodes = cur.fetchall()

    monday_count = 0
    wednesday_count = 0

    print(f"Processing {len(episodes)} episodes...")
    print(f"Mode: {'DRY RUN (no changes will be made)' if dry_run else 'LIVE UPDATE'}")
    print("-" * 80)

    updates = []

    for episode_id, file_stem, file_path in episodes:
        episode_type = classify_episode(file_stem, file_path or "")

        if episode_type == 'monday':
            monday_count += 1
        else:
            wednesday_count += 1

        updates.append((episode_type, episode_id))

    # Show summary
    print(f"\nClassification Summary:")
    print(f"  Monday (celebrity) episodes: {monday_count}")
    print(f"  Wednesday (expert) episodes: {wednesday_count}")
    print(f"  Total: {len(episodes)}")

    # Show some examples
    print("\nSample classifications:")
    print("\nMonday (celebrity) episodes:")
    cur.execute("""
        SELECT file_stem FROM episodes
        WHERE file_stem NOT LIKE '%(%'
        AND file_stem NOT ILIKE '%EXPERT%EXPERT%'
        LIMIT 5
    """)
    for (stem,) in cur.fetchall():
        print(f"  - {stem}")

    print("\nWednesday (expert) episodes:")
    cur.execute("""
        SELECT file_stem FROM episodes
        WHERE file_stem LIKE '%(%'
        OR file_stem ILIKE '%EXPERT%EXPERT%'
        LIMIT 5
    """)
    for (stem,) in cur.fetchall():
        print(f"  - {stem}")

    if not dry_run:
        print("\nUpdating database...")
        # Perform the updates
        cur.executemany(
            "UPDATE episodes SET episode_type = %s WHERE id = %s",
            updates
        )
        conn.commit()
        print(f"✓ Successfully updated {len(updates)} episodes")
    else:
        print("\n" + "=" * 80)
        print("DRY RUN COMPLETE - No changes made to database")
        print("Run with --live flag to apply these changes")
        print("=" * 80)

    cur.close()
    conn.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Populate episode_type column')
    parser.add_argument('--live', action='store_true',
                       help='Actually update the database (default is dry-run)')
    args = parser.parse_args()

    populate_episode_types(dry_run=not args.live)
