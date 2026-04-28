#!/usr/bin/env python3
"""
Match YouTube URLs from Excel to episodes in the database using file_stem.
Handles repeat guests (Returns, Part 2, etc.)
"""

import os
import re
import psycopg2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import difflib
from dotenv import load_dotenv
import openpyxl

load_dotenv()

# -----------------------
# Database Connection
# -----------------------
def with_conn():
    """Database connection."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        host = os.environ.get("PGHOST")
        db = os.environ.get("PGDATABASE")
        user = os.environ.get("PGUSER")
        password = os.environ.get("PGPASSWORD")
        port = int(os.environ.get("PGPORT", "5432"))
        
        if not all([host, db, user, password]):
            raise RuntimeError("Missing database configuration.")
        
        return psycopg2.connect(
            host=host,
            dbname=db,
            user=user,
            password=password,
            port=port,
            sslmode="require"
        )
    
    if "sslmode=" not in db_url:
        separator = "&" if "?" in db_url else "?"
        db_url = f"{db_url}{separator}sslmode=require"
    
    return psycopg2.connect(db_url)

# -----------------------
# Check Existing Schema
# -----------------------
def check_schema():
    """Check what columns exist in episodes table."""
    with with_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'episodes'
            ORDER BY ordinal_position;
        """)
        columns = [row[0] for row in cur.fetchall()]
        print(f"✓ Episodes table has columns: {', '.join(columns)}")
        return columns

# -----------------------
# Load YouTube Data
# -----------------------
def load_youtube_data(excel_path: str) -> List[Dict]:
    """Load YouTube URLs and titles from Excel file."""
    workbook = openpyxl.load_workbook(excel_path)
    sheet = workbook.active
    
    youtube_videos = []
    
    # Skip header row
    for row in sheet.iter_rows(min_row=2, values_only=True):
        if len(row) >= 2 and row[1]:  # Has URL
            title = str(row[0]).strip() if row[0] else ""
            url = str(row[1]).strip() if row[1] else ""
            
            if url.startswith("http"):
                youtube_videos.append({
                    "title": title,
                    "url": url
                })
    
    print(f"✓ Loaded {len(youtube_videos)} YouTube URLs from Excel")
    return youtube_videos

# -----------------------
# Title Matching Logic
# -----------------------
def extract_repeat_indicator(text: str) -> Optional[str]:
    """Extract repeat indicators like 'Returns', 'Part 2', etc."""
    # Look for common repeat patterns
    patterns = [
        r'\breturns?\b',
        r'\bpart\s*(\d+)\b',
        r'\bpt\.?\s*(\d+)\b',
        r'\b(second|third|fourth)\s+time\b',
        r'\bagain\b',
        r'\bback\b',
    ]
    
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(0)
    
    return None

def normalize_for_matching(text: str, keep_repeat_indicators: bool = True) -> str:
    """Normalize text for matching - optionally keep repeat indicators."""
    original = text
    
    # Remove common suffixes
    text = re.sub(r'\s*\|\s*Armchair Expert.*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*with Dax Shepard.*$', '', text, flags=re.IGNORECASE)
    
    # Remove "Fact Check for X" pattern
    text = re.sub(r'.*?\|\s*Fact Check for\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\|\s*Fact Check.*$', '', text, flags=re.IGNORECASE)
    
    # Remove episode numbers at start (like "123 - ")
    text = re.sub(r'^\s*\d+\s*[-–—]\s*', '', text)
    
    # Store repeat indicators before removing parentheses
    repeat_indicator = None
    if keep_repeat_indicators:
        repeat_indicator = extract_repeat_indicator(text)
    
    # Remove parenthetical descriptions but keep the main name
    text = re.sub(r'\s*\([^)]+\)\s*', ' ', text)
    
    # If we want to remove repeat indicators for matching
    if not keep_repeat_indicators:
        text = re.sub(r'\breturns?\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bpart\s*\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bpt\.?\s*\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(second|third|fourth)\s+time\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bagain\b', '', text, flags=re.IGNORECASE)
    
    # Normalize whitespace and lowercase
    text = re.sub(r'\s+', ' ', text).strip().lower()
    
    if keep_repeat_indicators and repeat_indicator:
        text = f"{text} {repeat_indicator.lower()}"
    
    return text

def extract_guest_name_from_youtube(youtube_title: str) -> Optional[str]:
    """Extract guest name from YouTube title."""
    # Pattern for fact checks: "Something | Fact Check for Guest Name"
    match = re.search(r'\|\s*Fact Check for\s+([^|]+)$', youtube_title, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Pattern for main episodes: "Guest Name | Armchair Expert"
    match = re.match(r'^([^|]+?)\s*\|', youtube_title)
    if match:
        name = match.group(1).strip()
        # Remove parenthetical descriptions
        name = re.sub(r'\s*\([^)]+\)', '', name).strip()
        return name
    
    return None

def is_fact_check(youtube_title: str) -> bool:
    """Determine if this is a fact check episode."""
    return bool(re.search(r'\bfact\s+check\b', youtube_title, re.IGNORECASE))

def is_armchair_anonymous(youtube_title: str) -> bool:
    """Determine if this is an Armchair Anonymous episode."""
    return bool(re.search(r'\barmchair\s+anonymous\b', youtube_title, re.IGNORECASE))

def is_moms_car(youtube_title: str) -> bool:
    """Determine if this is a Mom's Car segment."""
    return bool(re.search(r"\bmom'?s?\s+car\b", youtube_title, re.IGNORECASE))

# -----------------------
# Episode Matching
# -----------------------
def get_all_episodes():
    """Get all episodes from database with file_stem."""
    with with_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, file_stem, guests, youtube_url 
            FROM episodes 
            ORDER BY file_stem
        """)
        
        episodes = []
        for row in cur.fetchall():
            episode_id, file_stem, guests_json, existing_url = row
            
            # Parse guests from JSON
            import json
            guests = []
            if guests_json:
                try:
                    guests = json.loads(guests_json)
                except:
                    pass
            
            episodes.append({
                "id": episode_id,
                "file_stem": file_stem,
                "guests": guests,
                "normalized_stem": normalize_for_matching(file_stem, keep_repeat_indicators=True),
                "normalized_stem_no_repeat": normalize_for_matching(file_stem, keep_repeat_indicators=False),
                "existing_url": existing_url,
                "has_repeat_indicator": extract_repeat_indicator(file_stem) is not None
            })
        
        print(f"✓ Loaded {len(episodes)} episodes from database")
        return episodes

def match_youtube_to_episode(youtube_video: Dict, episodes: List[Dict]) -> Tuple[Optional[int], bool, float, str]:
    """
    Match a YouTube video to a database episode using file_stem.
    Handles repeat guests by matching both name AND repeat indicators.
    Returns: (episode_id, is_fact_check, confidence_score, match_method)
    """
    youtube_title = youtube_video["title"]
    
    # Skip certain types
    if is_moms_car(youtube_title):
        return None, False, 0.0, "skipped_moms_car"
    
    # Check if it's a fact check
    is_fc = is_fact_check(youtube_title)
    
    # Extract guest name from YouTube title
    guest_name = extract_guest_name_from_youtube(youtube_title)
    if not guest_name:
        return None, is_fc, 0.0, "no_guest_name"
    
    # Check if YouTube title has repeat indicator
    youtube_repeat = extract_repeat_indicator(youtube_title)
    
    # Get normalized versions
    guest_name_normalized = normalize_for_matching(guest_name, keep_repeat_indicators=True)
    guest_name_no_repeat = normalize_for_matching(guest_name, keep_repeat_indicators=False)
    
    # Try to match against file_stem
    best_match = None
    best_score = 0.0
    best_method = "none"
    
    # Find all episodes with this guest (by name only, ignoring repeat indicators)
    matching_episodes = []
    for episode in episodes:
        # Skip Armchair Anonymous in DB
        if re.search(r'\barmchair\s+anonymous\b', episode["file_stem"], re.IGNORECASE):
            continue
        
        # Check if guest name appears in file_stem (ignoring repeat indicators)
        if guest_name_no_repeat in episode["normalized_stem_no_repeat"]:
            matching_episodes.append(episode)
    
    # If we found multiple episodes with the same guest, use repeat indicators to disambiguate
    if len(matching_episodes) > 1:
        # YouTube has repeat indicator - try to find matching repeat in DB
        if youtube_repeat:
            for episode in matching_episodes:
                if episode["has_repeat_indicator"]:
                    # Both have repeat indicators - check if they match
                    ep_repeat = extract_repeat_indicator(episode["file_stem"])
                    if ep_repeat and ep_repeat.lower() == youtube_repeat.lower():
                        return episode["id"], is_fc, 1.0, f"exact_guest_with_matching_repeat({youtube_repeat})"
                    
                    # Fuzzy match on full normalized string with repeat indicators
                    similarity = difflib.SequenceMatcher(
                        None, 
                        guest_name_normalized, 
                        episode["normalized_stem"]
                    ).ratio()
                    if similarity > best_score:
                        best_score = similarity
                        best_match = episode["id"]
                        best_method = f"fuzzy_with_repeat({similarity:.2f})"
        else:
            # YouTube has NO repeat indicator - match to episode WITHOUT repeat indicator
            for episode in matching_episodes:
                if not episode["has_repeat_indicator"]:
                    return episode["id"], is_fc, 1.0, "exact_guest_first_appearance"
            
            # If all matching episodes have repeat indicators but YouTube doesn't,
            # it might be mislabeled - take the first one with lower confidence
            if matching_episodes:
                best_match = matching_episodes[0]["id"]
                best_score = 0.85
                best_method = "guest_match_ambiguous_repeat"
    
    elif len(matching_episodes) == 1:
        # Only one episode with this guest - perfect match
        episode = matching_episodes[0]
        
        # If both have repeat indicators, they should match
        if youtube_repeat and episode["has_repeat_indicator"]:
            ep_repeat = extract_repeat_indicator(episode["file_stem"])
            if ep_repeat and ep_repeat.lower() == youtube_repeat.lower():
                return episode["id"], is_fc, 1.0, f"only_guest_with_repeat({youtube_repeat})"
            else:
                # Repeat indicators don't match - lower confidence
                best_match = episode["id"]
                best_score = 0.85
                best_method = f"only_guest_but_repeat_mismatch"
        else:
            return episode["id"], is_fc, 1.0, "only_guest_match"
    
    # Fallback: No matches found using guest name in stem, try other methods
    if not best_match:
        for episode in episodes:
            # Skip Armchair Anonymous
            if re.search(r'\barmchair\s+anonymous\b', episode["file_stem"], re.IGNORECASE):
                continue
            
            # Method: Check against guests list from database
            if episode["guests"]:
                for db_guest in episode["guests"]:
                    db_guest_normalized = normalize_for_matching(db_guest, keep_repeat_indicators=False)
                    
                    # Exact match on guest name
                    if db_guest_normalized == guest_name_no_repeat:
                        similarity = 0.95
                        if similarity > best_score:
                            best_score = similarity
                            best_match = episode["id"]
                            best_method = "exact_guest_from_list"
                    
                    # Fuzzy match
                    similarity = difflib.SequenceMatcher(None, db_guest_normalized, guest_name_no_repeat).ratio()
                    if similarity > 0.85 and similarity > best_score:
                        best_score = similarity
                        best_match = episode["id"]
                        best_method = f"fuzzy_guest_list({similarity:.2f})"
            
            # Method: Fuzzy match the entire file_stem
            similarity = difflib.SequenceMatcher(
                None, 
                guest_name_no_repeat, 
                episode["normalized_stem_no_repeat"]
            ).ratio()
            if similarity > 0.8 and similarity > best_score:
                best_score = similarity
                best_match = episode["id"]
                best_method = f"fuzzy_stem({similarity:.2f})"
    
    return best_match, is_fc, best_score, best_method

# -----------------------
# Update Database
# -----------------------
def update_episode_youtube_url(episode_id: int, url: str, is_fact_check: bool):
    """Update episode with YouTube URL."""
    with with_conn() as conn, conn.cursor() as cur:
        # For now, just update the youtube_url column
        # We can discuss if you want separate columns for main vs fact check
        if is_fact_check:
            # You might want to add a fact_check_url column later
            # For now, we'll track this separately or skip fact checks
            pass
        else:
            cur.execute("""
                UPDATE episodes 
                SET youtube_url = %s,
                    updated_at = NOW()
                WHERE id = %s
            """, (url, episode_id))
        
        conn.commit()

# -----------------------
# Main Process
# -----------------------
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Match YouTube URLs to episodes using file_stem")
    parser.add_argument("excel_file", help="Path to Excel file with YouTube URLs")
    parser.add_argument("--dry-run", action="store_true", help="Show matches without updating database")
    parser.add_argument("--min-confidence", type=float, default=0.75, help="Minimum confidence score (0-1)")
    parser.add_argument("--update-fact-checks", action="store_true", help="Also update fact check URLs (requires separate column)")
    parser.add_argument("--show-all", action="store_true", help="Show all attempts including no-match")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("YouTube URL Matcher for Armchair Expert")
    print("=" * 80)
    
    # Check schema
    columns = check_schema()
    
    # Load data
    youtube_videos = load_youtube_data(args.excel_file)
    episodes = get_all_episodes()
    
    print("\n" + "=" * 80)
    print("Matching YouTube videos to episodes...")
    print("=" * 80 + "\n")
    
    # Match and update
    stats = {
        "matched_main": 0,
        "matched_fact_check": 0,
        "updated": 0,
        "already_has_url": 0,
        "skipped_low_confidence": 0,
        "skipped_anonymous": 0,
        "skipped_moms_car": 0,
        "skipped_fact_check": 0,
        "no_match": 0
    }
    
    matches_to_review = []
    
    for i, youtube_video in enumerate(youtube_videos, 1):
        title = youtube_video["title"]
        url = youtube_video["url"]
        
        # Skip certain types
        if is_armchair_anonymous(title):
            stats["skipped_anonymous"] += 1
            continue
        
        if is_moms_car(title):
            stats["skipped_moms_car"] += 1
            continue
        
        # Match to episode
        episode_id, is_fc, confidence, method = match_youtube_to_episode(youtube_video, episodes)
        
        if episode_id and confidence >= args.min_confidence:
            # Get episode info
            episode = next((e for e in episodes if e["id"] == episode_id), None)
            
            if is_fc:
                stats["matched_fact_check"] += 1
                if not args.update_fact_checks:
                    stats["skipped_fact_check"] += 1
                    if args.show_all:
                        print(f"[{i}] ⊙ FACT CHECK (skipped) - {title[:60]}")
                    continue
                video_type = "FACT CHECK"
            else:
                stats["matched_main"] += 1
                video_type = "MAIN"
            
            # Check if already has URL
            if episode and episode["existing_url"]:
                stats["already_has_url"] += 1
                if args.show_all:
                    print(f"[{i}] ⊙ ALREADY HAS URL")
                    print(f"  File: {episode['file_stem'][:65]}")
                    print(f"  Existing: {episode['existing_url']}")
                    print(f"  New: {url}")
                    print()
                continue
            
            print(f"[{i}] ✓ MATCH ({confidence:.2f}) - {video_type} - {method}")
            print(f"  YouTube: {title[:65]}")
            print(f"  Episode: {episode['file_stem'][:65]}")
            print(f"  URL: {url}")
            print()
            
            matches_to_review.append({
                "episode_id": episode_id,
                "file_stem": episode["file_stem"],
                "youtube_title": title,
                "url": url,
                "is_fact_check": is_fc,
                "confidence": confidence
            })
            
            # Update database
            if not args.dry_run and not is_fc:
                update_episode_youtube_url(episode_id, url, is_fc)
                stats["updated"] += 1
        
        elif episode_id:
            stats["skipped_low_confidence"] += 1
            if args.show_all:
                episode = next((e for e in episodes if e["id"] == episode_id), None)
                print(f"[{i}] ⚠ LOW CONFIDENCE ({confidence:.2f}) - {method}")
                print(f"  YouTube: {title[:65]}")
                if episode:
                    print(f"  Would match: {episode['file_stem'][:65]}")
                print()
        else:
            stats["no_match"] += 1
            if args.show_all and method != "skipped_moms_car":
                print(f"[{i}] ✗ NO MATCH - {method}")
                print(f"  YouTube: {title[:65]}")
                print()
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total YouTube videos: {len(youtube_videos)}")
    print(f"Matched (main episodes): {stats['matched_main']}")
    print(f"Matched (fact checks): {stats['matched_fact_check']}")
    print(f"Already have URL: {stats['already_has_url']}")
    print(f"Updated in database: {stats['updated']}")
    print(f"Skipped (low confidence): {stats['skipped_low_confidence']}")
    print(f"Skipped (fact checks): {stats['skipped_fact_check']}")
    print(f"Skipped (Armchair Anonymous): {stats['skipped_anonymous']}")
    print(f"Skipped (Mom's Car): {stats['skipped_moms_car']}")
    print(f"No match found: {stats['no_match']}")
    print("=" * 80)
    
    if args.dry_run:
        print("\n⚠️  DRY RUN - No database updates performed")
        print("Run without --dry-run to update the database")
    else:
        print(f"\n✓ Updated {stats['updated']} episodes with YouTube URLs")

if __name__ == "__main__":
    main()