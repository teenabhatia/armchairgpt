#!/usr/bin/env python3
"""
extract_quotes_llm.py

Re-extracts guest mentions from podcast transcripts using an LLM for better
context-awareness than entity detection alone. Reads existing utterances from
the database and writes results back to the mentions table.

Providers:
    groq    — Free, no credit card. Get key at console.groq.com. Uses llama-3.3-70b.
              32k context window, so transcripts are chunked automatically.
    gemini  — Free tier (gemini-1.5-flash-8b). Get key at aistudio.google.com.
              1M context window, no chunking needed.
    mistral — Free tier. Get key at console.mistral.ai.
              32k context window, chunked.

Usage:
    python extract_quotes_llm.py --episode "Chris Feistl" --dry-run
    python extract_quotes_llm.py --provider groq --episode "Chris Feistl" --dry-run
    python extract_quotes_llm.py --provider gemini --model gemini-1.5-flash-8b
    python extract_quotes_llm.py --dry-run                  # preview, no DB writes
    python extract_quotes_llm.py --replace-all              # clear ALL mentions first

Setup:
    Add to your .env:  GROQ_API_KEY, GEMINI_API_KEY, or MISTRAL_API_KEY
    pip install groq           # for groq
    pip install google-genai   # for gemini
    pip install mistralai      # for mistral
"""

import os
import sys
import json
import re
import time
import argparse
import urllib.request
import xml.etree.ElementTree as ET
from html import unescape

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

HOSTS = {"dax shepard", "dax", "monica padman", "monica"}
HOST_TOKENS = {"dax", "monica", "shepard", "padman"}

RSS_URL = "https://rss.art19.com/armchair-expert"
_rss_cache = None  # populated lazily on first fetch

GUEST_LIST_PATH = os.path.join(os.path.dirname(__file__), "Copy of Armchair Guest List_1000.xlsx")


def load_guest_catalog(path: str) -> set:
    """
    Load the master guest list from the Excel file.
    Returns a set of lowercase last names and full names for fuzzy matching.
    """
    try:
        import pandas as pd
        df = pd.read_excel(path, header=1)  # row 2 in Excel has column names
        names = df.iloc[:, 4].dropna().tolist()  # column E (0-indexed = 4)
        names = [str(n).strip() for n in names if str(n).strip() not in ("", "nan", "Guest Name")]
        catalog = set()
        for name in names:
            # Handle entries like "Allison Tolman & Colin Hanks" — split into individuals
            for part in re.split(r"\s*&\s*|\s+and\s+", name, flags=re.IGNORECASE):
                part = part.strip()
                # Remove suffixes like "- rerelease", "(on X)"
                part = re.sub(r"\s*[-–(].*$", "", part).strip()
                if part:
                    catalog.add(part.lower())
                    # Also index by last name alone for partial matching
                    last = part.split()[-1]
                    if len(last) > 2:
                        catalog.add(last.lower())
        print(f"Loaded {len(names)} guests from catalog ({len(catalog)} name tokens)")
        return catalog
    except Exception as e:
        print(f"WARNING: Could not load guest catalog: {e}")
        return set()


GUEST_CATALOG: set = set()  # populated in main()


# ─── RSS Feed ─────────────────────────────────────────────────────────────────

def _clean_rss_description(html: str) -> str:
    """Strip HTML tags and remove ad/privacy boilerplate from episode description."""
    if not html:
        return ""
    text = unescape(html)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Drop sentences/lines that are sponsor copy or privacy notices
    sentences = re.split(r"(?<=[.!?])\s+", text)
    clean = []
    skip_patterns = re.compile(
        r"privacy policy|art19\.com/privacy|see privacy|sponsored by|"
        r"tinyurl\.com|turbotax|allstate|head to |check .* first for|"
        r"learn more at|https?://",
        re.IGNORECASE,
    )
    for s in sentences:
        if not skip_patterns.search(s):
            clean.append(s)
    return " ".join(clean).strip()


def fetch_rss_episodes() -> dict:
    """
    Fetch and parse the RSS feed, returning a dict of episode_number -> info.
    Result is cached after the first call.
    """
    global _rss_cache
    if _rss_cache is not None:
        return _rss_cache

    try:
        req = urllib.request.Request(RSS_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            xml_data = resp.read()

        root = ET.fromstring(xml_data)
        channel = root.find("channel")

        itunes_ns  = "http://www.itunes.com/dtds/podcast-1.0.dtd"
        content_ns = "http://purl.org/rss/1.0/modules/content/"

        episodes = {}
        for item in channel.findall("item"):
            ep_el = item.find(f"{{{itunes_ns}}}episode")
            if ep_el is None or not ep_el.text:
                continue
            try:
                ep_num = int(ep_el.text)
            except ValueError:
                continue

            # Prefer content:encoded for full HTML; fall back to description
            content_el = item.find(f"{{{content_ns}}}encoded")
            desc_el    = item.find("description")
            raw_html   = (
                (content_el.text if content_el is not None else None)
                or (desc_el.text if desc_el is not None else None)
                or ""
            )

            title_el = item.find("title")
            episodes[ep_num] = {
                "title":       (title_el.text or "").strip() if title_el is not None else "",
                "description": _clean_rss_description(raw_html),
            }

        _rss_cache = episodes
        print(f"[RSS] Fetched {len(episodes)} episodes from feed")
    except Exception as e:
        print(f"[RSS] WARNING: Could not fetch feed: {e}")
        _rss_cache = {}

    return _rss_cache


def get_rss_info(file_stem: str) -> dict | None:
    """
    Look up RSS metadata for an episode.
    Strategy 1: extract episode number from file_stem (e.g. '935 - Chris Feistl...').
    Strategy 2: fuzzy-match the file_stem against RSS episode titles by shared word overlap.
    Returns dict with 'title' and 'description', or None if not found.
    """
    episodes = fetch_rss_episodes()

    # Strategy 1 — episode number prefix (e.g. "935 - Chris Feistl...")
    # Only treat a leading number as an episode number if it's followed by a separator,
    # i.e. the number is clearly an episode index rather than part of the guest name.
    m = re.match(r"^(\d+)\s*[-–—]\s*", file_stem.strip())
    if m:
        ep_num = int(m.group(1))
        if ep_num in episodes:
            return episodes[ep_num]

    # Strategy 2 — title word overlap
    # Normalise: lowercase, strip punctuation, drop common filler words
    _filler = {"the", "a", "an", "and", "or", "of", "in", "with", "interview", "pt", "part"}

    def _words(text: str) -> set:
        return {w for w in re.sub(r"[^a-z0-9 ]", " ", text.lower()).split() if w not in _filler and len(w) > 1}

    stem_words = _words(file_stem)
    if not stem_words:
        return None

    best_ep, best_score = None, 0
    for ep_num, info in episodes.items():
        rss_words = _words(info["title"])
        if not rss_words:
            continue
        overlap = len(stem_words & rss_words)
        # True Jaccard: intersection / union — penalises RSS title being much larger than stem
        score = overlap / len(stem_words | rss_words)
        if score > best_score:
            best_score = score
            best_ep = ep_num

    # Require at least 60% Jaccard similarity
    if best_score >= 0.6:
        return episodes[best_ep]

    return None


# ─── Fact Check Detection ──────────────────────────────────────────────────────

def find_fact_check_start(utterances: list) -> int | None:
    """
    Detect the index where the Fact Check segment begins and return it.
    Strategy:
      1. Scan the last 40% of the episode for an utterance containing
         'fact check' / 'fact-check' spoken by a host — this is the explicit marker.
      2. Fallback: find the index after the last guest utterance if followed by
         a sustained host-only block (≥15 utterances).
    Returns the utterance index to truncate at, or None if not detected.
    """
    if not utterances:
        return None

    def _is_host(speaker: str) -> bool:
        tokens = set(speaker.lower().split())
        return bool(tokens & HOST_TOKENS)

    # Pass 1 — explicit "fact check" phrase
    search_from = int(len(utterances) * 0.55)
    for i in range(search_from, len(utterances)):
        u = utterances[i]
        if _is_host(u["speaker"]) and re.search(r"fact[- ]check", u["text"], re.IGNORECASE):
            return i

    # Pass 2 — fallback: last guest utterance followed by long host-only block
    for i in range(len(utterances) - 1, -1, -1):
        if not _is_host(utterances[i]["speaker"]):
            # i is the last guest utterance; check what follows
            host_block = utterances[i + 1:]
            if len(host_block) >= 15 and all(_is_host(u["speaker"]) for u in host_block):
                return i + 1
            break  # there are more guest utterances later — no clean cutoff

    return None


# ─── DB Connection ────────────────────────────────────────────────────────────

def get_conn():
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        if "sslmode=" not in db_url:
            sep = "&" if "?" in db_url else "?"
            db_url = f"{db_url}{sep}sslmode=require"
        return psycopg2.connect(db_url)
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        port=int(os.environ.get("PGPORT", "5432")),
        sslmode="require",
    )


def ensure_llm_mentions_table(conn):
    """Create the mentions_llm table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS mentions_llm (
                id               BIGSERIAL PRIMARY KEY,
                episode_id       BIGINT REFERENCES episodes(id) ON DELETE CASCADE,
                speaker          TEXT,
                about_person     TEXT,
                quote            TEXT,
                start_ms         INT,
                end_ms           INT,
                extraction_method TEXT
            );
        """)
    conn.commit()


# ─── LLM Calls ────────────────────────────────────────────────────────────────

def call_gemini(transcript: str, guests: list, episode_title: str, model_name: str, episode_description: str = "") -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = build_prompt(transcript, guests, episode_title, episode_description)
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    max_output_tokens=8192,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return response.text
        except Exception as e:
            if attempt < 4:
                wait = 30 * (attempt + 1)
                print(f"  Gemini error (attempt {attempt + 1}/5), retrying in {wait}s: {type(e).__name__}: {str(e)[:120]}")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Gemini failed after 5 retries")


def _parse_retry_seconds(error_str: str) -> int:
    """Extract wait seconds from Groq rate limit messages like 'try again in 45m4s'."""
    m = re.search(r"try again in\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:([\d.]+)s)?", error_str, re.IGNORECASE)
    if not m:
        return 65
    h = int(m.group(1) or 0)
    mins = int(m.group(2) or 0)
    secs = float(m.group(3) or 0)
    return int(h * 3600 + mins * 60 + secs) + 5  # +5s buffer


def call_groq(transcript: str, guests: list, episode_title: str, model_name: str, episode_description: str = "") -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    prompt = build_prompt(transcript, guests, episode_title, episode_description)
    while True:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as e:
            wait = _parse_retry_seconds(str(e))
            print(f"  Rate limited — waiting {wait // 60}m {wait % 60}s then retrying...")
            time.sleep(wait)


def call_mistral(transcript: str, guests: list, episode_title: str, model_name: str, episode_description: str = "") -> str:
    from mistralai import Mistral
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    prompt = build_prompt(transcript, guests, episode_title, episode_description)
    response = client.chat.complete(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content


# ─── Prompt ───────────────────────────────────────────────────────────────────

def build_prompt(transcript: str, guests: list, episode_title: str, episode_description: str = "") -> str:
    guests_str = ", ".join(guests) if guests else "unknown"

    description_section = ""
    if episode_description:
        description_section = f"\nEpisode show notes (use for context on topics and names discussed):\n{episode_description}\n"

    return f"""You are analyzing a transcript from the Armchair Expert podcast, hosted by Dax Shepard and Monica Padman.

Episode: {episode_title}
Guests in this episode: {guests_str}{description_section}

The transcript below uses the format: [index] Speaker: text

STEP 1 — RESOLVE SPEAKER NAMES:
Some speakers may be labeled generically as "Speaker A", "Speaker B", "Speaker C", etc.
Use context clues in the transcript to identify which label belongs to which guest — hosts typically introduce guests by name early in the episode, and guests often refer to each other by name.
Map every generic label to the correct guest name from the guest list above before doing anything else.
If a speaker is clearly Dax Shepard or Monica Padman (even if labeled generically), treat them as hosts.

STEP 2 — EXTRACT MENTIONS:
Find every utterance where a GUEST mentions or tells a story about another NAMED person.

Return a JSON array where each object has exactly these fields:
- "utterance_index": integer (the number in brackets from the transcript line)
- "speaker": the REAL name of the guest (never a generic label like "Speaker B")
- "about_person": full name of the person being discussed
- "quote": verbatim text of the utterance

STRICT RULES:
1. ONLY include utterances where the speaker is a GUEST. Exclude: Dax Shepard, Monica Padman, Monica, Dax.
2. "speaker" must always be a real name from the guest list — never "Speaker A/B/C/etc".
3. "about_person" MUST be a real proper name (e.g. "Pablo Escobar", "Jorge Salcedo").
   NEVER use titles, roles, or relationships like "the prosecutor", "the ambassador", "my brother", "a friend", "someone". If you don't know their actual name, skip it.
4. Skip self-mentions — guest talking about themselves.
5. If one utterance mentions multiple distinct named people, create one entry per person.
6. Return ONLY a valid JSON array, no markdown fences, no explanation.

EXAMPLES of what to INCLUDE vs SKIP:

INCLUDE — guest mentions a real named person:
  speaker: "Chris Feistl", about_person: "Pablo Escobar", quote: "Escobar was killed on a rooftop..."

INCLUDE — speaker was labeled "Speaker B" but context shows it's the guest:
  speaker: "Chris Feistl" (resolved from "Speaker B"), about_person: "Jorge Salcedo", quote: "Salcedo was the head of security..."

SKIP — generic label not resolvable:
  speaker: "Speaker B" → resolve to real name first; if truly unresolvable, skip

SKIP — vague role with no name:
  "the prosecutor came in and shut it down" → NO (no real name)

SKIP — family descriptor with no name:
  "my older brother became a police officer" → NO (no real name)

SKIP — host is speaking:
  speaker: "Dax Shepard" → always skip

TRANSCRIPT:
{transcript}"""


# ─── Transcript Building ──────────────────────────────────────────────────────

def merge_consecutive_utterances(utterances: list) -> list:
    """
    Merge consecutive utterances from the same speaker into one entry.
    Produces longer, context-rich quotes instead of sentence fragments.
    The original idx (for timestamp lookup) is preserved from the first
    utterance in each group.
    """
    if not utterances:
        return []
    merged = []
    current = dict(utterances[0])
    for u in utterances[1:]:
        if u["speaker"] == current["speaker"]:
            current["text"]   = current["text"].rstrip() + " " + u["text"].lstrip()
            current["end_ms"] = u["end_ms"]
        else:
            merged.append(current)
            current = dict(u)
    merged.append(current)
    # Re-index so utterance_index values are sequential after merging
    for i, u in enumerate(merged):
        u["idx"] = i
    return merged


def build_transcript_text(utterances: list) -> str:
    return "\n".join(f"[{u['idx']}] {u['speaker']}: {u['text']}" for u in utterances)


def chunk_utterances(utterances: list, chunk_size: int = 250, overlap: int = 25) -> list:
    """Split utterances into overlapping windows for models with small context limits."""
    chunks = []
    i = 0
    while i < len(utterances):
        chunks.append(utterances[i : i + chunk_size])
        i += chunk_size - overlap
    return chunks


# ─── Response Parsing ─────────────────────────────────────────────────────────

def parse_llm_response(raw: str) -> list:
    text = raw.strip()
    # Strip markdown code fences if the model added them
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "mentions" in data:
            return data["mentions"]
        return []
    except json.JSONDecodeError:
        # Response was likely truncated — extract all complete objects individually
        objects = re.findall(r'\{[^{}]+\}', text, re.DOTALL)
        results = []
        for obj in objects:
            try:
                results.append(json.loads(obj))
            except json.JSONDecodeError:
                continue
        if results:
            print(f"  WARNING: Response truncated — recovered {len(results)} partial entries")
        else:
            print(f"  WARNING: Could not parse response (first 500 chars): {text[:500]}")
        return results


def is_host(name: str) -> bool:
    n = name.lower().strip()
    if n in HOSTS:
        return True
    # Drop any unresolved generic labels the LLM failed to map
    if re.match(r"^speaker\s+[a-z]$", n):
        return True
    return False


# ─── Episode Processing ───────────────────────────────────────────────────────

def parse_guest_names(guests: list, title: str) -> list:
    """
    Expand a guest list that may contain combined names like ['Chris Feistl Dave Mitchell']
    into individual names ['Chris Feistl', 'Dave Mitchell'] by splitting on common separators
    and cross-referencing with the episode title.
    """
    individual = []
    for g in guests:
        # Split on common separators used in episode filenames
        parts = re.split(r'\s+and\s+|\s*&\s*|\s*,\s*|\s+with\s+', g, flags=re.IGNORECASE)
        individual.extend([p.strip() for p in parts if p.strip()])

    # If splitting didn't help (still one long string), try splitting on title words
    # e.g. "Chris Feistl Dave Mitchell" — split every 2 capitalized words
    expanded = []
    for name in individual:
        words = name.split()
        # Heuristic: if >3 words and all capitalized, likely multiple full names
        if len(words) >= 4 and all(w[0].isupper() for w in words if w):
            # Split into pairs of words (first + last name)
            pairs = [' '.join(words[i:i+2]) for i in range(0, len(words), 2)]
            expanded.extend(pairs)
        else:
            expanded.append(name)
    return expanded if expanded else guests


def process_episode(conn, episode: dict, provider: str, model_name: str, dry_run: bool, skip_existing: bool = False):
    eid = episode["id"]
    title = episode["file_stem"]
    raw_guests = episode.get("guests") or []
    if isinstance(raw_guests, str):
        raw_guests = json.loads(raw_guests)
    guests = parse_guest_names(raw_guests, title)

    # Skip if already has quotes (unless forced)
    if skip_existing and not dry_run:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM mentions_llm WHERE episode_id = %s", (eid,))
            if cur.fetchone()[0] > 0:
                print(f"  SKIP {title} (already has quotes)")
                return

    print(f"\n{'─' * 60}")
    print(f"Episode : {title}")
    print(f"Guests  : {guests}")

    # ── RSS metadata ──
    rss_info = get_rss_info(title)
    episode_description = ""
    if rss_info:
        episode_description = rss_info.get("description", "")
        print(f"  RSS match: {rss_info['title']!r} ({len(episode_description)} chars of show notes)")
    else:
        print("  RSS match: not found (no episode number in filename or episode not in feed)")

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT id, speaker, start_ms, end_ms, text FROM utterances "
            "WHERE episode_id = %s ORDER BY start_ms",
            (eid,),
        )
        rows = cur.fetchall()

    if not rows:
        print("  No utterances found — skipping")
        return

    utterances = [dict(r, idx=i) for i, r in enumerate(rows)]
    utterances = merge_consecutive_utterances(utterances)
    print(f"  Utterances after merging consecutive speaker turns: {len(utterances)}")

    # ── Trim Fact Check ──
    fc_start = find_fact_check_start(utterances)
    if fc_start is not None:
        removed = len(utterances) - fc_start
        fc_ts = utterances[fc_start]["start_ms"] // 1000
        print(f"  Fact Check detected at utterance {fc_start} "
              f"({fc_ts // 60}:{fc_ts % 60:02d}) — trimming {removed} utterances")
        utterances = utterances[:fc_start]
    else:
        print("  Fact Check: not detected (processing full transcript)")

    # ── Call LLM ──
    raw_mentions = []

    if provider == "gemini":
        transcript = build_transcript_text(utterances)
        char_count = len(transcript)
        print(f"  Sending {char_count:,} chars to {model_name}...")
        try:
            raw = call_gemini(transcript, guests, title, model_name, episode_description)
            raw_mentions = parse_llm_response(raw)
        except Exception as e:
            print(f"  ERROR calling Gemini: {e}")
            return

    elif provider in ("mistral", "groq"):
        # Groq free tier: 12k TPM. ~150 utterances ≈ 8k tokens + prompt overhead = safe.
        chunk_size = 150 if provider == "groq" else 250
        overlap    = 15  if provider == "groq" else 25
        sleep_sec  = 6   if provider == "groq" else 0.5
        chunks = chunk_utterances(utterances, chunk_size=chunk_size, overlap=overlap)
        print(f"  Splitting into {len(chunks)} chunks for {model_name}...")
        call_fn = call_groq if provider == "groq" else call_mistral
        seen_indices = set()
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i + 1}/{len(chunks)} ({len(chunk)} utterances)...")
            transcript = build_transcript_text(chunk)
            try:
                raw = call_fn(transcript, guests, title, model_name, episode_description)
                for m in parse_llm_response(raw):
                    idx = m.get("utterance_index")
                    if idx not in seen_indices:
                        seen_indices.add(idx)
                        raw_mentions.append(m)
                if i < len(chunks) - 1:
                    time.sleep(sleep_sec)
            except Exception as e:
                print(f"  ERROR on chunk {i + 1}: {e}")
                continue

    print(f"  LLM returned {len(raw_mentions)} raw mentions")

    # ── Filter & resolve timestamps ──
    # Build a set of all guest name tokens for co-guest filtering
    guest_name_tokens = set()
    for g in guests:
        for token in g.lower().split():
            if len(token) > 2:
                guest_name_tokens.add(token)

    valid_mentions = []
    for m in raw_mentions:
        speaker = str(m.get("speaker", "")).strip()
        about   = str(m.get("about_person", "")).strip()
        quote   = str(m.get("quote", "")).strip()
        idx     = m.get("utterance_index")

        if not all([speaker, about, quote]):
            continue
        if is_host(speaker):
            continue
        if about.lower() == speaker.lower():
            continue  # self-mention
        # Skip co-guest mentions (guests mentioning each other within the same episode)
        about_tokens = set(about.lower().split())
        if about_tokens & guest_name_tokens:
            continue
        # Filter: about_person must be a known podcast guest
        if GUEST_CATALOG and not (about_tokens & GUEST_CATALOG):
            continue
        if idx is None or not isinstance(idx, int) or idx >= len(utterances):
            continue

        u = utterances[idx]
        valid_mentions.append({
            "speaker":           speaker,
            "about_person":      about,
            "quote":             quote,
            "start_ms":          u["start_ms"],
            "end_ms":            u["end_ms"],
            "extraction_method": f"llm_{provider}",
        })

    print(f"  Valid after filtering: {len(valid_mentions)}")

    if dry_run:
        for m in valid_mentions:
            ts = f"{m['start_ms'] // 60000}:{(m['start_ms'] % 60000) // 1000:02d}"
            print(f"    [{ts}] {m['speaker']} → {m['about_person']}: {m['quote'][:100]}...")
        return

    # ── Write to DB ──
    with conn.cursor() as cur:
        # Always replace this episode's LLM mentions on rerun (idempotent)
        cur.execute("DELETE FROM mentions_llm WHERE episode_id = %s", (eid,))

        for m in valid_mentions:
            cur.execute(
                """INSERT INTO mentions_llm
                       (episode_id, speaker, about_person, quote, start_ms, end_ms, extraction_method)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (eid, m["speaker"], m["about_person"], m["quote"],
                 m["start_ms"], m["end_ms"], m["extraction_method"]),
            )

    conn.commit()
    print(f"  Stored {len(valid_mentions)} mentions")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract guest mentions from podcast transcripts using an LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--episode",
        help="Process only episodes whose file_stem contains this substring (case-insensitive)",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "groq", "mistral"],
        default="groq",
        help="LLM provider to use (default: groq)",
    )
    parser.add_argument(
        "--model",
        help="Override the model name (e.g. gemini-2.0-flash-lite, mistral-small-latest)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip episodes that already have quotes in the database (faster for incremental runs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print extracted mentions without writing anything to the database",
    )
    args = parser.parse_args()

    default_models = {
        "gemini":  "gemini-2.5-flash",
        "groq":    "llama-3.1-8b-instant",
        "mistral": "mistral-small-latest",
    }
    model_name = args.model or default_models[args.provider]

    # Check API key is present
    key_env = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "mistral": "MISTRAL_API_KEY"}[args.provider]
    if not os.environ.get(key_env):
        print(f"ERROR: {key_env} is not set. Add it to your .env file.")
        sys.exit(1)

    print(f"Provider : {args.provider}")
    print(f"Model    : {model_name}")
    print(f"Dry run  : {args.dry_run}")

    global GUEST_CATALOG
    GUEST_CATALOG = load_guest_catalog(GUEST_LIST_PATH)

    conn = get_conn()

    ensure_llm_mentions_table(conn)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if args.episode:
            cur.execute(
                "SELECT id, file_stem, guests FROM episodes "
                "WHERE file_stem ILIKE %s ORDER BY file_stem",
                (f"%{args.episode}%",),
            )
        else:
            cur.execute("SELECT id, file_stem, guests FROM episodes ORDER BY file_stem")
        episodes = cur.fetchall()

    if not episodes:
        print("No matching episodes found.")
        conn.close()
        return

    print(f"\nProcessing {len(episodes)} episode(s)...")

    for ep in episodes:
        process_episode(conn, ep, args.provider, model_name, args.dry_run, skip_existing=args.skip_existing)

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
