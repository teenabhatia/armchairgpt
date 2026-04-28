# import os
# import re
# import time
# import json
# import requests
# import psycopg2
# from psycopg2.extras import execute_values
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv
# from pathlib import Path
# import argparse

# # -----------------------
# # Config / init
# # -----------------------
# load_dotenv()
# BASE_URL = "https://api.assemblyai.com"
# API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")
# if not API_KEY:
#     raise RuntimeError("Missing ASSEMBLYAI_API_KEY")
# headers = {"authorization": API_KEY}

# DB_URL = os.environ.get("DATABASE_URL", "")
# if not DB_URL:
#     raise RuntimeError("Missing DATABASE_URL (postgres)")

# # Your local audio file to process
# file_path = "/Users/teenabhatia/Desktop/Armchair Expert/rss/downloads/Kristen Bell.mp3"

# # Hosts (fixed)
# HOSTS = ["Dax Shepard", "Monica Padman"]

# # Embedding model
# EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims
# _emb_model = None

# def emb_model():
#     global _emb_model
#     if _emb_model is None:
#         _emb_model = SentenceTransformer(EMB_MODEL_NAME)
#     return _emb_model

# # -----------------------
# # Utilities
# # -----------------------
# def safe_stem(path: str) -> str:
#     stem = os.path.splitext(os.path.basename(path))[0]
#     stem = re.sub(r"\s+", " ", stem).strip()
#     stem = re.sub(r"[^\w\-\.\(\) ]", "", stem)
#     return stem or "audio"

# def parse_guests_from_filename(stem: str):
#     s = stem
#     s = re.sub(r"^\s*\d+\s*-\s*", "", s)
#     s = re.sub(r"\s*-\s*(Interview.*|Part\s*\d+|v\d+|Remaster.*)$", "", s, flags=re.I)
#     parts = re.split(r"\s*(?:&|,| and | with )\s*", s, flags=re.I)
#     bad = {"armchair","expert","live","best of","bonus","interview"}
#     out = []
#     for p in parts:
#         t = p.strip(" -–—")
#         if t and not any(b in t.lower() for b in bad):
#             out.append(t)
#     # de-dupe case-insensitive
#     seen, uniq = set(), []
#     for n in out:
#         k = n.lower()
#         if k not in seen:
#             uniq.append(n)
#             seen.add(k)
#     return uniq

# def episode_exists(file_stem: str) -> int | None:
#     """Return episode_id if already in DB, else None."""
#     with with_conn() as conn, conn.cursor() as cur:
#         cur.execute("SELECT id FROM episodes WHERE file_stem=%s", (file_stem,))
#         row = cur.fetchone()
#         return row[0] if row else None

# def best_overlapping_utterance(utterances, start_ms, end_ms):
#     best, best_overlap = None, 0
#     for u in utterances or []:
#         us, ue = u.get("start"), u.get("end")
#         if us is None or ue is None:
#             continue
#         overlap = max(0, min(ue, end_ms) - max(us, start_ms))
#         if overlap > best_overlap:
#             best_overlap, best = overlap, u
#     return best

# def map_hosts_and_guests(utterances, parsed_guests):
#     # total talk-time by diarized label
#     dur, first_seen = {}, {}
#     for u in utterances or []:
#         lab = u.get("speaker")
#         if not lab: 
#             continue
#         st, en = u.get("start"), u.get("end")
#         if st is None or en is None:
#             continue
#         dur[lab] = dur.get(lab, 0) + max(0, en - st)
#         if lab not in first_seen:
#             first_seen[lab] = st

#     labels = list(dur.keys())
#     # pick two most talkative as hosts
#     host_labels = sorted(labels, key=lambda L: -dur[L])[:2]
#     mapping = {}
#     for lab, name in zip(host_labels, HOSTS):
#         mapping[lab] = name

#     # remaining labels get mapped to parsed guest names by first-appearance order
#     remaining = [lab for lab in labels if lab not in mapping]
#     remaining_sorted = sorted(remaining, key=lambda L: first_seen.get(L, 10**12))
#     for lab, gname in zip(remaining_sorted, parsed_guests):
#         mapping[lab] = f"Guest: {gname}"

#     for lab in labels:
#         if lab not in mapping:
#             mapping[lab] = f"Speaker {lab}"
#     return mapping

# def embed_texts(texts):
#     # cosine-friendly unit vectors
#     vecs = emb_model().encode(texts, normalize_embeddings=True).tolist()
#     return vecs

# # -----------------------
# # Postgres / pgvector
# # -----------------------
# DDL = """
# CREATE EXTENSION IF NOT EXISTS vector;

# CREATE TABLE IF NOT EXISTS episodes (
#   id BIGSERIAL PRIMARY KEY,
#   file_stem TEXT UNIQUE,
#   file_path TEXT,
#   guests JSONB,
#   created_at TIMESTAMPTZ DEFAULT NOW()
# );

# CREATE TABLE IF NOT EXISTS utterances (
#   id BIGSERIAL PRIMARY KEY,
#   episode_id BIGINT REFERENCES episodes(id) ON DELETE CASCADE,
#   speaker TEXT,
#   start_ms INT,
#   end_ms INT,
#   text TEXT
# );

# CREATE TABLE IF NOT EXISTS mentions (
#   id BIGSERIAL PRIMARY KEY,
#   episode_id BIGINT REFERENCES episodes(id) ON DELETE CASCADE,
#   speaker TEXT,
#   about_person TEXT,
#   quote TEXT,
#   start_ms INT,
#   end_ms INT
# );

# -- Vectorized chunks (we'll store one row per utterance for now)
# CREATE TABLE IF NOT EXISTS chunks (
#   id BIGSERIAL PRIMARY KEY,
#   episode_id BIGINT REFERENCES episodes(id) ON DELETE CASCADE,
#   speaker TEXT,
#   start_ms INT,
#   end_ms INT,
#   text TEXT,
#   embedding vector(384)
# );

# -- If pgvector >= 0.5 on PG16+, prefer HNSW:
# -- CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);

# -- Otherwise IVFFLAT (run ANALYZE afterwards; adjust lists for your scale):
# CREATE INDEX IF NOT EXISTS chunks_embedding_ivf ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
# """

# def with_conn():
#     return psycopg2.connect(DB_URL)

# def ensure_schema():
#     with with_conn() as conn, conn.cursor() as cur:
#         cur.execute(DDL)
#         conn.commit()

# def upsert_episode(file_stem, file_path, guests):
#     with with_conn() as conn, conn.cursor() as cur:
#         cur.execute("""
#             INSERT INTO episodes (file_stem, file_path, guests)
#             VALUES (%s, %s, %s)
#             ON CONFLICT (file_stem) DO UPDATE SET file_path=EXCLUDED.file_path, guests=EXCLUDED.guests
#             RETURNING id
#         """, (file_stem, file_path, json.dumps(guests)))
#         eid = cur.fetchone()[0]
#         conn.commit()
#         return eid

# def clear_episode_rows(episode_id):
#     with with_conn() as conn, conn.cursor() as cur:
#         cur.execute("DELETE FROM utterances WHERE episode_id=%s", (episode_id,))
#         cur.execute("DELETE FROM mentions WHERE episode_id=%s", (episode_id,))
#         cur.execute("DELETE FROM chunks WHERE episode_id=%s", (episode_id,))
#         conn.commit()

# def insert_utterances(episode_id, rows):
#     # rows: (speaker, start_ms, end_ms, text)
#     with with_conn() as conn, conn.cursor() as cur:
#         execute_values(cur, """
#             INSERT INTO utterances (episode_id, speaker, start_ms, end_ms, text)
#             VALUES %s
#         """, [(episode_id, *r) for r in rows])
#         conn.commit()

# def insert_mentions(episode_id, rows):
#     # rows: (speaker, about_person, quote, start_ms, end_ms)
#     if not rows:
#         return
#     with with_conn() as conn, conn.cursor() as cur:
#         execute_values(cur, """
#             INSERT INTO mentions (episode_id, speaker, about_person, quote, start_ms, end_ms)
#             VALUES %s
#         """, [(episode_id, *r) for r in rows])
#         conn.commit()

# def insert_chunks_with_embeddings(episode_id, rows):
#     # rows: (speaker, start_ms, end_ms, text)
#     if not rows:
#         return
#     texts = [r[3] for r in rows]
#     vecs = embed_texts(texts)
#     # pgvector accepts literal like '[v1,v2,...]'
#     vec_literals = ["[" + ",".join(f"{x:.6f}" for x in v) + "]" for v in vecs]
#     with with_conn() as conn, conn.cursor() as cur:
#         execute_values(cur, """
#             INSERT INTO chunks (episode_id, speaker, start_ms, end_ms, text, embedding)
#             VALUES %s
#         """, [(episode_id, r[0], r[1], r[2], r[3], vec_literals[i]) for i, r in enumerate(rows)])
#         conn.commit()

# # Optional: simple semantic search helper for later
# def semantic_search(query, k=5):
#     qv = embed_texts([query])[0]
#     qlit = "[" + ",".join(f"{x:.6f}" for x in qv) + "]"
#     sql = """
#       SELECT id, episode_id, speaker, start_ms, end_ms, text,
#              1 - (embedding <#> %s::vector) AS cosine_sim
#       FROM chunks
#       ORDER BY embedding <#> %s::vector
#       LIMIT %s
#     """
#     with with_conn() as conn, conn.cursor() as cur:
#         cur.execute(sql, (qlit, qlit, k))
#         return cur.fetchall()

# # -----------------------
# # Transcribe + ingest one file (hybrid)
# # -----------------------
# def main():
#     ensure_schema()

#     # 1) Upload
#     with open(file_path, "rb") as f:
#         up = requests.post(f"{BASE_URL}/v2/upload", headers=headers, data=f)
#         up.raise_for_status()
#     audio_url = up.json()["upload_url"]

#     # 2) Transcribe (diarization + entity detection)
#     payload = {
#         "audio_url": audio_url,
#         "speech_models": ["universal-3-pro"],
#         "speaker_labels": True,
#         "speakers_expected": 4,
#         "entity_detection": True
#     }
#     resp = requests.post(f"{BASE_URL}/v2/transcript", json=payload, headers=headers)
#     resp.raise_for_status()
#     tid = resp.json()["id"]

#     # 3) Poll
#     endpoint = f"{BASE_URL}/v2/transcript/{tid}"
#     print("Transcription started. Polling...")
#     while True:
#         r = requests.get(endpoint, headers=headers)
#         r.raise_for_status()
#         res = r.json()
#         if res.get("status") == "completed":
#             break
#         if res.get("status") == "error":
#             raise RuntimeError(res.get("error"))
#         time.sleep(3)

#     utterances = res.get("utterances") or []
#     entities = [e for e in res.get("entities", []) if e.get("entity_type") == "person_name"]

#     # 4) Name speakers
#     stem = safe_stem(file_path)
#     parsed_guests = parse_guests_from_filename(stem)
#     label_to_name = map_hosts_and_guests(utterances, parsed_guests)

#     # 5) Prepare rows for DB
#     # utterance rows (speaker_resolved, start_ms, end_ms, text)
#     utt_rows = []
#     for u in utterances:
#         spk = label_to_name.get(u.get("speaker",""), u.get("speaker","SPK"))
#         utt_rows.append((spk, u.get("start") or -1, u.get("end") or -1, (u.get("text") or "").strip()))

#     # mentions: guest speaker mentioning another guest from the filename catalog
#     guest_catalog = [g for g in parsed_guests if g.lower() not in {"dax shepard","monica padman"}]
#     mention_rows = []
#     for ent in entities:
#         ent_name = (ent.get("text") or "").strip()
#         if not ent_name:
#             continue
#         s, e = ent.get("start"), ent.get("end")
#         if s is None or e is None:
#             continue
#         u = best_overlapping_utterance(utterances, s, e)
#         if not u:
#             continue
#         speaker_label = u.get("speaker","")
#         speaker_name = label_to_name.get(speaker_label, speaker_label)
#         if not speaker_name.startswith("Guest:"):
#             continue
#         # is target a guest from this episode?
#         if ent_name.lower() not in {g.lower() for g in guest_catalog}:
#             continue
#         # avoid self-mentions
#         self_guest = speaker_name.replace("Guest:", "").strip().lower()
#         if ent_name.lower() == self_guest:
#             continue
#         quote_text = (u.get("text") or "").strip()
#         mention_rows.append((speaker_name, ent_name, quote_text, u.get("start") or -1, u.get("end") or -1))

#     # 6) Upsert Episode + insert all rows
#     eid = upsert_episode(stem, os.path.abspath(file_path), parsed_guests)
#     # If re-running, clear per-episode rows to keep a single source of truth
#     clear_episode_rows(eid)
#     insert_utterances(eid, utt_rows)
#     insert_mentions(eid, mention_rows)
#     insert_chunks_with_embeddings(eid, utt_rows)  # one chunk per utterance (good default)

#     print(f"Ingested episode '{stem}' → episode_id={eid}")
#     print(f"Utterances: {len(utt_rows)} | Mentions: {len(mention_rows)}")

#     # Quick smoke test search (RAG-ish)
#     hits = semantic_search("Kristen Bell on comedy and improv", k=3)
#     for row in hits:
#         _id, ep, spk, st, en, txt, sim = row
#         print(f"\n[hit {sim:.3f}] {spk} [{st}–{en}]")
#         print(txt[:220], "...")
#     print("\nDone.")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# transcribe_to_pgvector.py

#!/usr/bin/env python3
# transcribe_to_pgvector.py
# Batch: AssemblyAI (diarization + entity detection) → LeMUR disambiguation (with transcript_ids)
# Store utterances, mentions, and vector embeddings in Postgres + pgvector
# Uses a GLOBAL guest catalog built from: filenames + DB + optional CSV (fuzzy / smart matching)

import os
import re
import csv
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import difflib

import requests
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# -----------------------
# Config / init
# -----------------------
load_dotenv()

BASE_URL = "https://api.assemblyai.com"

# Support multiple API keys — rotates automatically when one runs out of credits
_ALL_API_KEYS = [
    k for k in [
        os.environ.get("ASSEMBLYAI_API_KEY", ""),
        os.environ.get("ASSEMBLYAI_API_KEY_2", ""),
        os.environ.get("ASSEMBLYAI_API_KEY_3", ""),
        os.environ.get("ASSEMBLYAI_API_KEY_4", ""),
    ] if k.strip()
]
if not _ALL_API_KEYS:
    raise RuntimeError("Missing ASSEMBLYAI_API_KEY in environment (.env).")
_key_index = 0
API_KEY = _ALL_API_KEYS[0]
print(f"[keys] Loaded {len(_ALL_API_KEYS)} AssemblyAI API key(s).")

def _rotate_api_key():
    """Switch to the next API key when current one runs out of credits."""
    global _key_index, API_KEY
    _key_index += 1
    if _key_index >= len(_ALL_API_KEYS):
        raise RuntimeError("All AssemblyAI API keys exhausted (out of credits).")
    API_KEY = _ALL_API_KEYS[_key_index]
    print(f"[keys] Switched to API key {_key_index + 1} of {len(_ALL_API_KEYS)}.")

# If provided, process this single file; otherwise use --input-dir
file_path = ""   # e.g., "/Users/you/path/to/episode.m4a"

# Hosts (fixed)
HOSTS = ["Dax Shepard", "Monica Padman"]

# Skip Armchair Anonymous, Mom's Car, Best Of, and Holiday compilations by default
SKIP_TITLES_RE = re.compile(
    r"^\s*(armchair\s+anonymous\b|mom.?s\s+car\b|best\s+of\b|holiday\s+(dinner|spectacular)\b|introducing.*mom.?s\s+car\b)",
    re.IGNORECASE
)

# Embedding model (384 dims)
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_emb_model = None
def emb_model():
    global _emb_model
    if _emb_model is None:
        _emb_model = SentenceTransformer(EMB_MODEL_NAME)
    return _emb_model

# ------------- Helpers: filenames, parsing, mapping -------------
def safe_stem(path_or_str: str) -> str:
    stem = os.path.splitext(os.path.basename(path_or_str))[0]
    stem = re.sub(r"\s+", " ", stem).strip()
    stem = re.sub(r"[^\w\-\.\(\) ]", "", stem)
    return stem or "audio"

def parse_guests_from_filename(stem: str) -> List[str]:
    """
    Heuristics to pull guest names from filename.
    Examples:
      "935 - Chris Feistl and Dave Mitchell - Interview_TH v1" -> ["Chris Feistl","Dave Mitchell"]
      "Kristen Bell" -> ["Kristen Bell"]
    """
    s = stem
    s = re.sub(r"^\s*\d+\s*-\s*", "", s)  # drop numeric prefixes like "935 - "
    s = re.sub(r"\s*-\s*(Interview.*|Part\s*\d+|v\d+|Remaster.*)$", "", s, flags=re.I)

    parts = re.split(r"\s*(?:&|,| and | with )\s*", s, flags=re.I)
    bad = {"armchair", "expert", "live", "best of", "bonus", "interview"}
    out = []
    for p in parts:
        t = p.strip(" -–—")
        if t and not any(b in t.lower() for b in bad):
            out.append(t)

    # de-dupe case-insensitive
    seen, uniq = set(), []
    for n in out:
        k = n.lower()
        if k not in seen:
            uniq.append(n)
            seen.add(k)
    return uniq

def best_overlapping_utterance(utterances, start_ms, end_ms):
    """Return the utterance dict that overlaps [start_ms, end_ms] the most."""
    best, best_overlap = None, 0
    for u in utterances or []:
        us, ue = u.get("start"), u.get("end")
        if us is None or ue is None:
            continue
        overlap = max(0, min(ue, end_ms) - max(us, start_ms))
        if overlap > best_overlap:
            best_overlap, best = overlap, u
    return best

def map_hosts_and_guests(
    utterances,
    parsed_guests: List[str],
    present_hosts: set,
    fixed_host_labels: Optional[Dict[str, str]] = None,
    intro_skip_ms: int = 180000,        # 3 min
    min_guest_talk_ms: int = 60000,     # 60 sec AFTER skip
    ad_keywords: Optional[List[str]] = None,
) -> Dict[str, str]:
    HOST_NAME = {"dax": "Dax Shepard", "monica": "Monica Padman"}
    present_hosts = {h for h in (present_hosts or set()) if h in HOST_NAME}

    # Stats per diarization label
    stats = compute_label_stats(utterances, intro_skip_ms, ad_keywords, ignore_ranges=None)

    labels = list(stats.keys())

    # 1) Pre-lock hosts if LeMUR told us (fixed_host_labels)
    mapping: Dict[str, str] = {}
    fixed_host_labels = fixed_host_labels or {}
    for key, lab in fixed_host_labels.items():
        if key in present_hosts and lab in labels:
            mapping[lab] = HOST_NAME[key]

    # 2) Assign remaining present hosts by TOTAL talk-time (not impacted by ads much)
    remaining_labels = [L for L in sorted(labels, key=lambda L: -stats[L]["dur"]) if L not in mapping]
    remaining_host_names = [
        HOST_NAME[h] for h in ("dax", "monica")
        if h in present_hosts and HOST_NAME[h] not in mapping.values()
    ]
    for lab, host_name in zip(remaining_labels, remaining_host_names):
        mapping[lab] = host_name

    # 3) Choose guest candidates: enough talk AFTER skip, and not ad-like
    def is_ad_like(L):
        s = stats[L]
        return s["ad_hits"] >= 2 and s["dur_after"] < 30_000  # 2+ ad hits and <30s after skip

    candidates = [
        L for L in labels
        if L not in mapping
        and (stats[L]["dur_after"] >= min_guest_talk_ms or stats[L]["dur"] >= (min_guest_talk_ms * 1.5))
        and not is_ad_like(L)
    ]
    # Prefer the ones who actually talk AFTER skip; tie-break by earliest first_after
    candidates.sort(key=lambda L: (-stats[L]["dur_after"], stats[L]["first_after"] or 10**12))

    # 4) Map parsed guests to these candidates in order
    for lab, gname in zip(candidates, parsed_guests):
        mapping[lab] = f"Guest: {gname}"

    # 5) Anything leftover -> generic
    for lab in labels:
        if lab not in mapping:
            mapping[lab] = f"Speaker {lab}"
    return mapping




# ---------- GLOBAL GUEST CATALOG + MATCHING ----------
def build_global_guest_catalog_from_files(input_dir: Path, patterns: List[str]) -> List[str]:
    """Parse guest names from all filenames in the folder to build a global catalog."""
    names: List[str] = []
    for pat in patterns:
        for p in input_dir.glob(pat):
            stem = safe_stem(str(p))
            if SKIP_TITLES_RE.match(stem):
                continue
            names.extend(parse_guests_from_filename(stem))
    return _dedupe_names(names)

def build_catalog_from_db() -> List[str]:
    """Collect guests from already-ingested episodes in DB."""
    try:
        with with_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT jsonb_array_elements_text(guests) FROM episodes")
            rows = cur.fetchall()
    except Exception:
        return []
    names = [r[0] for r in rows if r and r[0]]
    return _dedupe_names(names)

def load_guest_csv(csv_path: Optional[str], explicit_delim: Optional[str] = None, prefer_col: Optional[str] = None) -> List[str]:
    """
    Load 'almost all guests' CSV robustly.
    - Tries csv.Sniffer with common delimiters.
    - If sniff fails, guesses from counts of , ; \t | in the sample.
    - Allows explicit delimiter override via CLI.
    - Picks a column named like 'name', 'guest', 'guest_name', 'full_name', 'person', or falls back to first column.
    """
    if not csv_path:
        return []
    p = Path(csv_path).expanduser()
    if not p.exists():
        print(f"[WARN] guest CSV not found: {p}")
        return []

    # If someone gave us an Excel file by mistake, ask them to export as CSV
    if p.suffix.lower() in {".xlsx", ".xls"}:
        print(f"[WARN] {p.name} looks like an Excel file. Please export to CSV, then re-run.")
        return []

    # Read a small sample and the whole file with UTF-8 (strip BOM if present)
    text = p.read_text(encoding="utf-8-sig", errors="replace")

    # 1) Decide delimiter
    delim = None
    if explicit_delim:
        delim = {"\\t":"\t"}.get(explicit_delim, explicit_delim)
    else:
        # Try Sniffer on common delimiters
        try:
            sniffer = csv.Sniffer()
            sample = text[:4096]
            delim = sniffer.sniff(sample, delimiters=",;\t|").delimiter
            # if the sniffer returns something odd, keep going
        except Exception:
            delim = None

        if not delim:
            # Fallback: pick the char with highest count among common candidates
            sample = text[:4096]
            candidates = [(",", sample.count(",")),
                          (";", sample.count(";")),
                          ("\t", sample.count("\t")),
                          ("|", sample.count("|"))]
            # prefer comma on ties
            candidates.sort(key=lambda x: (x[1], x[0] != ","), reverse=True)
            delim = candidates[0][0] if candidates[0][1] > 0 else ","

    # 2) Parse rows
    rows: List[List[str]] = []
    reader = csv.reader(text.splitlines(), delimiter=delim)
    for r in reader:
        # skip completely empty rows
        if not any(c.strip() for c in r):
            continue
        rows.append([c.strip() for c in r])

    if not rows:
        print(f"[WARN] guest CSV {p} parsed zero rows with delimiter {repr(delim)}")
        return []

    # 3) Header detection
    header = []
    has_header = False
    if rows:
        # simple heuristic: if all cells in first row are non-numeric / contain letters, treat as header
        header = rows[0]
        if any(re.search(r"[A-Za-z]", c or "") for c in header):
            has_header = True

    # 4) Choose column(s)
    start_idx = 1 if has_header else 0
    indices: List[int] = []
    if has_header:
        # prefer an explicit column name if provided
        if prefer_col:
            lowered = [h.strip().lower() for h in header]
            if prefer_col.strip().lower() in lowered:
                indices = [lowered.index(prefer_col.strip().lower())]
        if not indices:
            preferred = {"name","guest","guest_name","full_name","person"}
            lowered = [h.strip().lower() for h in header]
            for i, h in enumerate(lowered):
                if h in preferred or "name" in h:
                    indices.append(i)
    if not indices:
        indices = [0]  # fallback: first column

    # 5) Collect, clean, de-dupe (case-insensitive) and drop hosts
    names: List[str] = []
    for r in rows[start_idx:]:
        for i in indices:
            if i < len(r):
                val = (r[i] or "").strip()
                if val:
                    names.append(val)

    # de-dupe + remove hosts
    hostset = {h.lower() for h in HOSTS}
    seen, uniq = set(), []
    for n in names:
        k = n.lower()
        if k in seen or k in hostset:
            continue
        uniq.append(n)
        seen.add(k)

    print(f"[CSV] Loaded {len(uniq)} unique guest names from {p.name} using delimiter {repr(delim)}{' with header' if has_header else ''}.")
    return uniq


def _dedupe_names(names: List[str]) -> List[str]:
    # remove hosts; de-dupe case-insensitive
    hostset = {h.lower() for h in HOSTS}
    seen, uniq = set(), []
    for n in names:
        if not n:
            continue
        k = n.strip().lower()
        if k in hostset or k in seen:
            continue
        uniq.append(n.strip())
        seen.add(k)
    return uniq

def _name_tokens(n: str) -> List[str]:
    return [t for t in re.split(r"[^A-Za-z]+", n) if t]

def _normalize(n: str) -> str:
    return re.sub(r"\s+", " ", n).strip().lower()

def make_catalog_indexes(guest_catalog: List[str]):
    """Precompute dictionaries for quick/fuzzy matching."""
    full_lower = {_normalize(n): n for n in guest_catalog}
    last_to_full: Dict[str, List[str]] = {}
    first_to_full: Dict[str, List[str]] = {}
    for full in guest_catalog:
        toks = _name_tokens(full)
        if toks:
            first = toks[0].lower()
            last = toks[-1].lower()
            first_to_full.setdefault(first, []).append(full)
            last_to_full.setdefault(last, []).append(full)
    return full_lower, first_to_full, last_to_full

def resolve_guest_name(ent_text: str,
                       full_lower_map: Dict[str, str],
                       first_to_full: Dict[str, List[str]],
                       last_to_full: Dict[str, List[str]],
                       fuzzy_threshold: float = 0.85) -> Optional[str]:
    """
    Smart resolution:
      1) exact full-name match (case-insensitive)
      2) unique last-name match
      3) light fuzzy on full names (difflib ratio >= fuzzy_threshold)
    """
    raw = ent_text or ""
    if not raw.strip():
        return None
    norm = _normalize(raw)

    # 1) exact full match
    if norm in full_lower_map:
        return full_lower_map[norm]

    toks = _name_tokens(raw)
    # 2) single-token unique last name
    if len(toks) == 1:
        last = toks[0].lower()
        options = last_to_full.get(last, [])
        if len(options) == 1:
            return options[0]

    # 3) fuzzy on full names
    best_name, best_score = None, 0.0
    for canon_norm, canon_full in full_lower_map.items():
        score = difflib.SequenceMatcher(None, norm, canon_norm).ratio()
        if score > best_score:
            best_score, best_name = score, canon_full
    if best_score >= fuzzy_threshold:
        return best_name

    return None

def _mk_phrases_list(s: str) -> list[str]:
    return [p.strip().lower() for p in (s or "").split(",") if p.strip()]

def _contains_any(text: str, phrases: list[str]) -> bool:
    t = (text or "").lower()
    return any(p in t for p in phrases)

def detect_ad_blocks(utterances, intro_phrases: list[str], start_phrases: list[str],
                     resume_phrases: list[str], ad_max_ms: int = 6*60_000,
                     end_nonad_needed: int = 2):
    """
    Heuristic: open an ad block when we see an INTRO phrase or clear START phrase.
    Keep it open while ad-like lines continue; close on RESUME phrase OR
    after N consecutive non-ad lines OR time cap.
    Returns list of (start_ms, end_ms) ranges (inclusive/exclusive) in milliseconds.
    """
    blocks = []
    if not utterances:
        return blocks

    us = sorted(utterances, key=lambda u: (u.get("start") or 0))
    open_start = None
    nonad_streak = 0

    def is_ad_like(u):
        txt = (u.get("text") or "")
        return _contains_any(txt, start_phrases)  # tight signal

    for i, u in enumerate(us):
        st, en = u.get("start") or 0, u.get("end") or 0
        txt = (u.get("text") or "")

        # open if not open and an intro/start cue appears
        if open_start is None:
            if _contains_any(txt, intro_phrases) or _contains_any(txt, start_phrases):
                open_start = st
                nonad_streak = 0
                continue
        else:
            # if we are inside an ad block…
            if _contains_any(txt, resume_phrases):
                # close on explicit resume
                blocks.append((open_start, en))
                open_start, nonad_streak = None, 0
                continue

            if is_ad_like(u):
                nonad_streak = 0
            else:
                nonad_streak += 1
                if nonad_streak >= end_nonad_needed:
                    # close when show content clearly resumes
                    blocks.append((open_start, st))
                    open_start, nonad_streak = None, 0
                    continue

            # safety cap: long ad sequences
            if st - open_start >= ad_max_ms:
                blocks.append((open_start, st))
                open_start, nonad_streak = None, 0

    # if an ad block was open at the end, close it at last seen time
    if open_start is not None:
        last_end = us[-1].get("end") or us[-1].get("start") or open_start
        blocks.append((open_start, last_end))

    # merge overlapping/adjacent ranges
    blocks.sort()
    merged = []
    for s, e in blocks:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(s, e) for s, e in merged]

def in_any_block(st_ms: int, en_ms: int, blocks: list[tuple[int,int]]) -> bool:
    for s, e in blocks:
        if en_ms > s and st_ms < e:  # overlap
            return True
    return False


# -----------------------
# Postgres / pgvector
# -----------------------
DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS episodes (
  id BIGSERIAL PRIMARY KEY,
  file_stem TEXT UNIQUE,
  file_path TEXT,
  guests JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS utterances (
  id BIGSERIAL PRIMARY KEY,
  episode_id BIGINT REFERENCES episodes(id) ON DELETE CASCADE,
  speaker TEXT,
  start_ms INT,
  end_ms INT,
  text TEXT
);

CREATE TABLE IF NOT EXISTS mentions (
  id BIGSERIAL PRIMARY KEY,
  episode_id BIGINT REFERENCES episodes(id) ON DELETE CASCADE,
  speaker TEXT,
  about_person TEXT,
  quote TEXT,
  start_ms INT,
  end_ms INT
);

CREATE TABLE IF NOT EXISTS chunks (
  id BIGSERIAL PRIMARY KEY,
  episode_id BIGINT REFERENCES episodes(id) ON DELETE CASCADE,
  speaker TEXT,
  start_ms INT,
  end_ms INT,
  text TEXT,
  embedding vector(384)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_ivf ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""

def _connect_via_url():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return None
    if "sslmode=" not in db_url:
        sep = "&" if "?" in db_url else "?"
        db_url = f"{db_url}{sep}sslmode=require"
    return psycopg2.connect(db_url)

def _connect_via_parts():
    host = os.environ.get("PGHOST")
    db   = os.environ.get("PGDATABASE")
    user = os.environ.get("PGUSER")
    pw   = os.environ.get("PGPASSWORD")
    if not all([host, db, user, pw]):
        return None
    port = int(os.environ.get("PGPORT", "5432"))
    return psycopg2.connect(
        host=host, dbname=db, user=user, password=pw,
        port=port, sslmode="require"
    )

def with_conn():
    conn = _connect_via_url() or _connect_via_parts()
    if conn is None:
        raise RuntimeError("Missing database config. Set DATABASE_URL or PGHOST/PGDATABASE/PGUSER/PGPASSWORD.")
    return conn

def ensure_schema():
    with with_conn() as conn, conn.cursor() as cur:
        cur.execute(DDL)
        conn.commit()

def episode_exists(file_stem: str) -> Optional[int]:
    with with_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM episodes WHERE file_stem=%s", (file_stem,))
        row = cur.fetchone()
        return row[0] if row else None

def upsert_episode(file_stem: str, file_path: str, guests: List[str]) -> int:
    with with_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO episodes (file_stem, file_path, guests)
            VALUES (%s, %s, %s)
            ON CONFLICT (file_stem) DO UPDATE SET file_path=EXCLUDED.file_path, guests=EXCLUDED.guests
            RETURNING id
        """, (file_stem, file_path, json.dumps(guests)))
        eid = cur.fetchone()[0]
        conn.commit()
        return eid

def clear_episode_rows(episode_id: int):
    with with_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM utterances WHERE episode_id=%s", (episode_id,))
        cur.execute("DELETE FROM mentions   WHERE episode_id=%s", (episode_id,))
        cur.execute("DELETE FROM chunks     WHERE episode_id=%s", (episode_id,))
        conn.commit()

def insert_utterances(episode_id: int, rows: List[Tuple[str,int,int,str]]):
    with with_conn() as conn, conn.cursor() as cur:
        if not rows:
            return
        execute_values(cur, """
            INSERT INTO utterances (episode_id, speaker, start_ms, end_ms, text)
            VALUES %s
        """, [(episode_id, *r) for r in rows])
        conn.commit()

def insert_mentions(episode_id: int, rows: List[Tuple[str,str,str,int,int]]):
    with with_conn() as conn, conn.cursor() as cur:
        if not rows:
            return
        execute_values(cur, """
            INSERT INTO mentions (episode_id, speaker, about_person, quote, start_ms, end_ms)
            VALUES %s
        """, [(episode_id, *r) for r in rows])
        conn.commit()

def insert_chunks_with_embeddings(episode_id: int, rows: List[Tuple[str,int,int,str]]):
    if not rows:
        return
    texts = [r[3] for r in rows]
    vecs  = emb_model().encode(texts, normalize_embeddings=True).tolist()
    vec_literals = ["[" + ",".join(f"{x:.6f}" for x in v) + "]" for v in vecs]
    with with_conn() as conn, conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO chunks (episode_id, speaker, start_ms, end_ms, text, embedding)
            VALUES %s
        """, [(episode_id, r[0], r[1], r[2], r[3], vec_literals[i]) for i, r in enumerate(rows)])
        conn.commit()

# -----------------------
# AssemblyAI: transcribe (returns (result_json, transcript_id))
# -----------------------
def _is_credit_error(resp: requests.Response) -> bool:
    """Return True if the response indicates an out-of-credits / payment error."""
    if resp.status_code == 402:
        return True
    try:
        msg = resp.json().get("error", "").lower()
        return any(p in msg for p in ("credit", "payment", "billing", "limit reached", "quota"))
    except Exception:
        return False


def _upload_file(file_path: str) -> str:
    """Upload file to AssemblyAI CDN, rotating API keys on credit errors. Returns upload_url."""
    global API_KEY
    while True:
        headers = {"authorization": API_KEY}
        with open(file_path, "rb") as f:
            up = requests.post(f"{BASE_URL}/v2/upload", headers=headers, data=f, timeout=600)
        if _is_credit_error(up):
            print(f"[keys] Key {_key_index + 1} out of credits during upload — rotating.")
            _rotate_api_key()
            continue
        up.raise_for_status()
        return up.json()["upload_url"]


def transcribe(file_path: str) -> Tuple[dict, str]:
    global API_KEY

    # Upload with current key (CDN URL is account-scoped, so re-upload after key rotation)
    audio_url = _upload_file(file_path)

    # Submit transcript job — re-upload and retry with next key on credit/auth errors
    while True:
        headers = {"authorization": API_KEY}
        payload = {
            "audio_url": audio_url,
            "speech_models": ["universal-3-pro"],
            "speaker_labels": True,
            "speakers_expected": 4,
            "entity_detection": True
        }
        resp = requests.post(f"{BASE_URL}/v2/transcript", json=payload, headers=headers, timeout=60)
        if resp.status_code in (400, 403) or _is_credit_error(resp):
            body = resp.json().get("error", resp.text) if resp.content else resp.text
            print(f"[keys] Transcript submission failed (HTTP {resp.status_code}): {body}")
            if _key_index + 1 < len(_ALL_API_KEYS):
                _rotate_api_key()
                # Re-upload under the new key so the CDN URL is accessible
                print(f"[keys] Re-uploading file under new key...")
                audio_url = _upload_file(file_path)
                continue
            resp.raise_for_status()
        resp.raise_for_status()
        break
    tid = resp.json()["id"]

    # 3) Poll
    endpoint = f"{BASE_URL}/v2/transcript/{tid}"
    while True:
        headers = {"authorization": API_KEY}
        r = requests.get(endpoint, headers=headers, timeout=60)
        r.raise_for_status()
        res = r.json()
        if res.get("status") == "completed":
            return res, tid
        if res.get("status") == "error":
            raise RuntimeError(res.get("error"))
        time.sleep(3)


def transcribe_from_url(audio_url: str) -> Tuple[dict, str]:
    """Like transcribe() but skips the upload step — passes the URL directly to AssemblyAI.
    Use this for RSS/CDN audio URLs that AssemblyAI can fetch itself."""
    headers = {"authorization": API_KEY}
    payload = {
        "audio_url": audio_url,
        "speech_models": ["universal-3-pro"],
        "speaker_labels": True,
        "speakers_expected": 4,
        "entity_detection": True
    }
    resp = requests.post(f"{BASE_URL}/v2/transcript", json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    tid = resp.json()["id"]

    endpoint = f"{BASE_URL}/v2/transcript/{tid}"
    while True:
        r = requests.get(endpoint, headers=headers, timeout=60)
        r.raise_for_status()
        res = r.json()
        if res.get("status") == "completed":
            return res, tid
        if res.get("status") == "error":
            raise RuntimeError(res.get("error"))
        time.sleep(3)


def process_one_from_url(stem: str,
                         audio_url: str,
                         global_guest_catalog: Optional[List[str]] = None,
                         intro_skip_min: int = 3,
                         min_guest_talk_sec: int = 60) -> Dict:
    """Process a single episode from a remote audio URL (no local file needed).
    stem: episode name / file stem (used as DB key)
    audio_url: direct link to the audio file (e.g. from RSS enclosure)
    """
    if SKIP_TITLES_RE.match(stem):
        return {"stem": stem, "skipped": True, "reason": "Armchair Anonymous"}

    existing_id = episode_exists(stem)
    if existing_id:
        return {"stem": stem, "skipped": True, "episode_id": existing_id, "reason": "already_ingested"}

    print(f"[process_one_from_url] Transcribing: {stem}")
    result, transcript_id = transcribe_from_url(audio_url)

    utterances = result.get("utterances") or []
    entities = [e for e in result.get("entities", []) if e.get("entity_type") == "person_name"]
    ad_blocks = []  # ad detection not used in URL-based processing

    # Host detection via LeMUR
    fixed_host_labels = {}
    present_hosts = {"dax", "monica"}
    try:
        hd = lemur_map_hosts(utterances, transcript_id, API_KEY, max_minutes=15)
        ph = set(hd.get("present_hosts", []))
        conf = hd.get("confidence", {})
        lab = hd.get("labels", {})
        fixed = {k: lab[k] for k in ("dax", "monica") if float(conf.get(k, 0)) >= 0.6 and lab.get(k)}
        if ph:
            present_hosts = ph
        fixed_host_labels = fixed
        print(f"[LeMUR host-detect] present={sorted(present_hosts)} fixed={fixed_host_labels}")
    except Exception as e:
        print(f"[LeMUR host-detect] failed: {e}")

    parsed_guests = parse_guests_from_filename(stem)
    label_to_name = map_hosts_and_guests(
        utterances,
        parsed_guests,
        present_hosts,
        fixed_host_labels=fixed_host_labels,
        intro_skip_ms=intro_skip_min * 60_000,
        min_guest_talk_ms=min_guest_talk_sec * 1_000,
    )

    utt_rows: List[Tuple[str, int, int, str]] = []
    for u in utterances:
        spk = label_to_name.get(u.get("speaker", ""), u.get("speaker", "SPK"))
        utt_rows.append((spk, u.get("start") or -1, u.get("end") or -1, (u.get("text") or "").strip()))

    global_catalog = global_guest_catalog or []
    full_lower, first_to_full, last_to_full = make_catalog_indexes(global_catalog)

    mention_rows: List[Tuple[str, str, str, int, int]] = []
    for ent in entities:
        raw_name = (ent.get("text") or "").strip()
        s, e = ent.get("start"), ent.get("end")
        if not raw_name or s is None or e is None:
            continue
        u = best_overlapping_utterance(utterances, s, e)
        if not u:
            continue
        speaker_label = u.get("speaker", "")
        speaker_name = label_to_name.get(speaker_label, speaker_label)
        if speaker_name.lower() in {"dax shepard", "monica padman"}:
            continue
        resolved = resolve_guest_name(raw_name, full_lower, first_to_full, last_to_full)
        # If not in known catalog, accept the raw name if it looks like a full person name
        # (2+ words, no digits, not a host) so new guests get picked up automatically
        if not resolved:
            parts = raw_name.strip().split()
            if (len(parts) >= 2
                    and all(p[0].isupper() for p in parts if p)
                    and not any(c.isdigit() for c in raw_name)
                    and raw_name.lower() not in {"dax shepard", "monica padman"}):
                resolved = raw_name.strip()
            else:
                continue
        self_guest = speaker_name.replace("Guest:", "").strip().lower()
        if resolved.lower() == self_guest:
            continue
        quote_text = (u.get("text") or "").strip()
        mention_rows.append((speaker_name, resolved, quote_text, u.get("start") or -1, u.get("end") or -1))

    try:
        lemur_rows = lemur_extract_mentions(label_to_name, utterances, global_catalog, API_KEY, transcript_id,
                                            full_lower=full_lower, first_to_full=first_to_full, last_to_full=last_to_full)
    except Exception as e:
        print(f"LeMUR pass failed: {e}")
        lemur_rows = []

    seen = set((s, a, q, st) for (s, a, q, st, en) in mention_rows)
    added = 0
    for row in lemur_rows:
        key = (row[0], row[1], row[2], row[3])
        if key not in seen:
            mention_rows.append(row)
            seen.add(key)
            added += 1

    eid = upsert_episode(stem, audio_url, parsed_guests)
    insert_utterances(eid, utt_rows)
    insert_mentions(eid, mention_rows)
    insert_chunks_with_embeddings(eid, utt_rows)

    return {
        "stem": stem,
        "skipped": False,
        "episode_id": eid,
        "utterances": len(utt_rows),
        "mentions": len(mention_rows),
        "lemur_added": added
    }


# -----------------------
# LeMUR hybrid pass (uses transcript_ids for context)
# -----------------------
def lemur_extract_mentions(label_to_name, utterances, global_guest_catalog, api_key, transcript_id: Optional[str],
                           full_lower=None, first_to_full=None, last_to_full=None) -> List[Tuple[str,str,str,int,int]]:
    """
    Build enumerated, speaker-labeled transcript for indexing;
    Ask LeMUR (with transcript_ids) to return STRICT JSON:
    { "mentions": [ { "utterance_index": int, "speaker": "...", "about": "...", "quote": "..." } ] }
    Map utterance_index to (start_ms, end_ms). Returns list of tuples:
    (speaker_name, about_person, quote, start_ms, end_ms)
    """
    # Build catalog indexes if not provided
    if full_lower is None or first_to_full is None or last_to_full is None:
        full_lower, first_to_full, last_to_full = make_catalog_indexes(global_guest_catalog or [])

    # 1) Enumerated lines (ties JSON back to timestamps)
    lines = []
    index_to_span = {}
    for i, u in enumerate(utterances, 1):
        spk = label_to_name.get(u.get("speaker",""), u.get("speaker","SPK"))
        txt = (u.get("text") or "").strip()
        st, en = u.get("start"), u.get("end")
        if not txt:
            continue
        lines.append(f"[{i}]({st}-{en}) {spk}: {txt}")
        index_to_span[i] = (st or -1, en or -1)

    if not lines:
        return []

    # 2) Prompt with known guest list + strict JSON schema
    known_guests = ", ".join(sorted(set(global_guest_catalog), key=str.lower))[:3000]
    prompt = f"""
You are extracting "person mentions" from a podcast transcript.

Rules:
- Two hosts: "Dax Shepard" and "Monica Padman". Never include them as speaker or about_person.
- Find every moment where a non-host speaker mentions ANY real person by name (celebrity, public figure, friend, family member, anyone).
- Do NOT limit to known guests only — find ALL person name mentions.
- Known guests list (use for canonical name resolution if the person is on the list): {known_guests}
- The transcript lines below are enumerated: [index](start_ms-end_ms) SpeakerName: text

Return STRICT JSON (and nothing else):
{{
  "mentions": [
    {{
      "utterance_index": <integer index from the lines>,
      "speaker": "<exact SpeakerName from the line>",
      "about": "<Full Name of the person being mentioned — use canonical form from known guests list if they appear there, otherwise use the name as spoken>",
      "quote": "<the minimal quote span from that utterance that contains the mention>"
    }}
  ]
}}

Constraints:
- Use the given index; do not invent indexes.
- "speaker" must NOT be Dax Shepard or Monica Padman.
- "about" must NOT be Dax Shepard or Monica Padman.
- "about" must NOT equal the speaker (no self-mentions).
- The "quote" must be at least 10 words long and contain meaningful context around the mention.
- Only include confident person name mentions — skip pronouns, vague references, or unclear names.

Transcript (excerpt with indexes):
{chr(10).join(lines)}
""".strip()

    # 3) Call LeMUR
    headers = {"authorization": api_key, "content-type": "application/json"}
    body = {
        "prompt": prompt,
        "final_model": os.environ.get("LEMUR_MODEL", "anthropic/claude-3-haiku"),
        "temperature": 0,
        "max_output_size": 3000,
    }
    body["input_text"] = ""  # transcript text is embedded directly in the prompt

    url = "https://api.assemblyai.com/lemur/v3/generate/task"
    r = requests.post(url, headers=headers, json=body, timeout=180)
    r.raise_for_status()
    resp_text = r.json().get("response", "")

    if resp_text is None:
        raise RuntimeError(f"LeMUR request failed: empty response")

    # 4) Parse STRICT JSON payload
    try:
        data = json.loads(resp_text)
    except Exception:
        m = re.search(r"\{.*\}", resp_text, re.S)
        if not m:
            return []
        data = json.loads(m.group(0))

    HOSTS_LOWER = {"dax shepard", "monica padman"}
    out: List[Tuple[str,str,str,int,int]] = []
    for m in data.get("mentions", []):
        idx = m.get("utterance_index")
        spk = (m.get("speaker") or "").strip()
        about = (m.get("about") or "").strip()
        quote = (m.get("quote") or "").strip()
        if not isinstance(idx, int) or not spk or not about or not quote:
            continue
        if len(quote) < 25:
            continue
        st, en = index_to_span.get(idx, (-1, -1))
        # Skip hosts on either side
        if spk.lower() in HOSTS_LOWER or about.lower() in HOSTS_LOWER:
            continue
        # Skip self-mentions
        if spk.lower() == about.lower():
            continue
        # Try to canonicalize against known guests, fall back to raw name
        canonical = resolve_guest_name(about, full_lower, first_to_full, last_to_full)
        about = canonical if canonical else about
        out.append((spk, about, quote, st, en))
    return out

def lemur_map_hosts(utterances, transcript_id: Optional[str], api_key: str, max_minutes: int = 15):
    """
    Ask LeMUR to determine:
      - Which hosts (dax, monica) are present in this episode
      - Which diarization label (A/B/C/...) maps to each host (if confident)
    Returns a dict:
      {
        "present_hosts": ["dax", "monica"]   # subset, order not important
        "labels": {"dax": "A", "monica": "C"},   # only when confident
        "confidence": {"dax": 0.82, "monica": 0.12}
      }
    """
    # Build a brief, indexed excerpt (first N minutes) so the model can use textual cues
    # (LLMs can't hear voices; they rely on lines like "I'm Dax", "Monica asks", etc.)
    ms_limit = max_minutes * 60 * 1000
    excerpt_lines = []
    for i, u in enumerate(sorted(utterances or [], key=lambda x: (x.get("start") or 0)), 1):
        st, en = u.get("start") or 0, u.get("end") or 0
        if st > ms_limit:
            break
        lab = u.get("speaker", "SPK")
        txt = (u.get("text") or "").strip()
        if not txt:
            continue
        excerpt_lines.append(f"[{i}]({st}-{en}) {lab}: {txt}")

    if not excerpt_lines:
        return {"present_hosts": [], "labels": {}, "confidence": {}}

    prompt = f"""
You will identify Armchair Expert hosts in a diarized transcript excerpt.

Hosts:
- Dax Shepard -> key "dax"
- Monica Padman -> key "monica"

Transcript lines use diarization labels like A, B, C:
  [index](start_ms-end_ms) LABEL: text

Tasks:
1) Decide which hosts are present in this episode excerpt (dax, monica). If a host is clearly not present, omit them.
2) If present, map each host to the most likely LABEL (A/B/C/...), using textual cues (e.g., "I'm Dax", on-air intros, how other speakers address them).
3) Provide a confidence 0.0–1.0 for each host label assignment.

Return STRICT JSON only:
{{
  "present_hosts": ["dax", "monica"],     // subset; omit any not present
  "labels": {{ "dax": "A", "monica": "C" }},  // include a key only if reasonably confident; else omit
  "confidence": {{ "dax": 0.82, "monica": 0.55 }}
}}

Transcript excerpt:
{chr(10).join(excerpt_lines)}
""".strip()

    # Call LeMUR
    headers = {"authorization": api_key, "content-type": "application/json"}
    body = {
        "prompt": prompt,
        "final_model": os.environ.get("LEMUR_MODEL", "anthropic/claude-3-haiku"),
        "temperature": 0,
        "max_output_size": 2000,
    }
    body["input_text"] = ""  # transcript text is embedded directly in the prompt

    url = "https://api.assemblyai.com/lemur/v3/generate/task"
    r = requests.post(url, headers=headers, json=body, timeout=120)
    r.raise_for_status()
    raw_resp = r.json().get("response", "")

    if raw_resp is None:
        raise RuntimeError("LeMUR host-detect failed: empty response")

    # Parse STRICT JSON
    try:
        data = json.loads(raw_resp)
    except Exception:
        import re as _re
        m = _re.search(r"\{.*\}", raw_resp, _re.S)
        data = json.loads(m.group(0)) if m else {"present_hosts": [], "labels": {}, "confidence": {}}

    # Normalize types
    present = data.get("present_hosts") or []
    labels  = data.get("labels") or {}
    conf    = data.get("confidence") or {}
    return {
        "present_hosts": [h for h in present if h in ("dax","monica")],
        "labels": {k:v for k,v in labels.items() if k in ("dax","monica")},
        "confidence": {k: float(conf.get(k, 0)) for k in ("dax","monica")}
    }

DEFAULT_AD_KEYWORDS = [
    "promo code", "use code", "sponsor", "sponsored", "brought to you by",
    "dot com", ".com", "slash", "visit", "shop", "free trial", "free shipping",
    "offer", "terms apply", "learn more at"
]

def compute_label_stats(utterances, intro_skip_ms: int, ad_keywords=None, ignore_ranges: list[tuple[int,int]] = None):
    ad_keywords = [k.lower() for k in (ad_keywords or [])]
    ignore_ranges = ignore_ranges or []
    stats = {}
    for u in utterances or []:
        st, en = u.get("start") or 0, u.get("end") or 0
        if in_any_block(st, en, ignore_ranges):
            continue  # <-- ignore ad blocks entirely for mapping stats
        lab = u.get("speaker")
        if not lab:
            continue
        dur = max(0, en - st)
        s = stats.setdefault(lab, {"dur": 0, "dur_after": 0, "first": None, "first_after": None, "ad_hits": 0})
        s["dur"] += dur
        if st >= intro_skip_ms:
            s["dur_after"] += dur
            if s["first_after"] is None:
                s["first_after"] = st
        txt = (u.get("text") or "").lower()
        s["ad_hits"] += sum(1 for kw in ad_keywords if kw in txt)
    return stats


# -----------------------
# Single-file pipeline
# -----------------------
def process_one(file_path: str,
                skip_anonymous: bool = True,
                reprocess: bool = False,
                global_guest_catalog: Optional[List[str]] = None,
                present_hosts: Optional[set] = None,
                ad_detect: bool = False,
                ad_intro: Optional[List[str]] = None,
                ad_start: Optional[List[str]] = None,
                ad_resume: Optional[List[str]] = None,
                ad_max_ms: int = 360000,
                ad_end_nonad: int = 2,
                intro_skip_ms: int = 180_000,
                min_guest_talk_ms: int = 60_000) -> Dict:

    file_path = os.path.abspath(file_path)
    stem = safe_stem(file_path)
    present_hosts = present_hosts or {"dax", "monica"}

    if skip_anonymous and SKIP_TITLES_RE.match(stem):
        return {"file": file_path, "stem": stem, "skipped": True, "reason": "Armchair Anonymous"}

    existing_id = episode_exists(stem)
    if existing_id and not reprocess:
        return {"file": file_path, "stem": stem, "skipped": True, "episode_id": existing_id, "reason": "already_ingested"}

    # --- Transcribe via AssemblyAI ---
    result, transcript_id = transcribe(file_path)

    utterances = result.get("utterances") or []
    entities = [e for e in result.get("entities", []) if e.get("entity_type") == "person_name"]

    ad_blocks = []
    if ad_detect:
        ad_blocks = detect_ad_blocks(
            utterances,
            ad_intro or [],
            ad_start or [],
            ad_resume or [],
            ad_max_ms=ad_max_ms,
            end_nonad_needed=ad_end_nonad
        )
        if ad_blocks:
            print(f"[ads] detected {len(ad_blocks)} block(s): {ad_blocks}")


    # present_hosts is passed in; None means "auto"
    fixed_host_labels = {}
    if present_hosts is None:
        try:
            hd = lemur_map_hosts(utterances, transcript_id, API_KEY, max_minutes=15)
            # Use LeMUR's presence and labels when confident
            ph = set(hd.get("present_hosts", []))
            conf = hd.get("confidence", {})
            lab  = hd.get("labels", {})

            # accept a host label only if conf >= 0.6
            fixed = {}
            for k in ("dax","monica"):
                if float(conf.get(k, 0)) >= 0.6 and lab.get(k):
                    fixed[k] = lab[k]

            if ph:
                present_hosts = ph
            fixed_host_labels = fixed
            print(f"[LeMUR host-detect] present={sorted(list(present_hosts or []))} fixed={fixed_host_labels}")
        except Exception as e:
            print(f"[LeMUR host-detect] failed: {e}")
            # fall back to both hosts if auto fails
            present_hosts = {"dax","monica"}

    parsed_guests = parse_guests_from_filename(stem)

    # Final speaker map
    label_to_name = map_hosts_and_guests(
        utterances,
        parsed_guests,
        present_hosts or {"dax","monica"},
        fixed_host_labels=fixed_host_labels,             # keep if you added auto-host
        intro_skip_ms=intro_skip_ms,
        min_guest_talk_ms=min_guest_talk_ms,
    )



    # Speaker mapping

    # Utterance rows
    utt_rows: List[Tuple[str,int,int,str]] = []
    for u in utterances:
        spk = label_to_name.get(u.get("speaker",""), u.get("speaker","SPK"))
        utt_rows.append((spk, u.get("start") or -1, u.get("end") or -1, (u.get("text") or "").strip()))

    # ----- Baseline mentions from Entity Detection (deterministic) -----
    global_catalog = global_guest_catalog or []
    full_lower, first_to_full, last_to_full = make_catalog_indexes(global_catalog)

    mention_rows: List[Tuple[str,str,str,int,int]] = []
    for ent in entities:
        raw_name = (ent.get("text") or "").strip()
        s, e = ent.get("start"), ent.get("end")
        if not raw_name or s is None or e is None:
            continue
        u = best_overlapping_utterance(utterances, s, e)
        if not u:
            continue
        if ad_blocks and in_any_block(u.get("start") or 0, u.get("end") or 0, ad_blocks):
            continue  # don't record mentions from ads

        speaker_label = u.get("speaker","")
        speaker_name = label_to_name.get(speaker_label, speaker_label)
        # only non-host speakers count as "guest speaker"
        if speaker_name.lower() in {"dax shepard", "monica padman"}:
            continue
        # resolve entity to canonical guest name (smart)
        resolved = resolve_guest_name(raw_name, full_lower, first_to_full, last_to_full)
        if not resolved:
            continue
        # avoid self-mentions (guest saying their own name)
        self_guest = speaker_name.replace("Guest:", "").strip().lower()
        if resolved.lower() == self_guest:
            continue
        quote_text = (u.get("text") or "").strip()
        mention_rows.append((speaker_name, resolved, quote_text, u.get("start") or -1, u.get("end") or -1))

    # ----- LeMUR hybrid pass (fills pronouns/nicknames; finds all person names) -----
    try:
        lemur_rows = lemur_extract_mentions(label_to_name, utterances, global_catalog, API_KEY, transcript_id,
                                            full_lower=full_lower, first_to_full=first_to_full, last_to_full=last_to_full)
    except Exception as e:
        print(f"LeMUR pass failed: {e}")
        lemur_rows = []

    # Merge + de-dupe by (speaker, about, quote, start_ms)
    seen = set((s,a,q,st) for (s,a,q,st,en) in mention_rows)
    added = 0
    for row in lemur_rows:
        key = (row[0], row[1], row[2], row[3])
        if key not in seen:
            mention_rows.append(row)
            seen.add(key)
            added += 1

    # Upsert + insert
    eid = upsert_episode(stem, file_path, parsed_guests)
    clear_episode_rows(eid)
    insert_utterances(eid, utt_rows)
    insert_mentions(eid, mention_rows)
    insert_chunks_with_embeddings(eid, utt_rows)

    return {
        "file": file_path,
        "stem": stem,
        "skipped": False,
        "episode_id": eid,
        "utterances": len(utt_rows),
        "mentions": len(mention_rows),
        "lemur_added": added
    }

# -----------------------
# Batch main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Batch transcribe → Postgres + pgvector (AssemblyAI + LeMUR hybrid)")
    parser.add_argument("--input-dir", type=str, help="Folder of audio files (mp3/m4a/wav/flac). If omitted, uses file_path variable.")
    parser.add_argument("--patterns", type=str, default="*.mp3,*.m4a,*.wav,*.flac",
                        help="Comma-separated glob patterns (default: *.mp3,*.m4a,*.wav,*.flac)")
    parser.add_argument("--guest-csv", type=str, default="",
                        help="CSV containing guest names (smart/fuzzy used; not exact-only).")
    parser.add_argument("--reprocess", action="store_true", help="Force reprocess even if already ingested.")
    parser.add_argument("--include-anonymous", action="store_true", help="Include 'Armchair Anonymous' episodes.")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N files (0 = no limit).")
    parser.add_argument("--guest-csv-delim", type=str, default="",
    help="Delimiter override for guest CSV. One of ',', ';', '\\t', '|'")
    parser.add_argument("--guest-csv-col", type=str, default="",
    help="Column name in the guest CSV that holds full names (default: auto-detect or first column)")
    parser.add_argument(
    "--hosts-present",
    type=str,
    default="auto",  # 'auto' = use LeMUR; or pass 'dax', 'dax,monica', etc.
    help="Comma list from {dax,monica} or 'auto' to detect via LeMUR.")
    parser.add_argument("--ad-detect", action="store_true",
    help="Detect ad blocks (e.g., 'stay tuned for more armchair expert', 'we are supported by...') and ignore them for mapping & mentions.")
    parser.add_argument("--ad-intro-phrases", type=str,
        default="stay tuned for more armchair expert,stay tuned for more armchair",
        help="Comma-separated phrases that typically INTRODUCE an ad block (case-insensitive).")
    parser.add_argument("--ad-start-phrases", type=str,
        default="we are supported by,this episode is brought to you by,support for this podcast comes from",
        help="Comma-separated phrases that often START individual ads (case-insensitive).")
    parser.add_argument("--ad-resume-phrases", type=str,
        default="and now back to,and we are back,back to the show,back with",
        help="Comma-separated phrases that end the ad block / resume the show (case-insensitive).")
    parser.add_argument("--ad-max-min", type=int, default=6,
        help="Safety cap: maximum minutes to treat as a single ad block before auto-closing (default 6).")
    parser.add_argument("--ad-end-nonad-utter", type=int, default=2,
        help="Close an ad block after this many consecutive non-ad utterances (default 2).")
    parser.add_argument("--intro-skip-min", type=int, default=3,
    help="Ignore the first N minutes when mapping guests (default: 3).")
    parser.add_argument("--min-guest-talk-sec", type=int, default=60,
    help="Require at least this many seconds of talk AFTER the skip window to consider a label a guest (default: 60).")

    args = parser.parse_args()

    raw = (args.hosts_present or "").strip().lower()
    present_hosts = None if raw == "auto" else set(
    h.strip() for h in raw.split(",") if h.strip())

    ad_intro = _mk_phrases_list(args.ad_intro_phrases)
    ad_start = _mk_phrases_list(args.ad_start_phrases)
    ad_resume = _mk_phrases_list(args.ad_resume_phrases)




    ensure_schema()

    # Build file list
    files: List[str] = []
    if args.input_dir:
        base = Path(args.input_dir).expanduser().resolve()
        pats = [p.strip() for p in args.patterns.split(",") if p.strip()]
        for pat in pats:
            files.extend([str(p) for p in base.glob(pat)])
        files.sort()
        # Build catalogs
        file_catalog = build_global_guest_catalog_from_files(base, pats)
    else:
        if not file_path:
            print("No --input-dir and file_path is empty. Provide one of them.")
            return
        files = [file_path]
        file_catalog = parse_guests_from_filename(safe_stem(file_path))

    if args.limit and len(files) > args.limit:
        files = files[:args.limit]

    if not files:
        print("No audio files found. Check --input-dir and --patterns.")
        return

    # Merge catalogs: files + DB + CSV
    db_catalog  = build_catalog_from_db()
    csv_catalog = load_guest_csv(args.guest_csv, args.guest_csv_delim or None, args.guest_csv_col or None) if args.guest_csv else []
    global_catalog = _dedupe_names([*file_catalog, *db_catalog, *csv_catalog])

    print(f"Found {len(files)} file(s) to consider.")
    print(f"Global guest catalog size: {len(global_catalog)} (files={len(file_catalog)}, db={len(db_catalog)}, csv={len(csv_catalog)})")

    skip_anonymous = not args.include_anonymous
    done = 0

    for idx, fp in enumerate(files, 1):
        try:
            res = process_one(
                fp,
                skip_anonymous=skip_anonymous,
                reprocess=args.reprocess,
                global_guest_catalog=global_catalog,
                present_hosts=present_hosts,
                ad_detect=args.ad_detect,
                ad_intro=ad_intro,
                ad_start=ad_start,
                ad_resume=ad_resume,
                ad_max_ms=args.ad_max_min*60_000,
                ad_end_nonad=args.ad_end_nonad_utter,
                intro_skip_ms=args.intro_skip_min * 60_000,
                min_guest_talk_ms=args.min_guest_talk_sec * 1_000,
            )
            if res.get("skipped"):
                reason = res.get("reason", "skipped")
                eid = res.get("episode_id", "-")
                print(f"[{idx}/{len(files)}] SKIP  {res['stem']}  (ep={eid})  [{reason}]")
            else:
                print(f"[{idx}/{len(files)}] OK    {res['stem']}  (ep={res['episode_id']}, "
                      f"utterances={res['utterances']}, mentions={res['mentions']}, lemur+={res.get('lemur_added',0)})")
            done += 1
        except PermissionError as e:
            print(f"[{idx}/{len(files)}] FAIL  {fp}  -> PermissionError: {e}")
        except Exception as e:
            print(f"[{idx}/{len(files)}] FAIL  {fp}  -> {e}")

    print(f"\nFinished. Processed {done}/{len(files)} file(s).")

# ---------- Minimal semantic search helper (optional) ----------
def semantic_search(query: str, k: int = 5):
    qv = emb_model().encode([query], normalize_embeddings=True).tolist()[0]
    qlit = "[" + ",".join(f"{x:.6f}" for x in qv) + "]"
    sql = """
      SELECT id, episode_id, speaker, start_ms, end_ms, text,
             1 - (embedding <#> %s::vector) AS cosine_sim
      FROM chunks
      ORDER BY embedding <#> %s::vector
      LIMIT %s
    """
    with with_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (qlit, qlit, k))
        return cur.fetchall()

# -----------------------
# DB connection helpers (at end to keep things tidy)
# -----------------------
def _connect_via_url():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return None
    if "sslmode=" not in db_url:
        sep = "&" if "?" in db_url else "?"
        db_url = f"{db_url}{sep}sslmode=require"
    return psycopg2.connect(db_url)

def _connect_via_parts():
    host = os.environ.get("PGHOST")
    db   = os.environ.get("PGDATABASE")
    user = os.environ.get("PGUSER")
    pw   = os.environ.get("PGPASSWORD")
    if not all([host, db, user, pw]):
        return None
    port = int(os.environ.get("PGPORT", "5432"))
    return psycopg2.connect(
        host=host, dbname=db, user=user, password=pw,
        port=port, sslmode="require"
    )


def with_conn():
    # Prefer DATABASE_URL (pooler) — always resolves via DNS
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        if "sslmode=" not in db_url:
            sep = "&" if "?" in db_url else "?"
            db_url = f"{db_url}{sep}sslmode=require"
        return psycopg2.connect(db_url)

    # Fallback: individual PG* vars
    host = os.environ.get("PGHOST")
    hostaddr = os.environ.get("PGHOSTADDR")  # IPv6 literal to bypass DNS
    db   = os.environ.get("PGDATABASE")
    user = os.environ.get("PGUSER")
    pw   = os.environ.get("PGPASSWORD")
    port = int(os.environ.get("PGPORT", "5432"))

    if not all([host, db, user, pw]):
        raise RuntimeError("Missing DB config. Set DATABASE_URL or PGHOST/PGDATABASE/PGUSER/PGPASSWORD.")

    return psycopg2.connect(
        host=host,
        hostaddr=hostaddr,   # can be None; safe either way
        dbname=db,
        user=user,
        password=pw,
        port=port,
        sslmode="require",
    )


def ensure_schema():
    with with_conn() as conn, conn.cursor() as cur:
        cur.execute(DDL)
        conn.commit()


if __name__ == "__main__":
    main()
