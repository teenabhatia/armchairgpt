# ArmchairGPT — System Overview

**Built by:** Teena Bhatia & Dania Hasan

---

## What is ArmchairGPT?

ArmchairGPT is an internal question-answering agent built on top of the full Armchair Expert podcast archive. It lets anyone ask natural-language questions about past episodes and receive grounded, cited answers drawn directly from the transcript database — not from an LLM's training data.

The system covers ~800 episodes of the Armchair Expert podcast with Dax Shepard and Monica Padman, spanning guests from scientists and athletes to actors and politicians. Every episode has been transcribed, speaker-diarized, and stored in a vector database. ArmchairGPT sits on top of that database and makes it queryable in plain English.

**Example queries the system can answer:**
- *"When did Dax talk about addiction with a doctor?"* → Returns the Anna Lembke (psychiatrist) episode with a timestamped citation and YouTube link
- *"Who has mentioned Kristen Bell?"* → Returns every guest who mentioned her, with exact YouTube timestamps
- *"How many times has Dax said grateful?"* → 1,294 times across 570 episodes
- *"Find a clip about therapy and mental health"* → Returns clip-worthy moments with timestamps and reasons

---

## High-Level Architecture

The system has two independent layers: an **ingestion pipeline** that runs offline to build the database, and a **query-time agent** that runs on every user request.

```
┌─────────────────────────────────────────────────────┐
│                  INGESTION PIPELINE                  │
│  Audio → AssemblyAI → Diarization → Chunking →      │
│  Embeddings → Supabase/pgvector                      │
└──────────────────────────┬──────────────────────────┘
                           │  (runs once + weekly)
                           ▼
┌─────────────────────────────────────────────────────┐
│               SUPABASE POSTGRESQL DB                 │
│  episodes · utterances · chunks · mentions_llm       │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│                  QUERY-TIME AGENT                    │
│  FastAPI → Query Planner → Route → Answer → Verify  │
└─────────────────────────────────────────────────────┘
```

---

## Layer 1 — Ingestion Pipeline

Before the agent can answer any question, all episode audio must be processed and stored. This pipeline runs once for historical episodes and weekly for new ones.

### Step 1: Transcription & Diarization
Each episode's audio is sent to **AssemblyAI**, which returns:
- A full transcript with word-level timestamps
- Speaker labels (Speaker A, Speaker B, etc.) via speaker diarization
- Named entity detection (people, places, organisations mentioned)

A post-processing step maps the generic speaker labels to real names (Dax, Monica, or the guest) using LeMUR (AssemblyAI's LLM layer) and the known guest list.

### Step 2: Chunking
The full transcript is split into overlapping ~30-second text chunks. Each chunk stores:
- The speaker and episode it came from
- Start and end timestamps in milliseconds
- The raw text

Overlapping chunks ensure that a sentence spanning a chunk boundary is fully captured in at least one chunk.

### Step 3: Embedding
Each chunk's text is passed through `sentence-transformers/all-MiniLM-L6-v2`, producing a 384-dimensional vector representation of its meaning. These vectors are stored alongside the text in the `chunks` table using **pgvector**, a PostgreSQL extension for vector similarity search.

The same model is used at query time to embed the user's question — this is what makes semantic search work.

### Step 4: Mentions Extraction
A separate LLM pass (`extract_quotes_llm.py`) reads each transcript and extracts structured mention records: who mentioned whom, what they said, and when. These are stored in `mentions_llm`. A further step (`update_youtube_timestamps.py`) uses YouTube SRT caption files to pin each mention to an exact YouTube timestamp.

### Step 5: YouTube URL Matching
Episode YouTube URLs are matched from a spreadsheet to database episodes using fuzzy title matching (`match_youtube_urls.py`). ~720 of 816 episodes have YouTube links.

---

## Layer 2 — The Query-Time Agent

When a user submits a question, it passes through a 4-tool pipeline. The pipeline is fully traced — every step appends a status record that is returned with the response and shown in the UI.

### Tool 1 — Query Planner (Mistral-small-latest)

The planner is the entry point. It takes the raw user query and produces a structured `QueryPlan` with:
- **Intent**: what kind of query this is (see routing section below)
- **Entities**: topic, person names, keywords
- **Filters**: specific guest, date range, podcast series
- **Strategy**: semantic / hybrid / metadata-first
- **top_k**: how many chunks to retrieve

**Guardrails applied here:**
- Query under 3 or over 500 characters → rejected immediately
- Pure JSON/code input → rejected (injection guard)
- No extractable entity or keyword → intent set to `clarify`

### Routing

After planning, the agent routes to one of five paths:

```
mention_lookup  →  SQL on mentions_llm  →  Answer Generation  →  Support Verification
frequency       →  SQL COUNT on utterances  →  return directly (no LLM)
clarify         →  return clarification message immediately
QA / search /
clip_discovery  →  Tool 2  →  Tool 3  →  Answer Generation  →  Tool 4
```

### Tool 2 — Evidence Retrieval (pgvector)

For RAG queries (QA, search, clip_discovery), the planner's search text is embedded using the same `all-MiniLM-L6-v2` model used at ingestion. The resulting vector is compared against all ~200k chunk embeddings using cosine similarity:

```sql
1 - (chunk.embedding <=> query_vector) AS similarity_score
```

Only chunks above a minimum similarity threshold (0.25) are returned. Metadata filters (guest name, date range) are applied as SQL `WHERE` conditions. If filtered results are empty, the system falls back to a pure semantic search with a lower threshold (0.15).

**Additional guardrails:**
- Chunks under 80 characters are excluded (eliminates single-word noise)
- Clip discovery queries use a stricter minimum of 150 characters and a higher top_k (≥20)

### Tool 3 — Episode Resolution (pure Python)

The flat list of retrieved chunks is reorganised into a ranked list of episodes:

1. **Group** chunks by `episode_id`
2. **Merge** adjacent chunks within 30 seconds of each other into continuous segments
3. **Score** each episode: `0.7 × peak_similarity + 0.3 × mean_similarity`
4. **Rank** episodes by score, return top episodes with their merged segments

This step converts "20 scattered chunks from 12 episodes" into "5 episodes ranked by relevance, each with their best contiguous transcript moments."

### Answer Generation (Kimi-K2.5 via HuggingFace)

The top 4 resolved episodes with their transcript segments are formatted and sent to **Kimi-K2.5** (`moonshotai/Kimi-K2.5:novita`), a large-context reasoning model. The system prompt strictly instructs the model to:
- Answer **only** from the provided transcript excerpts
- Cite every claim with episode title and timestamp
- Say "I couldn't find enough in the transcripts" if evidence is insufficient — never fabricate

The model never draws on its training data to answer questions about the podcast.

### Tool 4 — Support Verification (Mistral-small-latest)

After generation, a second Mistral call reads the answer alongside the evidence and checks for **contradictions** — claims in the answer that are inconsistent with the retrieved transcript text.

- If no contradictions: `action = return_answer`
- If a claim is contradicted by the evidence: `action = abstain` — the answer is replaced with a safe message
- The verifier is deliberately calibrated to flag only direct contradictions, not claims that are simply absent from the (truncated) evidence window

---

## The Mentions & Frequency Routes

These routes bypass the vector search entirely and hit the database directly.

**Mentions lookup** queries the `mentions_llm` table — a pre-extracted table of every named person mentioned across all episodes, with the speaker, quote, and exact YouTube timestamp. This answers "who has mentioned X?" in a single SQL query.

**Frequency analysis** runs a `COUNT` over the `utterances` table with a `ILIKE` text match. It returns total occurrences, episode breakdown, and example quotes — no LLM involved at all.

---

## Database Schema

| Table | Contents |
|---|---|
| `episodes` | Episode metadata: title, guests (JSONB), youtube_url, episode_type |
| `utterances` | Individual speaker turns with start/end timestamps and text |
| `chunks` | ~30s text segments with 384-dim vector embeddings for similarity search |
| `mentions_llm` | Structured mention records: speaker, about_person, quote, youtube_start_ms, youtube_end_ms |

Database: **Supabase PostgreSQL** with the `pgvector` extension. ~816 episodes, ~200k chunks.

---

## Tech Stack

| Component | Technology |
|---|---|
| Transcription | AssemblyAI (diarization + entity detection) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector database | Supabase PostgreSQL + pgvector |
| Query Planning | Mistral-small-latest |
| Answer Generation | Kimi-K2.5 (moonshotai/Kimi-K2.5:novita via HuggingFace) |
| Support Verification | Mistral-small-latest |
| Backend | FastAPI (Python) |
| Frontend | Vanilla HTML/CSS/JS (single-page chat UI) |

---

## Guardrails Summary

| Layer | Guardrail | Behaviour |
|---|---|---|
| Input | Min/max query length | Reject with error message |
| Input | Injection guard (pure JSON/code) | Reject with error message |
| Retrieval | Min similarity threshold (0.25) | Filter low-quality chunks |
| Retrieval | Min chunk length (80–150 chars) | Filter noise/single-word fragments |
| Generation | Citation-only system prompt | Model must cite or say "not found" |
| Verification | Contradiction check (Mistral) | Replace answer with abstain if hallucination detected |
| SQL routes | Parameterised queries only | No SQL injection possible |
