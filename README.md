# ArmchairGPT

An internal RAG agent for the **Armchair Expert** podcast. Ask natural-language questions about past episodes and receive grounded, cited answers drawn directly from the transcript archive.

**Built by:** Teena Bhatia & Dania Hasan

---

## What it does

| Query type | Example | How it's handled |
|---|---|---|
| Factual QA | "When did Dax talk about addiction with a doctor?" | Vector search → episode resolution → Kimi-K2.5 answer |
| Topic search | "Episodes about grief and loss" | Semantic retrieval across all transcripts |
| Clip discovery | "Find a funny clip about marriage" | Top relevant spans returned with timestamps |
| Who mentioned whom | "Who has talked about Kristen Bell?" | Direct SQL on `mentions` table |
| Buzz word count | "How many times has Dax said grateful?" | SQL COUNT on `utterances` table |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

Required variables:
```
MISTRAL_API_KEY=   # mistral.ai — used for Query Planning and Support Verification
HF_TOKEN=          # huggingface.co — used for Kimi-K2.5 answer generation
DATABASE_URL=      # Supabase PostgreSQL connection string (pgvector)
```

### 3. Run the server

```bash
uvicorn api:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser.

### One-command local run (recommended)

If you have a virtualenv at `.venv/` and a populated `.env`, you can run:

```bash
bash scripts/run_local.sh
```

---

## Run the agent from the command line

```bash
python3 agent.py "When did Dax talk about addiction with a doctor?"
python3 agent.py "Who has mentioned Kristen Bell?"
python3 agent.py "How many times has Dax said grateful?"
```

---

## Evaluation framework

This repo checks **intended tasks** and **edge cases, failures, and adversarial inputs** in two layers:

1. **JSONL harness** — `eval/cases.jsonl` is run by `eval/run_eval.py` (default `--mode offline` stubs retrieval / generation / verification so CI does not call the network). Each case asserts response shape and expectations such as allowed `action` / `plan.intent`, required trace steps, and guardrail outcomes for empty, JSON-shaped, oversized, and injection-style strings. Use `--mode live` with a configured `.env` to exercise the real stack; the harness still checks the same expectations, not semantic answer quality.

2. **Pytest** — `tests/test_agent_failure_adversarial.py` asserts specific termination behaviors (`clarify`, `not_found`, `abstain`, `abort`) under monkeypatched tools. These complement the JSONL suite (e.g. ambiguous query → clarify, empty retrieval → not_found, verifier contradiction → abstain, retrieval error → abort, planner guardrails, SQL-like and out-of-domain queries).

Run the full evaluation surface:

```bash
bash scripts/run_eval_framework.sh
```

Or only pytest cases marked for agent evaluation:

```bash
python3 -m pytest -m agent_eval
```

---

## Run tests (Tool 1)

```bash
python -m pytest tests/test_query_planner.py -v
```

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│  Tool 1: Query Planning         │  Mistral-small-latest
│  intent · entities · filters    │
└────────────────┬────────────────┘
                 │
        ┌────────┴────────┐
        │  Route by intent │
        └────────┬────────┘
     ┌───────────┼───────────┐
     │           │           │
  mention_    frequency    QA / search /
  lookup                  clip_discovery
     │           │           │
     ▼           ▼           ▼
 MentionsLookup  FrequencyAnalyzer  ┌──────────────────────┐
 (SQL on         (SQL COUNT on      │ Tool 2: Evidence      │
  mentions)       utterances)       │ Retrieval (pgvector)  │
     │           │           └──────────────────────┘
     │           │                  │
     │           │           ┌──────▼──────────────────┐
     │           │           │ Tool 3: Episode Resolution│
     │           │           │ Group · rank · merge spans│
     │           │           └──────────────────────────┘
     │           │                  │
     └───────────┴──────────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Answer Generation│  Kimi-K2.5 (HF / Novita)
        │ Grounded + cited │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Tool 4: Support  │  Mistral-small-latest
        │ Verification     │  return / abstain / clarify
        └────────┬────────┘
                 │
                 ▼
           Final Answer
```

### Agent components

| Component | Purpose | Guardrails |
|---|---|---|
| Query Planning (Tool 1) | Parse intent, entities, filters | Min/max query length; injection guard; empty query rejected |
| Evidence Retrieval (Tool 2) | Cosine similarity search on transcript chunks | Min similarity threshold; fallback to pure semantic if filters return nothing |
| Episode Resolution (Tool 3) | Group chunks → merge adjacent spans → rank episodes | Only merges same-episode chunks; caps segments per episode |
| Answer Generation | Kimi-K2.5 synthesises grounded answer | System prompt enforces citation-only answers; no hallucination instruction |
| Support Verification (Tool 4) | Mistral checks every claim against evidence | Returns `abstain` if unsupported claims detected |
| Mentions Lookup | SQL on `mentions` table | Parameterised queries; handles no-results gracefully |
| Frequency Analysis | SQL COUNT on `utterances` | Handles empty phrase; returns examples with context |

---

## Data

Transcripts were ingested from the Armchair Expert audio archive using AssemblyAI (speaker diarization + entity detection) and stored in Supabase PostgreSQL with pgvector.

**Tables:**
- `episodes` — episode metadata (title, guests, release date)
- `utterances` — individual speaker turns with timestamps
- `mentions` — guest cross-references (who mentioned whom)
- `chunks` — text segments with 384-dimensional embeddings (`all-MiniLM-L6-v2`)
