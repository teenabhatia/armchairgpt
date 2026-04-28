"""
Tool 1: Query Planning Tool
Parses a natural language user query into a structured plan that drives
the Evidence Retrieval and Episode Resolution tools downstream.
"""

import json
import os
import time
from typing import Optional
from pydantic import BaseModel, field_validator, model_validator
from mistralai.client import Mistral
from dotenv import load_dotenv

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

INTENT_VALUES = {"QA", "search", "clip_discovery", "mention_lookup", "frequency", "clarify"}
STRATEGY_VALUES = {"semantic", "hybrid", "metadata_first"}
MAX_QUERY_LEN = 500
MIN_QUERY_LEN = 3
DEFAULT_TOP_K = 8
MAX_TOP_K = 20

# ── Output schema (Pydantic) ──────────────────────────────────────────────────

class QueryEntities(BaseModel):
    topic: Optional[str] = None
    persons: list[str] = []
    keywords: list[str] = []


class QueryFilters(BaseModel):
    guest: Optional[str] = None          # specific named guest
    guest_type: Optional[str] = None     # e.g. "doctor", "athlete"
    date_range: Optional[str] = None     # e.g. "2023", "before 2020"
    series: Optional[str] = None         # e.g. "Armchair Expert", "Synced"


class QueryPlan(BaseModel):
    intent: str
    entities: QueryEntities
    filters: QueryFilters
    top_k: int = DEFAULT_TOP_K
    strategy: str = "semantic"
    clarification_needed: Optional[str] = None  # set when intent == "clarify"

    @field_validator("intent")
    @classmethod
    def intent_must_be_valid(cls, v: str) -> str:
        if v not in INTENT_VALUES:
            return "search"
        return v

    @field_validator("strategy")
    @classmethod
    def strategy_must_be_valid(cls, v: str) -> str:
        if v not in STRATEGY_VALUES:
            return "semantic"
        return v

    @field_validator("top_k")
    @classmethod
    def top_k_in_range(cls, v: int) -> int:
        return max(1, min(v, MAX_TOP_K))

    @model_validator(mode="after")
    def require_entity_or_keyword(self) -> "QueryPlan":
        has_content = (
            self.entities.topic
            or self.entities.persons
            or self.entities.keywords
            or self.filters.guest
            or self.filters.guest_type
        )
        if not has_content and self.intent != "clarify":
            self.intent = "clarify"
            self.clarification_needed = (
                "Your query is too broad. Could you specify a topic, guest, or keyword?"
            )
        return self


# ── Errors ────────────────────────────────────────────────────────────────────

class PlannerError(Exception):
    pass


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the Query Planning Tool for ArmchairGPT, an internal knowledge base
for the Armchair Expert podcast with Dax Shepard. Your job is to parse a user's natural language
question into a structured JSON retrieval plan.

The podcast has hundreds of episodes. Each episode has guests, timestamps, and transcript chunks.
Hosts are Dax Shepard and Monica Padman. Guests are the interviewees.

Return ONLY valid JSON matching this exact schema — no markdown, no explanation:

{
  "intent": "<QA | search | clip_discovery | mention_lookup | frequency | clarify>",
  "entities": {
    "topic": "<main topic or null>",
    "persons": ["<person names mentioned, excluding hosts>"],
    "keywords": ["<key terms for retrieval>"]
  },
  "filters": {
    "guest": "<specific guest name or null>",
    "guest_type": "<profession/type like doctor, athlete, or null>",
    "date_range": "<year or range like 2022-2023 or null>",
    "series": "<podcast series name or null>"
  },
  "top_k": <integer 5-15>,
  "strategy": "<semantic | hybrid | metadata_first>",
  "clarification_needed": "<message to user if ambiguous, else null>"
}

Intent guide:
- QA: user wants a specific factual answer ("When did Dax talk about X?")
- search: user wants to explore a topic ("episodes about mental health")
- clip_discovery: user wants shareable/quotable moments ("find a funny clip about marriage")
- mention_lookup: user wants to know who talked about a SPECIFIC NAMED public figure ("who has mentioned Kristen Bell?", "has anyone talked about LeBron James?"). Only use when the target is a real named person — NOT for relational terms like "his father", "her mother", "their kids", "his ex". Those should be QA or search.
- frequency: user wants a count of how often a word/phrase appears ("how many times has Dax said gratitude?", "how often do guests mention therapy?")
- clarify: query is too vague to plan (set clarification_needed)

Strategy guide:
- semantic: topic/theme queries with no specific guest/date
- hybrid: has both semantic content and at least one metadata filter
- metadata_first: query is primarily about a specific guest or date range

Rules:
- If no meaningful entity or keyword can be extracted, use intent=clarify
- Default top_k=8; use higher (12-15) for clip_discovery or broad searches
- Do not include Dax Shepard or Monica Padman in entities.persons (they are always the hosts)
- Always include at least 2 keywords even if they overlap with topic
- For mention_lookup: put the person being mentioned in entities.persons; put the speaker (if specified) in filters.guest
- For frequency: put the word/phrase to count in entities.keywords
"""


# ── Main tool class ───────────────────────────────────────────────────────────

class QueryPlanner:
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
            raise PlannerError("MISTRAL_API_KEY not set. Add it to your .env file.")
        self._client = Mistral(api_key=key)
        self._model_name = "mistral-small-latest"

    # ── Input guardrails ──────────────────────────────────────────────────────

    def _validate_input(self, query: str) -> str:
        if not isinstance(query, str):
            raise PlannerError("Query must be a string.")
        query = query.strip()
        if len(query) < MIN_QUERY_LEN:
            raise PlannerError(
                f"Query too short (min {MIN_QUERY_LEN} characters). Please be more specific."
            )
        if len(query) > MAX_QUERY_LEN:
            raise PlannerError(
                f"Query too long (max {MAX_QUERY_LEN} characters). Please shorten your request."
            )
        # Basic injection guard: reject queries that are pure JSON/code
        stripped = query.replace("{", "").replace("}", "").replace("[", "").replace("]", "").strip()
        if len(stripped) < MIN_QUERY_LEN:
            raise PlannerError("Query appears to be structured data, not a natural language question.")
        return query

    # ── LLM call ─────────────────────────────────────────────────────────────

    def _call_llm(self, query: str, retries: int = 3) -> dict:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f'User query: "{query}"\n\nReturn the JSON plan.'},
        ]
        last_error = None
        for attempt in range(retries):
            try:
                response = self._client.chat.complete(
                    model=self._model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                raw = response.choices[0].message.content.strip()
                try:
                    return json.loads(raw)
                except json.JSONDecodeError as e:
                    raise PlannerError(f"LLM returned invalid JSON: {raw[:200]}") from e
            except PlannerError:
                raise
            except Exception as e:
                last_error = e
                if attempt < retries - 1 and any(
                    code in str(e) for code in ("503", "429", "rate", "unavailable")
                ):
                    time.sleep(2 ** attempt)
                    continue
                break
        raise PlannerError(f"LLM call failed after {retries} attempts: {last_error}") from last_error

    # ── Output validation ─────────────────────────────────────────────────────

    def _validate_output(self, data: dict) -> QueryPlan:
        try:
            entities = QueryEntities(**(data.get("entities") or {}))
            filters = QueryFilters(**(data.get("filters") or {}))
            return QueryPlan(
                intent=data.get("intent") or "search",
                entities=entities,
                filters=filters,
                top_k=data.get("top_k") or DEFAULT_TOP_K,
                strategy=data.get("strategy") or "semantic",
                clarification_needed=data.get("clarification_needed"),
            )
        except Exception as e:
            raise PlannerError(f"Output schema validation failed: {e}") from e

    # ── Public API ────────────────────────────────────────────────────────────

    def plan(self, query: str) -> QueryPlan:
        """
        Parse a natural language query into a structured QueryPlan.

        Raises PlannerError on invalid input or LLM failure.
        """
        clean_query = self._validate_input(query)
        raw = self._call_llm(clean_query)
        return self._validate_output(raw)
