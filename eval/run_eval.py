"""
Evaluation harness for ArmchairGPT.

Covers the “evaluation framework” requirement together with pytest (see README):

- Intended tasks: labeled queries in eval/cases.jsonl (QA, search, clips, mentions, frequency).
- Edge / failure / adversarial: same file plus tests/test_agent_failure_adversarial.py
  (clarify, not_found, abstain, abort; oversized input; out-of-domain; etc.).

This module:
- Runs eval/cases.jsonl through agent.run
- Validates response schema (action, trace, …) and per-case expected fields
- Modes: offline (stubbed tools, no network) vs live (real DB + LLMs in .env)

Offline mode does not judge answer grounding; use live runs or SupportVerifier tests for that.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH_DEFAULT = REPO_ROOT / "eval" / "cases.jsonl"


@dataclass
class EvalResult:
    case_id: str
    passed: bool
    failures: list[str]
    response: dict[str, Any]


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cases.append(json.loads(line))
    return cases


def _ensure_response_shape(resp: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if not isinstance(resp, dict):
        return ["response is not a dict"]
    if "action" not in resp:
        failures.append("missing field: action")
    if "trace" not in resp:
        failures.append("missing field: trace")
    else:
        if not isinstance(resp["trace"], list):
            failures.append("trace is not a list")
    # answer is optional for not_found/clarify but usually present
    return failures


def _trace_steps(resp: dict[str, Any]) -> set[str]:
    steps = set()
    for t in (resp.get("trace") or []):
        if isinstance(t, dict) and "step" in t:
            steps.add(str(t["step"]))
    return steps


def _check_expectations(case: dict[str, Any], resp: dict[str, Any]) -> list[str]:
    exp: dict[str, Any] = case.get("expected") or {}
    failures: list[str] = []

    action = resp.get("action")
    if "actions" in exp and action not in set(exp["actions"]):
        failures.append(f"action={action!r} not in expected actions={exp['actions']!r}")

    if exp.get("must_error_or_clarify"):
        # agent.run returns {"error":..., "action": "..."} on some failures
        if action not in {"clarify", "abort"} and "error" not in resp:
            failures.append("expected clarify/abort or error field for invalid input")

    if exp.get("requires_frequency_field") and "frequency" not in resp:
        failures.append("expected frequency field but it was missing")

    # plan is returned by most success paths; but not guaranteed on clarify from planner error
    intents = exp.get("intents")
    if intents:
        plan = resp.get("plan") or {}
        intent = plan.get("intent")
        if intent not in set(intents):
            failures.append(f"plan.intent={intent!r} not in expected intents={intents!r}")

    must_steps = exp.get("must_have_trace_steps") or []
    if must_steps:
        steps = _trace_steps(resp)
        missing = [s for s in must_steps if s not in steps]
        if missing:
            failures.append(f"missing trace steps: {missing!r}")

    return failures


def _offline_planner_stub(query: str) -> dict[str, Any]:
    q = (query or "").strip().lower()
    if len(q) < 3:
        return {
            "intent": "clarify",
            "entities": {"topic": None, "persons": [], "keywords": []},
            "filters": {"guest": None, "guest_type": None, "date_range": None, "series": None},
            "top_k": 8,
            "strategy": "semantic",
            "clarification_needed": "Your query is too broad. Could you specify a topic, guest, or keyword?",
        }
    if q == "{}":
        return {
            "intent": "clarify",
            "entities": {"topic": None, "persons": [], "keywords": []},
            "filters": {"guest": None, "guest_type": None, "date_range": None, "series": None},
            "top_k": 8,
            "strategy": "semantic",
            "clarification_needed": "Please ask in natural language (not JSON).",
        }
    if "how many times" in q or "how often" in q or "count" in q:
        return {
            "intent": "frequency",
            "entities": {"topic": None, "persons": [], "keywords": ["grateful"] if "grateful" in q else ["phrase"]},
            "filters": {"guest": None, "guest_type": None, "date_range": None, "series": None},
            "top_k": 8,
            "strategy": "metadata_first",
            "clarification_needed": None,
        }
    if "who has mentioned" in q or "mentioned kristen bell" in q:
        return {
            "intent": "mention_lookup",
            "entities": {"topic": None, "persons": ["Kristen Bell"] if "kristen bell" in q else [], "keywords": ["mentioned", "reference"]},
            "filters": {"guest": None, "guest_type": None, "date_range": None, "series": None},
            "top_k": 8,
            "strategy": "metadata_first",
            "clarification_needed": None,
        }
    if "find a clip" in q or "clip" in q:
        return {
            "intent": "clip_discovery",
            "entities": {"topic": None, "persons": [], "keywords": ["clip", "moment"]},
            "filters": {"guest": None, "guest_type": None, "date_range": None, "series": None},
            "top_k": 12,
            "strategy": "semantic",
            "clarification_needed": None,
        }
    if q.endswith("?") or q.startswith("when did") or q.startswith("what did"):
        return {
            "intent": "QA",
            "entities": {"topic": None, "persons": [], "keywords": ["topic", "quote"]},
            "filters": {"guest": None, "guest_type": "doctor" if "doctor" in q else None, "date_range": None, "series": None},
            "top_k": 8,
            "strategy": "hybrid" if "doctor" in q else "semantic",
            "clarification_needed": None,
        }
    return {
        "intent": "search",
        "entities": {"topic": None, "persons": [], "keywords": ["episodes", "topic"]},
        "filters": {"guest": None, "guest_type": None, "date_range": None, "series": None},
        "top_k": 8,
        "strategy": "semantic",
        "clarification_needed": None,
    }


def _install_offline_stubs() -> None:
    # Import locally so we can patch after modules are loaded.
    from tools.query_planner import QueryPlanner
    from tools.answer_generation import AnswerGenerator, GeneratedAnswer, Citation
    from tools.support_verification import SupportVerifier, VerificationResult
    from tools.evidence_retrieval import EvidenceRetriever, EvidenceChunk, RetrievalResult
    from tools.mentions_lookup import MentionsLookup, MentionsResult, MentionRecord
    from tools.frequency_analysis import FrequencyAnalyzer, FrequencyResult, EpisodeFrequency, ExampleQuote

    def _qp_call_llm(self: QueryPlanner, query: str, retries: int = 3) -> dict:  # noqa: ARG001
        return _offline_planner_stub(query)

    def _retrieve(self: EvidenceRetriever, plan):  # noqa: ANN001
        # Deterministic retrieval result with a single mock chunk.
        chunk = EvidenceChunk(
            chunk_id=1,
            episode_id=1,
            episode_title="Offline Episode",
            guests=["Offline Guest"],
            speaker="Dax",
            start_ms=0,
            end_ms=15_000,
            text="(OFFLINE EVAL) Mock transcript excerpt relevant to the query.",
            similarity_score=0.9,
            youtube_url=None,
        )
        return RetrievalResult(
            chunks=[chunk],
            query_text_used="offline",
            filters_applied=getattr(plan, "filters", {}).model_dump() if hasattr(getattr(plan, "filters", None), "model_dump") else {},
            filters_relaxed=False,
            total_found=1,
        )

    def _mentions_about(self: MentionsLookup, person: str, limit: int = 20):  # noqa: ARG001
        rec = MentionRecord(
            episode_title="Offline Episode",
            episode_id=1,
            speaker="Monica",
            about_person=person,
            quote="(OFFLINE EVAL) Monica mentions the person.",
            start_ms=5_000,
            end_ms=8_000,
            youtube_url=None,
            youtube_start_ms=None,
            youtube_end_ms=None,
        )
        return MentionsResult(records=[rec], total_found=1, person_queried=person, lookup_type="about")

    def _mentions_conn(self: MentionsLookup, person_a: str, person_b: str, limit: int = 20):  # noqa: ARG001
        return MentionsResult(records=[], total_found=0, person_queried=f"{person_a} ↔ {person_b}", lookup_type="both")

    def _freq_count(self: FrequencyAnalyzer, phrase: str):
        return FrequencyResult(
            phrase=phrase,
            total_utterances=123,
            total_episodes=45,
            top_episodes=[EpisodeFrequency(episode_title="Offline Episode", episode_id=1, count=10)],
            example_quotes=[ExampleQuote(episode_title="Offline Episode", speaker="Dax", text=f"(OFFLINE EVAL) {phrase}", start_ms=0)],
        )

    def _gen(self: AnswerGenerator, query: str, plan, resolution):  # noqa: ANN001
        # Deterministic “answer”: echo top evidence titles if present.
        if not getattr(resolution, "episodes", None):
            return GeneratedAnswer(
                answer="I couldn't find any relevant transcript excerpts for your query.",
                citations=[],
                grounded=False,
            )
        top = resolution.episodes[0]
        ts = "0:00"
        if top.segments:
            ts = "0:00"  # keep stable offline
        cits = [
            Citation(
                episode_title=top.episode_title,
                episode_id=top.episode_id,
                start_ms=top.segments[0].start_ms if top.segments else 0,
                end_ms=top.segments[0].end_ms if top.segments else 0,
                quote=(top.segments[0].text[:200] if top.segments else ""),
                youtube_url=top.youtube_url,
                youtube_timestamp_ms=None,
            )
        ] if top else []
        return GeneratedAnswer(
            answer=f"(OFFLINE EVAL) Found relevant evidence in “{top.episode_title}” ({ts}).",
            citations=cits,
            grounded=True,
        )

    def _verify(self: SupportVerifier, answer, resolution):  # noqa: ANN001
        return VerificationResult(
            supported=True,
            unsupported_claims=[],
            action="return_answer",
            raw_answer=answer.answer,
            final_answer=answer.answer,
        )

    QueryPlanner._call_llm = _qp_call_llm  # type: ignore[method-assign]
    EvidenceRetriever.retrieve = _retrieve  # type: ignore[method-assign]
    MentionsLookup.lookup_about = _mentions_about  # type: ignore[method-assign]
    MentionsLookup.lookup_connection = _mentions_conn  # type: ignore[method-assign]
    FrequencyAnalyzer.count = _freq_count  # type: ignore[method-assign]
    AnswerGenerator.generate = _gen  # type: ignore[method-assign]
    SupportVerifier.verify = _verify  # type: ignore[method-assign]


def _run_case(run_fn: Callable[[str], dict[str, Any]], case: dict[str, Any]) -> EvalResult:
    case_id = str(case.get("id", "unknown"))
    query = str(case.get("query", ""))
    resp = run_fn(query)
    failures = []
    failures.extend(_ensure_response_shape(resp))
    failures.extend(_check_expectations(case, resp))
    return EvalResult(case_id=case_id, passed=(len(failures) == 0), failures=failures, response=resp)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=str, default=str(CASES_PATH_DEFAULT))
    ap.add_argument("--mode", choices=["offline", "live"], default="offline")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of cases (0 = all)")
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON summary")
    args = ap.parse_args(argv)

    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"Cases file not found: {cases_path}", file=sys.stderr)
        return 2

    if args.mode == "offline":
        _install_offline_stubs()

    # Import after optional stubbing.
    import agent  # noqa: WPS433

    cases = _load_cases(cases_path)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    results: list[EvalResult] = []
    for c in cases:
        results.append(_run_case(agent.run, c))

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    summary = {
        "mode": args.mode,
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "failures": [
            {"id": r.case_id, "failures": r.failures, "action": r.response.get("action"), "intent": (r.response.get("plan") or {}).get("intent")}
            for r in results
            if not r.passed
        ],
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Eval mode={args.mode}  passed={passed}/{len(results)}  failed={failed}")
        for r in results:
            if not r.passed:
                print(f"- FAIL {r.case_id}: " + "; ".join(r.failures))

    # Optional: in live mode, warn if env is missing likely-required keys
    if args.mode == "live":
        missing = [k for k in ("DATABASE_URL", "MISTRAL_API_KEY", "HF_TOKEN") if not os.getenv(k)]
        if missing:
            print(f"Note: missing env vars for full live pipeline: {missing!r}", file=sys.stderr)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

