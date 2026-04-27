"""
Agent-level tests for failure modes + adversarial-style inputs.

Uses monkeypatch for deterministic behavior (no live LLM / DB required).
"""

import pytest

pytestmark = pytest.mark.agent_eval

import agent
from tools.query_planner import QueryPlan, QueryEntities, QueryFilters
from tools.evidence_retrieval import EvidenceRetriever, RetrievalResult, EvidenceChunk, RetrievalError
from tools.answer_generation import AnswerGenerator, GeneratedAnswer, Citation
from tools.support_verification import SupportVerifier, VerificationResult


def _plan_qa() -> QueryPlan:
    return QueryPlan(
        intent="QA",
        entities=QueryEntities(topic="test", keywords=["x", "y"], persons=[]),
        filters=QueryFilters(),
        top_k=5,
        strategy="semantic",
    )


def _plan_mention(person: str) -> QueryPlan:
    return QueryPlan(
        intent="mention_lookup",
        entities=QueryEntities(persons=[person], keywords=["ref", "note"]),
        filters=QueryFilters(),
        top_k=8,
        strategy="semantic",
    )


def _patch_planner(monkeypatch, plan: QueryPlan):
    monkeypatch.setattr(agent.QueryPlanner, "plan", lambda self, q: plan)


def test_ambiguous_query_returns_clarify(monkeypatch):
    plan = QueryPlan(
        intent="clarify",
        entities=QueryEntities(topic=None, keywords=[], persons=[]),
        filters=QueryFilters(),
        top_k=8,
        strategy="semantic",
        clarification_needed="Your query is too broad. Could you specify a topic, guest, or keyword?",
    )
    _patch_planner(monkeypatch, plan)
    r = agent.run("Tell me something interesting")
    assert r["action"] == "clarify"
    assert r.get("message")


def test_not_found_when_no_mentions(monkeypatch):
    class _EmptyMentions:
        def lookup_about(self, person: str, limit: int = 20):  # noqa: ARG001
            from tools.mentions_lookup import MentionsResult

            return MentionsResult(records=[], total_found=0, person_queried=person, lookup_type="about")

    _patch_planner(monkeypatch, _plan_mention("ZZZNonexistentPersonZZZ"))
    monkeypatch.setattr(agent, "MentionsLookup", lambda: _EmptyMentions())
    r = agent.run("ignored")
    assert r["action"] == "not_found"


def test_not_found_when_no_chunks(monkeypatch):
    class _NoChunks(EvidenceRetriever):
        def retrieve(self, plan):  # noqa: ARG001
            return RetrievalResult(
                chunks=[],
                query_text_used="x",
                filters_applied={},
                filters_relaxed=False,
                total_found=0,
            )

    _patch_planner(monkeypatch, _plan_qa())
    monkeypatch.setattr(agent, "EvidenceRetriever", _NoChunks)
    r = agent.run("ignored")
    assert r["action"] == "not_found"


def test_abstain_when_verifier_flags_contradiction(monkeypatch):
    class _OneChunk(EvidenceRetriever):
        def retrieve(self, plan):  # noqa: ARG001
            return RetrievalResult(
                chunks=[
                    EvidenceChunk(
                        chunk_id=1,
                        episode_id=1,
                        episode_title="Ep",
                        guests=[],
                        speaker="Dax",
                        start_ms=0,
                        end_ms=1000,
                        text="They discussed running.",
                        similarity_score=0.9,
                        youtube_url=None,
                    )
                ],
                query_text_used="run",
                filters_applied={},
                filters_relaxed=False,
                total_found=1,
            )

    class _Gen(AnswerGenerator):
        def __init__(self, api_key=None):  # noqa: ARG001
            pass

        def generate(self, query, plan, resolution):  # noqa: ARG001
            return GeneratedAnswer(
                answer="They discussed cooking pasta for an hour.",
                citations=[
                    Citation(
                        episode_title="Ep",
                        episode_id=1,
                        start_ms=0,
                        end_ms=1000,
                        quote="They discussed running.",
                        youtube_url=None,
                        youtube_timestamp_ms=None,
                    )
                ],
                grounded=True,
            )

    class _Ver(SupportVerifier):
        def __init__(self, api_key=None):  # noqa: ARG001
            pass

        def verify(self, answer, resolution):  # noqa: ARG001
            return VerificationResult(
                supported=False,
                unsupported_claims=["cooking pasta contradicts running"],
                action="abstain",
                raw_answer=answer.answer,
                final_answer=(
                    "I found some relevant transcript excerpts but couldn't verify all the "
                    "details in my answer. Please check the cited episodes directly."
                ),
            )

    _patch_planner(monkeypatch, _plan_qa())
    monkeypatch.setattr(agent, "EvidenceRetriever", _OneChunk)
    monkeypatch.setattr(agent, "AnswerGenerator", _Gen)
    monkeypatch.setattr(agent, "SupportVerifier", _Ver)
    r = agent.run("ignored")
    assert r["action"] == "abstain"


def test_abort_on_retrieval_error(monkeypatch):
    class _Boom(EvidenceRetriever):
        def retrieve(self, plan):  # noqa: ARG001
            raise RetrievalError("db down")

    _patch_planner(monkeypatch, _plan_qa())
    monkeypatch.setattr(agent, "EvidenceRetriever", _Boom)
    r = agent.run("ignored")
    assert r["action"] == "abort"
    assert "error" in r


def test_planner_guardrails_empty_whitespace_oversized_json(monkeypatch):
    from tools.query_planner import PlannerError

    def boom(self, q: str):  # noqa: ARG001
        raise PlannerError("guardrail")

    monkeypatch.setattr(agent.QueryPlanner, "plan", boom)
    for q in ["", "   ", "a" * 501, "{}"]:
        r = agent.run(q)
        assert r["action"] == "clarify"
        assert "error" in r


def test_sql_like_and_out_of_domain_still_terminate_cleanly(monkeypatch):
    """Adversarial-shaped text should not crash the agent; planner is stubbed to QA + empty retrieval."""
    _patch_planner(monkeypatch, _plan_qa())

    class _NoChunks(EvidenceRetriever):
        def retrieve(self, plan):  # noqa: ARG001
            return RetrievalResult(
                chunks=[],
                query_text_used="x",
                filters_applied={},
                filters_relaxed=False,
                total_found=0,
            )

    monkeypatch.setattr(agent, "EvidenceRetriever", _NoChunks)
    for q in [
        "SELECT * FROM episodes WHERE guest = 'Dax'",
        "What is the capital of France?",
    ]:
        r = agent.run(q)
        assert r["action"] == "not_found"
        assert "trace" in r
