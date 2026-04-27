import os
import pytest

from tools.evidence_retrieval import EvidenceRetriever
from tools.frequency_analysis import FrequencyAnalyzer
from tools.mentions_lookup import MentionsLookup
from tools.query_planner import QueryEntities, QueryFilters, QueryPlan


def _has_db() -> bool:
    return bool(os.getenv("DATABASE_URL") or (os.getenv("PGHOST") and os.getenv("PGUSER") and os.getenv("PGPASSWORD")))


@pytest.mark.integration
@pytest.mark.skipif(not _has_db(), reason="DB env vars not set")
def test_frequency_sql_smoke():
    freq = FrequencyAnalyzer()
    res = freq.count("grateful")
    assert res.total_utterances >= 0
    assert res.total_episodes >= 0


@pytest.mark.integration
@pytest.mark.skipif(not _has_db(), reason="DB env vars not set")
def test_mentions_sql_smoke():
    m = MentionsLookup()
    res = m.lookup_about("Kristen Bell", limit=3)
    assert res.total_found >= 0


@pytest.mark.integration
@pytest.mark.skipif(not _has_db(), reason="DB env vars not set")
def test_vector_retrieval_smoke():
    # Avoid calling the planner/LLMs; just build a plan directly.
    plan = QueryPlan(
        intent="search",
        entities=QueryEntities(topic="addiction", keywords=["addiction", "recovery"], persons=[]),
        filters=QueryFilters(),
        top_k=5,
        strategy="semantic",
    )
    retriever = EvidenceRetriever()
    # Avoid downloading the embedding model in CI/offline environments.
    # The DB `chunks.embedding` is expected to be 384-dim (all-MiniLM-L6-v2).
    retriever._embed = lambda text: [0.0] * 384  # type: ignore[method-assign]
    result = retriever.retrieve(plan)
    assert result.total_found >= 0
    # If the DB is populated, we expect at least one chunk.
    # Don't hard-fail if empty (still validates schema + query path).
