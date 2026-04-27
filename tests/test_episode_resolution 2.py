from tools.episode_resolution import EpisodeResolver
from tools.evidence_retrieval import EvidenceChunk, RetrievalResult


def test_episode_resolution_merges_adjacent_spans_and_caps_segments():
    r = EpisodeResolver()
    retrieval = RetrievalResult(
        chunks=[
            EvidenceChunk(
                chunk_id=1,
                episode_id=10,
                episode_title="Ep A",
                guests=["Guest"],
                speaker="Dax",
                start_ms=0,
                end_ms=10_000,
                text="first",
                similarity_score=0.9,
                youtube_url=None,
            ),
            # Within MERGE_GAP_MS => should merge into one segment with ellipsis
            EvidenceChunk(
                chunk_id=2,
                episode_id=10,
                episode_title="Ep A",
                guests=["Guest"],
                speaker="Monica",
                start_ms=20_000,
                end_ms=30_000,
                text="second",
                similarity_score=0.8,
                youtube_url=None,
            ),
            # Different episode => separate ResolvedEpisode
            EvidenceChunk(
                chunk_id=3,
                episode_id=11,
                episode_title="Ep B",
                guests=[],
                speaker="Guest",
                start_ms=0,
                end_ms=5_000,
                text="other episode",
                similarity_score=0.95,
                youtube_url=None,
            ),
        ],
        query_text_used="x",
        filters_applied={},
        filters_relaxed=False,
        total_found=3,
    )

    result = r.resolve(retrieval)
    assert result.total_episodes == 2
    assert result.episodes[0].episode_title in {"Ep A", "Ep B"}

    ep_a = next(e for e in result.episodes if e.episode_title == "Ep A")
    assert len(ep_a.segments) == 1
    assert ep_a.segments[0].start_ms == 0
    assert ep_a.segments[0].end_ms == 30_000
    assert "first" in ep_a.segments[0].text
    assert "second" in ep_a.segments[0].text
    assert set(ep_a.segments[0].speakers) == {"Dax", "Monica"}
