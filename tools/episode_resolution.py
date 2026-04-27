"""
Tool 3: Episode Resolution Tool
Groups retrieved chunks by episode, merges temporally adjacent spans,
and ranks episodes by aggregated relevance score.
"""

from typing import Optional
from pydantic import BaseModel
from tools.evidence_retrieval import EvidenceChunk, RetrievalResult

# ── Constants ─────────────────────────────────────────────────────────────────

MERGE_GAP_MS = 30_000       # merge chunks within 30 s of each other
MAX_SEGMENTS_PER_EPISODE = 5
# Aggregate score: weight peak similarity heavily, mean as tiebreaker
PEAK_WEIGHT = 0.7
MEAN_WEIGHT = 0.3


# ── Output schema ─────────────────────────────────────────────────────────────

class EpisodeSegment(BaseModel):
    start_ms: int
    end_ms: int
    text: str
    speakers: list[str]
    peak_similarity: float
    youtube_timestamp_ms: Optional[int] = None  # accurate YT time (set only for mentions route)


class ResolvedEpisode(BaseModel):
    episode_id: int
    episode_title: str
    guests: list[str]
    relevance_score: float      # aggregated across all chunks from this episode
    segments: list[EpisodeSegment]
    youtube_url: Optional[str] = None


class ResolutionResult(BaseModel):
    episodes: list[ResolvedEpisode]
    total_episodes: int


# ── Errors ────────────────────────────────────────────────────────────────────

class ResolutionError(Exception):
    pass


# ── Main tool class ───────────────────────────────────────────────────────────

class EpisodeResolver:

    def _aggregate_score(self, scores: list[float]) -> float:
        if not scores:
            return 0.0
        peak = max(scores)
        mean = sum(scores) / len(scores)
        return round(PEAK_WEIGHT * peak + MEAN_WEIGHT * mean, 4)

    def _merge_spans(self, chunks: list[EvidenceChunk]) -> list[EpisodeSegment]:
        """
        Sort chunks by start time, then merge any two consecutive chunks
        whose gap is within MERGE_GAP_MS into a single segment.
        """
        sorted_chunks = sorted(chunks, key=lambda c: c.start_ms)
        segments: list[EpisodeSegment] = []
        i = 0
        while i < len(sorted_chunks):
            seg_chunks = [sorted_chunks[i]]
            j = i + 1
            while j < len(sorted_chunks):
                gap = sorted_chunks[j].start_ms - seg_chunks[-1].end_ms
                if gap <= MERGE_GAP_MS:
                    seg_chunks.append(sorted_chunks[j])
                    j += 1
                else:
                    break

            texts = []
            last_end = -1
            for c in seg_chunks:
                # add ellipsis when chunks are not contiguous within the merged span
                if texts and c.start_ms - last_end > 2000:
                    texts.append("...")
                texts.append(c.text.strip())
                last_end = c.end_ms

            speakers = list(dict.fromkeys(
                c.speaker for c in seg_chunks if c.speaker
            ))
            peak_sim = max(c.similarity_score for c in seg_chunks)

            segments.append(EpisodeSegment(
                start_ms=seg_chunks[0].start_ms,
                end_ms=seg_chunks[-1].end_ms,
                text=" ".join(texts),
                speakers=speakers,
                peak_similarity=round(peak_sim, 4),
            ))
            i = j

        # Return highest-scoring segments first, capped per episode
        segments.sort(key=lambda s: -s.peak_similarity)
        return segments[:MAX_SEGMENTS_PER_EPISODE]

    def resolve(self, retrieval: RetrievalResult) -> ResolutionResult:
        """
        Group chunks by episode, merge adjacent spans, and rank episodes
        by aggregated relevance score.

        Raises ResolutionError if input is malformed.
        """
        if not retrieval.chunks:
            return ResolutionResult(episodes=[], total_episodes=0)

        # Group by episode_id
        by_episode: dict[int, list[EvidenceChunk]] = {}
        for chunk in retrieval.chunks:
            by_episode.setdefault(chunk.episode_id, []).append(chunk)

        resolved: list[ResolvedEpisode] = []
        for episode_id, chunks in by_episode.items():
            scores = [c.similarity_score for c in chunks]
            agg_score = self._aggregate_score(scores)
            segments = self._merge_spans(chunks)

            # Use metadata from any chunk (all share the same episode)
            sample = chunks[0]
            resolved.append(ResolvedEpisode(
                episode_id=episode_id,
                episode_title=sample.episode_title,
                guests=sample.guests,
                relevance_score=agg_score,
                segments=segments,
                youtube_url=sample.youtube_url,
            ))

        # Rank episodes by aggregated score
        resolved.sort(key=lambda e: -e.relevance_score)

        return ResolutionResult(
            episodes=resolved,
            total_episodes=len(resolved),
        )
