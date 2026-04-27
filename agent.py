"""
ArmchairGPT Agent — full pipeline with trace output.

Routing:
  mention_lookup  → MentionsLookup → AnswerGenerator → SupportVerifier
  frequency       → FrequencyAnalyzer (no LLM needed)
  QA/search/clip  → EvidenceRetriever → EpisodeResolver → AnswerGenerator → SupportVerifier
  clarify         → return clarification request immediately
"""

import json
from tools.query_planner import QueryPlanner, PlannerError
from tools.evidence_retrieval import EvidenceRetriever, RetrievalError
from tools.episode_resolution import EpisodeResolver, ResolutionResult, ResolutionError
from tools.answer_generation import AnswerGenerator, GenerationError
from tools.support_verification import SupportVerifier, VerificationError
from tools.mentions_lookup import MentionsLookup, MentionsError
from tools.frequency_analysis import FrequencyAnalyzer, FrequencyError


def run(user_query: str) -> dict:
    trace = []

    def t(step: str, status: str, detail: str):
        trace.append({"step": step, "status": status, "detail": detail})

    planner   = QueryPlanner()
    retriever = EvidenceRetriever()
    resolver  = EpisodeResolver()
    mentions  = MentionsLookup()
    freq      = FrequencyAnalyzer()
    _generator = None
    _verifier  = None

    def generator():
        nonlocal _generator
        if _generator is None:
            _generator = AnswerGenerator()
        return _generator

    def verifier():
        nonlocal _verifier
        if _verifier is None:
            _verifier = SupportVerifier()
        return _verifier

    # ── Tool 1: Query Planning ────────────────────────────────────────────────
    try:
        plan = planner.plan(user_query)
    except PlannerError as e:
        t("Query Planning", "error", str(e))
        return {"error": str(e), "action": "clarify", "trace": trace}

    if plan.intent == "clarify":
        t("Query Planning", "warning", "Query too vague — asking for clarification")
        return {"action": "clarify", "message": plan.clarification_needed, "trace": trace}

    t("Query Planning", "ok",
      f"intent={plan.intent} · strategy={plan.strategy} · top_k={plan.top_k}")

    # ── Route: mention_lookup ─────────────────────────────────────────────────
    if plan.intent == "mention_lookup":
        persons = plan.entities.persons
        if not persons:
            t("Mentions Lookup", "warning", "No person specified")
            return {"action": "clarify",
                    "message": "Who would you like to look up mentions of?",
                    "trace": trace}

        person = persons[0]
        speaker = plan.filters.guest
        try:
            result = mentions.lookup_connection(speaker, person) if speaker \
                else mentions.lookup_about(person)
        except MentionsError as e:
            t("Mentions Lookup", "error", str(e))
            return {"error": str(e), "action": "abort", "trace": trace}

        t("Mentions Lookup", "ok" if result.total_found else "warning",
          f"{result.total_found} mention(s) of '{person}' across "
          f"{len({r.episode_id for r in result.records})} episode(s)")

        if not result.records:
            return {"action": "not_found",
                    "message": f"No mentions of '{person}' found in the transcript archive.",
                    "trace": trace}

        from tools.episode_resolution import ResolvedEpisode, EpisodeSegment
        eps_by_id: dict = {}
        for r in result.records:
            if r.episode_id not in eps_by_id:
                eps_by_id[r.episode_id] = ResolvedEpisode(
                    episode_id=r.episode_id, episode_title=r.episode_title,
                    guests=[], relevance_score=1.0, segments=[],
                    youtube_url=r.youtube_url)
            # Use exact YouTube timestamp if available, otherwise fall back to transcript time
            seg_start = r.youtube_start_ms if r.youtube_start_ms is not None else r.start_ms
            seg_end   = r.youtube_end_ms   if r.youtube_end_ms   is not None else r.end_ms
            eps_by_id[r.episode_id].segments.append(EpisodeSegment(
                start_ms=seg_start, end_ms=seg_end,
                text=f"{r.speaker}: \"{r.quote}\"",
                speakers=[r.speaker], peak_similarity=1.0))
        resolution = ResolutionResult(
            episodes=list(eps_by_id.values())[:5],
            total_episodes=len(eps_by_id))

    # ── Route: frequency ──────────────────────────────────────────────────────
    elif plan.intent == "frequency":
        keywords = plan.entities.keywords
        phrase = keywords[0] if keywords else (plan.entities.topic or "")
        if not phrase:
            t("Frequency Analysis", "warning", "No phrase specified")
            return {"action": "clarify",
                    "message": "What word or phrase would you like to count?",
                    "trace": trace}
        try:
            freq_result = freq.count(phrase)
        except FrequencyError as e:
            t("Frequency Analysis", "error", str(e))
            return {"error": str(e), "action": "abort", "trace": trace}

        t("Frequency Analysis", "ok",
          f"\"{phrase}\" in {freq_result.total_utterances} utterances "
          f"across {freq_result.total_episodes} episodes")

        top_eps = "\n".join(
            f"  • {e.episode_title}: {e.count}×"
            for e in freq_result.top_episodes[:5])
        answer_text = (
            f'"{phrase}" appears **{freq_result.total_utterances} times** '
            f'across **{freq_result.total_episodes} episodes**.\n\n'
            f'Top episodes:\n{top_eps}')
        if freq_result.example_quotes:
            q = freq_result.example_quotes[0]
            answer_text += (f'\n\nExample — {q.speaker} in "{q.episode_title}":\n'
                            f'"{q.text[:200]}"')
        return {
            "action": "return_answer",
            "answer": answer_text,
            "citations": [],
            "supported": True,
            "unsupported_claims": [],
            "plan": plan.model_dump(),
            "frequency": freq_result.model_dump(),
            "trace": trace,
        }

    # ── Route: RAG pipeline ───────────────────────────────────────────────────
    else:
        try:
            retrieval = retriever.retrieve(plan)
        except RetrievalError as e:
            t("Evidence Retrieval", "error", str(e))
            return {"error": str(e), "action": "abort", "trace": trace}

        if not retrieval.chunks:
            t("Evidence Retrieval", "warning", "No relevant chunks found")
            return {"action": "not_found",
                    "message": "No relevant transcript spans found for your query.",
                    "trace": trace}

        t("Evidence Retrieval",
          "warning" if retrieval.filters_relaxed else "ok",
          f"{retrieval.total_found} chunks retrieved"
          + (" (filters relaxed — no exact match)" if retrieval.filters_relaxed else ""))

        try:
            resolution = resolver.resolve(retrieval)
        except ResolutionError as e:
            t("Episode Resolution", "error", str(e))
            return {"error": str(e), "action": "abort", "trace": trace}

        top_ep = resolution.episodes[0].episode_title if resolution.episodes else "—"
        t("Episode Resolution", "ok",
          f"{resolution.total_episodes} episode(s) · top: {top_ep}")

    # ── Answer Generation ─────────────────────────────────────────────────────
    try:
        answer = generator().generate(user_query, plan, resolution)
    except GenerationError as e:
        t("Answer Generation", "error", str(e))
        return {"error": str(e), "action": "abort", "trace": trace}

    t("Answer Generation",
      "ok" if answer.grounded else "warning",
      "Grounded answer synthesised" if answer.grounded else "Answer flagged as ungrounded")

    # ── Tool 4: Support Verification ─────────────────────────────────────────
    try:
        verification = verifier().verify(answer, resolution)
    except VerificationError as e:
        t("Support Verification", "error", str(e))
        return {"error": str(e), "action": "abort", "trace": trace}

    t("Support Verification",
      "ok" if verification.supported else "warning",
      f"action={verification.action}"
      + (f" · {len(verification.unsupported_claims)} unsupported claim(s)"
         if verification.unsupported_claims else ""))

    return {
        "action": verification.action,
        "answer": verification.final_answer,
        "supported": verification.supported,
        "unsupported_claims": verification.unsupported_claims,
        "citations": [c.model_dump() for c in answer.citations],
        "plan": plan.model_dump(),
        "resolution_summary": {
            "total_episodes": resolution.total_episodes,
            "top_episode": resolution.episodes[0].episode_title if resolution.episodes else None,
        },
        "trace": trace,
    }


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "When did Dax talk about addiction with a doctor?"
    result = run(query)
    print("\n" + "="*60)
    print(json.dumps(result, indent=2))
