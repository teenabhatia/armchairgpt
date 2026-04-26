"""
ArmchairGPT Agent — full pipeline.

Routing:
  mention_lookup  → MentionsLookup → AnswerGenerator → SupportVerifier
  frequency       → FrequencyAnalyzer → AnswerGenerator → SupportVerifier
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
    planner   = QueryPlanner()
    retriever = EvidenceRetriever()
    resolver  = EpisodeResolver()
    mentions  = MentionsLookup()
    freq      = FrequencyAnalyzer()
    # Generator and verifier are lazy — only instantiated when the route needs them
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
        return {"error": str(e), "action": "clarify"}

    if plan.intent == "clarify":
        return {"action": "clarify", "message": plan.clarification_needed}

    print(f"[Tool 1] intent={plan.intent}  strategy={plan.strategy}  top_k={plan.top_k}")
    print(f"         entities={plan.entities.model_dump()}  filters={plan.filters.model_dump()}\n")

    # ── Route: mention_lookup ─────────────────────────────────────────────────
    if plan.intent == "mention_lookup":
        try:
            persons = plan.entities.persons
            if not persons:
                return {"action": "clarify", "message": "Who would you like to look up mentions of?"}

            person = persons[0]
            speaker = plan.filters.guest

            if speaker:
                result = mentions.lookup_connection(speaker, person)
            else:
                result = mentions.lookup_about(person)

            print(f"[Mentions] Found {result.total_found} mention(s) of '{person}'")

            # Synthesise a plain-language answer from mentions
            if not result.records:
                return {
                    "action": "not_found",
                    "message": f"No mentions of '{person}' found in the transcript archive.",
                    "plan": plan.model_dump(),
                }

            # Build a minimal ResolutionResult so AnswerGenerator can work uniformly
            from tools.episode_resolution import ResolvedEpisode, EpisodeSegment
            eps_by_id: dict = {}
            for r in result.records:
                if r.episode_id not in eps_by_id:
                    eps_by_id[r.episode_id] = ResolvedEpisode(
                        episode_id=r.episode_id,
                        episode_title=r.episode_title,
                        guests=[],
                        relevance_score=1.0,
                        segments=[],
                    )
                eps_by_id[r.episode_id].segments.append(EpisodeSegment(
                    start_ms=r.start_ms,
                    end_ms=r.end_ms,
                    text=f"{r.speaker}: \"{r.quote}\"",
                    speakers=[r.speaker],
                    peak_similarity=1.0,
                ))
            resolution = ResolutionResult(
                episodes=list(eps_by_id.values())[:5],
                total_episodes=len(eps_by_id),
            )

        except MentionsError as e:
            return {"error": str(e), "action": "abort"}

    # ── Route: frequency ──────────────────────────────────────────────────────
    elif plan.intent == "frequency":
        keywords = plan.entities.keywords or plan.entities.topic
        phrase = keywords[0] if isinstance(keywords, list) and keywords else (keywords or "")
        if not phrase:
            return {"action": "clarify", "message": "What word or phrase would you like to count?"}

        try:
            freq_result = freq.count(phrase)
        except FrequencyError as e:
            return {"error": str(e), "action": "abort"}

        print(f"[Frequency] '{phrase}' found in {freq_result.total_utterances} utterances "
              f"across {freq_result.total_episodes} episodes")

        # Build summary answer directly (no LLM needed for a count)
        top_eps = "\n".join(
            f"  - {e.episode_title}: {e.count} time(s)"
            for e in freq_result.top_episodes[:5]
        )
        answer_text = (
            f"The phrase \"{phrase}\" appears in {freq_result.total_utterances} utterances "
            f"across {freq_result.total_episodes} episode(s).\n\nTop episodes:\n{top_eps}"
        )
        if freq_result.example_quotes:
            q = freq_result.example_quotes[0]
            answer_text += f'\n\nExample: {q.speaker} in "{q.episode_title}": "{q.text[:200]}"'

        return {
            "action": "return_answer",
            "answer": answer_text,
            "plan": plan.model_dump(),
            "frequency": freq_result.model_dump(),
        }

    # ── Route: RAG pipeline (QA / search / clip_discovery) ───────────────────
    else:
        try:
            retrieval = retriever.retrieve(plan)
        except RetrievalError as e:
            return {"error": str(e), "action": "abort"}

        if retrieval.filters_relaxed:
            print("[Tool 2] Metadata filters returned no results — fell back to semantic search.")

        if not retrieval.chunks:
            return {
                "action": "not_found",
                "message": "No relevant transcript spans found for your query.",
                "plan": plan.model_dump(),
            }

        print(f"[Tool 2] Retrieved {retrieval.total_found} chunk(s) "
              f"(query: '{retrieval.query_text_used}')")

        try:
            resolution = resolver.resolve(retrieval)
        except ResolutionError as e:
            return {"error": str(e), "action": "abort"}

        print(f"[Tool 3] Resolved {resolution.total_episodes} episode(s):")
        for ep in resolution.episodes[:3]:
            print(f"  - {ep.episode_title}  score={ep.relevance_score:.3f}  "
                  f"segments={len(ep.segments)}")

    # ── Answer Generation (shared by all non-frequency routes) ───────────────
    try:
        answer = generator().generate(user_query, plan, resolution)
    except GenerationError as e:
        return {"error": str(e), "action": "abort"}

    print(f"\n[Generation] Answer generated (grounded={answer.grounded})")
    print(f"  {answer.answer[:200]}...")

    # ── Tool 4: Support Verification ─────────────────────────────────────────
    try:
        verification = verifier().verify(answer, resolution)
    except VerificationError as e:
        return {"error": str(e), "action": "abort"}

    print(f"[Tool 4] supported={verification.supported}  "
          f"action={verification.action}  "
          f"unsupported={verification.unsupported_claims}")

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
    }


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "When did Dax talk about addiction with a doctor?"
    result = run(query)
    print("\n" + "="*60)
    print(json.dumps(result, indent=2))
