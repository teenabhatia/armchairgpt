"""
ArmchairGPT Agent — main entry point.
Tools 1-2 implemented. Tools 3-4 and answer generation are stubs.
"""

import json
from tools.query_planner import QueryPlanner, QueryPlan, PlannerError
from tools.evidence_retrieval import EvidenceRetriever, RetrievalResult, RetrievalError
from tools.episode_resolution import EpisodeResolver, ResolutionResult, ResolutionError


def run(user_query: str) -> dict:
    planner = QueryPlanner()
    retriever = EvidenceRetriever()
    resolver = EpisodeResolver()

    # ── Tool 1: Query Planning ────────────────────────────────────────────────
    try:
        plan = planner.plan(user_query)
    except PlannerError as e:
        return {"error": str(e), "action": "clarify"}

    if plan.intent == "clarify":
        return {
            "action": "clarify",
            "message": plan.clarification_needed,
        }

    print(f"[Tool 1] Query plan:\n{json.dumps(plan.model_dump(), indent=2)}\n")

    # ── Tool 2: Evidence Retrieval ────────────────────────────────────────────
    try:
        retrieval = retriever.retrieve(plan)
    except RetrievalError as e:
        return {"error": str(e), "action": "abort"}

    if retrieval.filters_relaxed:
        print("[Tool 2] Metadata filters returned no results — fell back to pure semantic search.")

    if not retrieval.chunks:
        return {
            "action": "not_found",
            "message": "No relevant transcript spans found for your query.",
            "plan": plan.model_dump(),
        }

    print(f"[Tool 2] Retrieved {retrieval.total_found} chunk(s) "
          f"(query: '{retrieval.query_text_used}', filters_relaxed={retrieval.filters_relaxed})")
    for i, c in enumerate(retrieval.chunks[:3], 1):
        print(f"  [{i}] {c.episode_title} | {c.speaker} | "
              f"{c.start_ms//1000}s–{c.end_ms//1000}s | sim={c.similarity_score:.3f}")
        print(f"      {c.text[:120]}...")

    # ── Tool 3: Episode Resolution ────────────────────────────────────────────
    try:
        resolution = resolver.resolve(retrieval)
    except ResolutionError as e:
        return {"error": str(e), "action": "abort"}

    print(f"\n[Tool 3] Resolved {resolution.total_episodes} episode(s):")
    for i, ep in enumerate(resolution.episodes, 1):
        guests_str = ", ".join(ep.guests) if ep.guests else "unknown guests"
        print(f"  [{i}] {ep.episode_title}  (score={ep.relevance_score:.3f}, "
              f"guests={guests_str}, segments={len(ep.segments)})")
        for seg in ep.segments[:2]:
            t0, t1 = seg.start_ms // 1000, seg.end_ms // 1000
            print(f"      {t0}s–{t1}s  {seg.text[:120]}...")

    # ── Answer Generation (stub) ──────────────────────────────────────────────
    print("\n[Generation] Answer synthesis — not yet implemented")

    # ── Tool 4: Support Verification (stub) ──────────────────────────────────
    print("[Tool 4] Support Verification — not yet implemented")

    return {
        "plan": plan.model_dump(),
        "retrieval": {
            "total_found": retrieval.total_found,
            "query_text_used": retrieval.query_text_used,
            "filters_relaxed": retrieval.filters_relaxed,
        },
        "resolution": {
            "total_episodes": resolution.total_episodes,
            "episodes": [e.model_dump() for e in resolution.episodes],
        },
        "status": "pipeline incomplete — answer generation and Tool 4 pending",
    }


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "When did Dax talk about addiction with a doctor?"
    result = run(query)
    print(json.dumps(result, indent=2))
