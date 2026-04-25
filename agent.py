"""
ArmchairGPT Agent — main entry point.
Currently wires Tool 1 (Query Planning). Tools 2-4 are stubs.
"""

import json
from tools.query_planner import QueryPlanner, QueryPlan, PlannerError


def run(user_query: str) -> dict:
    planner = QueryPlanner()

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

    # ── Tool 2: Evidence Retrieval (stub) ─────────────────────────────────────
    print("[Tool 2] Evidence Retrieval — not yet implemented")

    # ── Tool 3: Episode Resolution (stub) ─────────────────────────────────────
    print("[Tool 3] Episode Resolution — not yet implemented")

    # ── Tool 4: Support Verification (stub) ──────────────────────────────────
    print("[Tool 4] Support Verification — not yet implemented")

    return {"plan": plan.model_dump(), "status": "pipeline incomplete — Tools 2-4 pending"}


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "When did Dax talk about addiction with a doctor?"
    result = run(query)
    print(json.dumps(result, indent=2))
