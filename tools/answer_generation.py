"""
Answer Generation Layer
Synthesizes a grounded natural-language answer from resolved episode evidence
using Kimi-K2.5 (large-context reasoning via Moonshot AI API).
"""

import os
from typing import Optional
from huggingface_hub import InferenceClient
from pydantic import BaseModel
from dotenv import load_dotenv

from tools.query_planner import QueryPlan
from tools.episode_resolution import ResolutionResult

load_dotenv()

MAX_EPISODES_IN_CONTEXT = 4
MAX_SEGMENTS_PER_EPISODE = 3

_SYSTEM_PROMPT = """You are ArmchairGPT, an internal knowledge assistant for the Armchair Expert podcast with Dax Shepard and Monica Padman.

Answer questions using ONLY the transcript excerpts provided. Rules:
- Ground every factual claim in the provided excerpts — do not add information from your training data
- Cite each claim: include the episode title and timestamp in parentheses, e.g. (Anna Lembke, 0:39)
- If the evidence is insufficient, respond: "I couldn't find enough in the transcripts to answer this."
- For clip discovery queries, describe why each moment is clip-worthy and give the timestamp range
- Be concise — a paragraph or a short bulleted list is usually enough
- Format timestamps as M:SS or H:MM:SS
"""


# ── Output schema ─────────────────────────────────────────────────────────────

class Citation(BaseModel):
    episode_title: str
    episode_id: int
    start_ms: int
    end_ms: int
    quote: str


class GeneratedAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool


# ── Errors ────────────────────────────────────────────────────────────────────

class GenerationError(Exception):
    pass


# ── Main class ────────────────────────────────────────────────────────────────

class AnswerGenerator:

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("HF_TOKEN")
        if not key:
            raise GenerationError("HF_TOKEN not set. Add your Hugging Face token to .env.")
        self._client = InferenceClient(api_key=key)
        self._model = "moonshotai/Kimi-K2.5:novita"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _ms_to_ts(self, ms: int) -> str:
        s = ms // 1000
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    def _format_evidence(self, resolution: ResolutionResult) -> str:
        parts = ["## Transcript Evidence\n"]
        for ep in resolution.episodes[:MAX_EPISODES_IN_CONTEXT]:
            guests_str = ", ".join(ep.guests) if ep.guests else "unknown"
            parts.append(f"### {ep.episode_title}  (guests: {guests_str})")
            for seg in ep.segments[:MAX_SEGMENTS_PER_EPISODE]:
                t0 = self._ms_to_ts(seg.start_ms)
                t1 = self._ms_to_ts(seg.end_ms)
                spk = ", ".join(seg.speakers)
                parts.append(f"[{t0}–{t1}] {spk}: \"{seg.text}\"")
            parts.append("")
        return "\n".join(parts)

    def _build_user_message(self, query: str, plan: QueryPlan, evidence: str) -> str:
        task_hint = {
            "QA": "Answer the question directly with citations.",
            "search": "Summarize the most relevant episodes and what was discussed, with citations.",
            "clip_discovery": "List the best clip moments with timestamps and a one-sentence reason each is clip-worthy.",
        }.get(plan.intent, "Answer based on the evidence.")
        return f"Question: {query}\nTask: {task_hint}\n\n{evidence}"

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, query: str, plan: QueryPlan, resolution: ResolutionResult) -> GeneratedAnswer:
        if not resolution.episodes:
            return GeneratedAnswer(
                answer="I couldn't find any relevant transcript excerpts for your query.",
                citations=[],
                grounded=False,
            )

        evidence = self._format_evidence(resolution)
        user_msg = self._build_user_message(query, plan, evidence)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=800,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as e:
            raise GenerationError(f"Kimi API call failed: {e}") from e

        answer_text = response.choices[0].message.content.strip()

        citations = [
            Citation(
                episode_title=ep.episode_title,
                episode_id=ep.episode_id,
                start_ms=ep.segments[0].start_ms if ep.segments else 0,
                end_ms=ep.segments[0].end_ms if ep.segments else 0,
                quote=ep.segments[0].text[:200] if ep.segments else "",
            )
            for ep in resolution.episodes[:MAX_EPISODES_IN_CONTEXT]
            if ep.segments
        ]

        not_found = {"couldn't find", "no information", "not in the transcripts", "insufficient"}
        grounded = not any(p in answer_text.lower() for p in not_found)

        return GeneratedAnswer(answer=answer_text, citations=citations, grounded=grounded)
