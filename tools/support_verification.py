"""
Tool 4: Support Verification Tool
Checks every factual claim in the generated answer against retrieved evidence.
Uses Mistral (lightweight structured task) to flag unsupported claims and
decide whether to return, abstain, or request clarification.
"""

import json
import os
import time
from typing import Optional
from mistralai.client import Mistral
from pydantic import BaseModel
from dotenv import load_dotenv

from tools.answer_generation import GeneratedAnswer
from tools.episode_resolution import ResolutionResult

load_dotenv()

_SYSTEM_PROMPT = """You are a fact-checking tool for ArmchairGPT.

You receive a generated answer and the transcript evidence it was based on.
Your job is to identify any claims in the answer that are NOT directly supported
by the provided evidence.

Return ONLY valid JSON matching this exact schema — no markdown, no explanation:

{
  "supported": <true if ALL claims are grounded, false otherwise>,
  "unsupported_claims": ["<claim text>", ...],
  "action": "<return_answer | abstain | request_clarification>"
}

Action guide:
- return_answer: all claims are supported OR only minor wording/style issues
- abstain: one or more factual claims cannot be verified from the evidence
- request_clarification: the answer is ambiguous or the query needs more detail

Be strict: if a specific fact (name, date, quote, episode) is stated but not
present in the evidence, list it as unsupported.
"""


# ── Output schema ─────────────────────────────────────────────────────────────

class VerificationResult(BaseModel):
    supported: bool
    unsupported_claims: list[str]
    action: str          # return_answer | abstain | request_clarification
    raw_answer: str      # the original answer (passed through or flagged)
    final_answer: str    # answer to show the user (may be replaced with abstain message)


# ── Errors ────────────────────────────────────────────────────────────────────

class VerificationError(Exception):
    pass


ABSTAIN_MESSAGE = (
    "I found some relevant transcript excerpts but couldn't verify all the "
    "details in my answer. Please check the cited episodes directly."
)

VALID_ACTIONS = {"return_answer", "abstain", "request_clarification"}


# ── Main class ────────────────────────────────────────────────────────────────

class SupportVerifier:

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
            raise VerificationError("MISTRAL_API_KEY not set.")
        self._client = Mistral(api_key=key)
        self._model = "mistral-small-latest"

    def _format_evidence_summary(self, resolution: ResolutionResult) -> str:
        lines = ["## Evidence"]
        for ep in resolution.episodes[:4]:
            lines.append(f"Episode: {ep.episode_title}")
            for seg in ep.segments[:2]:
                lines.append(f'  "{seg.text[:300]}"')
        return "\n".join(lines)

    def _call_mistral(self, answer: GeneratedAnswer, resolution: ResolutionResult) -> dict:
        evidence = self._format_evidence_summary(resolution)
        user_msg = f"## Answer to verify\n{answer.answer}\n\n{evidence}"

        last_error = None
        for attempt in range(3):
            try:
                resp = self._client.chat.complete(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                raw = resp.choices[0].message.content.strip()
                return json.loads(raw)
            except json.JSONDecodeError as e:
                raise VerificationError(f"Verifier returned invalid JSON: {e}") from e
            except Exception as e:
                last_error = e
                if attempt < 2 and any(c in str(e) for c in ("429", "503", "rate")):
                    time.sleep(2 ** attempt)
                    continue
                break
        raise VerificationError(f"Verifier call failed: {last_error}") from last_error

    # ── Public API ────────────────────────────────────────────────────────────

    def verify(self, answer: GeneratedAnswer, resolution: ResolutionResult) -> VerificationResult:
        """
        Verify that the generated answer is grounded in the retrieved evidence.
        Always returns a VerificationResult — never raises on verification failure,
        only on API/network errors.
        """
        if not answer.grounded:
            return VerificationResult(
                supported=False,
                unsupported_claims=["Answer was already flagged as ungrounded at generation."],
                action="abstain",
                raw_answer=answer.answer,
                final_answer=ABSTAIN_MESSAGE,
            )

        data = self._call_mistral(answer, resolution)

        supported = bool(data.get("supported", False))
        unsupported = [str(c) for c in (data.get("unsupported_claims") or [])]
        action = str(data.get("action", "return_answer"))

        # Guardrail: reject invalid action values
        if action not in VALID_ACTIONS:
            action = "abstain" if unsupported else "return_answer"

        final_answer = answer.answer if action == "return_answer" else ABSTAIN_MESSAGE

        return VerificationResult(
            supported=supported,
            unsupported_claims=unsupported,
            action=action,
            raw_answer=answer.answer,
            final_answer=final_answer,
        )
