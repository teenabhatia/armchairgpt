"""
Tool 2: Evidence Retrieval Tool
Embeds the query plan into a vector, searches the chunks table via cosine
similarity, applies optional metadata filters, and returns top-k transcript
spans with episode context.
"""

import os
import re
from typing import Optional
import psycopg2
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from tools.query_planner import QueryPlan

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MIN_SIMILARITY = 0.25     # discard chunks below this cosine similarity
FALLBACK_MIN_SIMILARITY = 0.15  # used when metadata filters return nothing

_emb_model: Optional[SentenceTransformer] = None


# ── Output schema ─────────────────────────────────────────────────────────────

class EvidenceChunk(BaseModel):
    chunk_id: int
    episode_id: int
    episode_title: str
    guests: list[str]
    speaker: str
    start_ms: int
    end_ms: int
    text: str
    similarity_score: float
    youtube_url: Optional[str] = None


class RetrievalResult(BaseModel):
    chunks: list[EvidenceChunk]
    query_text_used: str
    filters_applied: dict
    filters_relaxed: bool = False
    total_found: int


# ── Errors ────────────────────────────────────────────────────────────────────

class RetrievalError(Exception):
    pass


# ── Main tool class ───────────────────────────────────────────────────────────

class EvidenceRetriever:

    def _emb_model(self) -> SentenceTransformer:
        global _emb_model
        if _emb_model is None:
            _emb_model = SentenceTransformer(EMB_MODEL_NAME)
        return _emb_model

    def _conn(self):
        db_url = os.environ.get("DATABASE_URL")
        if db_url:
            if "sslmode=" not in db_url:
                sep = "&" if "?" in db_url else "?"
                db_url = f"{db_url}{sep}sslmode=require"
            return psycopg2.connect(db_url)
        host = os.environ.get("PGHOST")
        db   = os.environ.get("PGDATABASE")
        user = os.environ.get("PGUSER")
        pw   = os.environ.get("PGPASSWORD")
        port = int(os.environ.get("PGPORT", "5432"))
        if not all([host, db, user, pw]):
            raise RetrievalError(
                "Missing DB config. Set DATABASE_URL or PGHOST/PGDATABASE/PGUSER/PGPASSWORD."
            )
        return psycopg2.connect(
            host=host, dbname=db, user=user, password=pw,
            port=port, sslmode="require"
        )

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        vecs = self._emb_model().encode([text], normalize_embeddings=True)
        return vecs[0].tolist()

    def _to_vec_literal(self, vec: list[float]) -> str:
        return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

    # ── Build search text from plan ───────────────────────────────────────────

    def _build_search_text(self, plan: QueryPlan) -> str:
        parts = []
        if plan.entities.topic:
            parts.append(plan.entities.topic)
        parts.extend(plan.entities.keywords)
        parts.extend(plan.entities.persons)
        if plan.filters.guest:
            parts.append(plan.filters.guest)
        if plan.filters.guest_type:
            parts.append(plan.filters.guest_type)
        return " ".join(dict.fromkeys(parts)) if parts else "podcast discussion"

    # ── Build SQL WHERE conditions ────────────────────────────────────────────

    def _build_filters(self, plan: QueryPlan) -> tuple[list[str], list]:
        conditions: list[str] = []
        params: list = []

        if plan.filters.guest:
            conditions.append(
                "EXISTS ("
                "  SELECT 1 FROM jsonb_array_elements_text(e.guests) AS g"
                "  WHERE g ILIKE %s"
                ")"
            )
            params.append(f"%{plan.filters.guest}%")

        if plan.filters.series:
            conditions.append("e.file_stem ILIKE %s")
            params.append(f"%{plan.filters.series}%")

        if plan.filters.date_range:
            years = re.findall(r"\d{4}", plan.filters.date_range)
            dr_lower = plan.filters.date_range.lower()
            if len(years) == 2:
                conditions.append(
                    "EXTRACT(YEAR FROM e.created_at) BETWEEN %s AND %s"
                )
                params.extend([int(years[0]), int(years[1])])
            elif len(years) == 1:
                year = int(years[0])
                if "before" in dr_lower:
                    conditions.append("EXTRACT(YEAR FROM e.created_at) < %s")
                    params.append(year)
                elif "after" in dr_lower:
                    conditions.append("EXTRACT(YEAR FROM e.created_at) > %s")
                    params.append(year)
                else:
                    conditions.append("EXTRACT(YEAR FROM e.created_at) = %s")
                    params.append(year)

        return conditions, params

    # ── Execute search ────────────────────────────────────────────────────────

    def _run_query(
        self,
        vec_literal: str,
        conditions: list[str],
        filter_params: list,
        top_k: int,
        min_sim: float,
    ) -> list[tuple]:
        where_sql = ("AND " + " AND ".join(conditions)) if conditions else ""
        sql = f"""
            WITH scored AS (
                SELECT
                    c.id            AS chunk_id,
                    c.episode_id,
                    c.speaker,
                    c.start_ms,
                    c.end_ms,
                    c.text,
                    e.file_stem     AS episode_title,
                    e.guests,
                    1 - (c.embedding <=> %s::vector) AS similarity_score,
                    e.youtube_url
                FROM chunks c
                JOIN episodes e ON c.episode_id = e.id
                WHERE c.text IS NOT NULL
                  AND c.text <> ''
                  {where_sql}
            )
            SELECT * FROM scored
            WHERE similarity_score >= %s
            ORDER BY similarity_score DESC
            LIMIT %s
        """
        params = [vec_literal] + filter_params + [min_sim, top_k]
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    # ── Row → EvidenceChunk ───────────────────────────────────────────────────

    def _row_to_chunk(self, row: tuple) -> EvidenceChunk:
        chunk_id, episode_id, speaker, start_ms, end_ms, text, ep_title, guests_raw, sim, yt_url = row
        guests: list[str] = guests_raw if isinstance(guests_raw, list) else []
        return EvidenceChunk(
            chunk_id=chunk_id,
            episode_id=episode_id,
            episode_title=ep_title or "",
            guests=guests,
            speaker=speaker or "",
            start_ms=start_ms or 0,
            end_ms=end_ms or 0,
            text=text or "",
            similarity_score=round(float(sim), 4),
            youtube_url=yt_url or None,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, plan: QueryPlan) -> RetrievalResult:
        """
        Retrieve top-k transcript chunks matching the query plan.

        Falls back to unfiltered semantic search if metadata filters return
        no results above the similarity threshold.

        Raises RetrievalError on DB/embedding failure.
        """
        search_text = self._build_search_text(plan)

        try:
            vec = self._embed(search_text)
        except Exception as e:
            raise RetrievalError(f"Embedding failed: {e}") from e

        vec_literal = self._to_vec_literal(vec)
        conditions, filter_params = self._build_filters(plan)
        filters_applied = {
            "guest": plan.filters.guest,
            "series": plan.filters.series,
            "date_range": plan.filters.date_range,
        }

        try:
            rows = self._run_query(
                vec_literal, conditions, filter_params, plan.top_k, MIN_SIMILARITY
            )
        except Exception as e:
            raise RetrievalError(f"Database query failed: {e}") from e

        # Guardrail: if filters returned nothing, retry without metadata filters
        filters_relaxed = False
        if not rows and conditions:
            try:
                rows = self._run_query(
                    vec_literal, [], [], plan.top_k, FALLBACK_MIN_SIMILARITY
                )
                filters_relaxed = True
            except Exception as e:
                raise RetrievalError(f"Fallback query failed: {e}") from e

        chunks = [self._row_to_chunk(r) for r in rows]
        return RetrievalResult(
            chunks=chunks,
            query_text_used=search_text,
            filters_applied=filters_applied,
            filters_relaxed=filters_relaxed,
            total_found=len(chunks),
        )
