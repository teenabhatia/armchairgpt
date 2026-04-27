"""
Mentions Lookup Tool
Directly queries the `mentions` table (populated at ingestion) to answer
"who has talked about who" — no vector search needed.
Returns structured results: who spoke, about whom, what they said, and when.
"""

import os
from typing import Optional
import psycopg2
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

DEFAULT_LIMIT = 20


# ── Output schema ─────────────────────────────────────────────────────────────

class MentionRecord(BaseModel):
    episode_title: str
    episode_id: int
    speaker: str
    about_person: str
    quote: str
    start_ms: int
    end_ms: int
    youtube_url: Optional[str] = None
    youtube_start_ms: Optional[int] = None
    youtube_end_ms: Optional[int] = None


class MentionsResult(BaseModel):
    records: list[MentionRecord]
    total_found: int
    person_queried: str
    lookup_type: str       # "about" | "speaker" | "both"


# ── Errors ────────────────────────────────────────────────────────────────────

class MentionsError(Exception):
    pass


# ── Main class ────────────────────────────────────────────────────────────────

class MentionsLookup:

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
            raise MentionsError("Missing DB config.")
        return psycopg2.connect(
            host=host, dbname=db, user=user, password=pw,
            port=port, sslmode="require"
        )

    def _run(self, sql: str, params: list) -> list[tuple]:
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()
        except Exception as e:
            raise MentionsError(f"Database query failed: {e}") from e

    # ── Public API ────────────────────────────────────────────────────────────

    _SELECT = """
        SELECT e.file_stem, m.episode_id, m.speaker, m.about_person,
               m.quote, m.start_ms, m.end_ms,
               e.youtube_url, m.youtube_start_ms, m.youtube_end_ms
        FROM mentions_llm m
        JOIN episodes e ON m.episode_id = e.id
    """

    def lookup_about(self, person: str, limit: int = DEFAULT_LIMIT) -> MentionsResult:
        """Who has talked about `person`, and what did they say?"""
        sql = self._SELECT + """
            WHERE m.about_person ILIKE %s
            ORDER BY e.created_at DESC
            LIMIT %s
        """
        rows = self._run(sql, [f"%{person}%", limit])
        return MentionsResult(
            records=[self._row(r) for r in rows],
            total_found=len(rows),
            person_queried=person,
            lookup_type="about",
        )

    def lookup_by_speaker(self, person: str, limit: int = DEFAULT_LIMIT) -> MentionsResult:
        """Who has `person` talked about across all episodes?"""
        sql = self._SELECT + """
            WHERE m.speaker ILIKE %s
            ORDER BY e.created_at DESC
            LIMIT %s
        """
        rows = self._run(sql, [f"%{person}%", limit])
        return MentionsResult(
            records=[self._row(r) for r in rows],
            total_found=len(rows),
            person_queried=person,
            lookup_type="speaker",
        )

    def lookup_connection(self, person_a: str, person_b: str, limit: int = DEFAULT_LIMIT) -> MentionsResult:
        """Did person_a and person_b ever talk about each other?"""
        sql = self._SELECT + """
            WHERE (m.speaker ILIKE %s AND m.about_person ILIKE %s)
               OR (m.speaker ILIKE %s AND m.about_person ILIKE %s)
            ORDER BY e.created_at DESC
            LIMIT %s
        """
        rows = self._run(sql, [
            f"%{person_a}%", f"%{person_b}%",
            f"%{person_b}%", f"%{person_a}%",
            limit,
        ])
        return MentionsResult(
            records=[self._row(r) for r in rows],
            total_found=len(rows),
            person_queried=f"{person_a} ↔ {person_b}",
            lookup_type="both",
        )

    def _row(self, r: tuple) -> MentionRecord:
        ep_title, ep_id, speaker, about, quote, start_ms, end_ms, yt_url, yt_start, yt_end = r
        return MentionRecord(
            episode_title=ep_title or "",
            episode_id=ep_id,
            speaker=speaker or "",
            about_person=about or "",
            quote=quote or "",
            start_ms=start_ms or 0,
            end_ms=end_ms or 0,
            youtube_url=yt_url or None,
            youtube_start_ms=yt_start,
            youtube_end_ms=yt_end,
        )
