"""
Frequency Analysis Tool
Counts how many times a word or phrase appears across the transcript archive,
broken down by episode. Answers "how many times has X been said / mentioned?"
Uses direct SQL on the `utterances` table — no vector search needed.
"""

import os
from typing import Optional
import psycopg2
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

TOP_EPISODES_LIMIT = 10
EXAMPLE_QUOTES_LIMIT = 5


# ── Output schema ─────────────────────────────────────────────────────────────

class EpisodeFrequency(BaseModel):
    episode_title: str
    episode_id: int
    count: int


class ExampleQuote(BaseModel):
    episode_title: str
    speaker: str
    text: str
    start_ms: int


class FrequencyResult(BaseModel):
    phrase: str
    total_utterances: int        # number of utterances containing the phrase
    total_episodes: int          # number of distinct episodes
    top_episodes: list[EpisodeFrequency]
    example_quotes: list[ExampleQuote]


# ── Errors ────────────────────────────────────────────────────────────────────

class FrequencyError(Exception):
    pass


# ── Main class ────────────────────────────────────────────────────────────────

class FrequencyAnalyzer:

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
            raise FrequencyError("Missing DB config.")
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
            raise FrequencyError(f"Database query failed: {e}") from e

    # ── Public API ────────────────────────────────────────────────────────────

    def count(self, phrase: str) -> FrequencyResult:
        """
        Count how many utterances contain `phrase` (case-insensitive),
        across how many distinct episodes, and return a per-episode breakdown.
        """
        like = f"%{phrase}%"

        # Total counts
        totals = self._run(
            """
            SELECT COUNT(*) AS utterance_count,
                   COUNT(DISTINCT episode_id) AS episode_count
            FROM utterances
            WHERE text ILIKE %s
            """,
            [like],
        )
        total_utterances = int(totals[0][0]) if totals else 0
        total_episodes   = int(totals[0][1]) if totals else 0

        # Per-episode breakdown (top N)
        ep_rows = self._run(
            """
            SELECT e.file_stem, u.episode_id, COUNT(*) AS cnt
            FROM utterances u
            JOIN episodes e ON u.episode_id = e.id
            WHERE u.text ILIKE %s
            GROUP BY e.file_stem, u.episode_id
            ORDER BY cnt DESC
            LIMIT %s
            """,
            [like, TOP_EPISODES_LIMIT],
        )
        top_episodes = [
            EpisodeFrequency(episode_title=r[0] or "", episode_id=r[1], count=int(r[2]))
            for r in ep_rows
        ]

        # Example quotes
        quote_rows = self._run(
            """
            SELECT e.file_stem, u.speaker, u.text, u.start_ms
            FROM utterances u
            JOIN episodes e ON u.episode_id = e.id
            WHERE u.text ILIKE %s
              AND LENGTH(u.text) > 30
            ORDER BY RANDOM()
            LIMIT %s
            """,
            [like, EXAMPLE_QUOTES_LIMIT],
        )
        example_quotes = [
            ExampleQuote(
                episode_title=r[0] or "",
                speaker=r[1] or "",
                text=r[2] or "",
                start_ms=r[3] or 0,
            )
            for r in quote_rows
        ]

        return FrequencyResult(
            phrase=phrase,
            total_utterances=total_utterances,
            total_episodes=total_episodes,
            top_episodes=top_episodes,
            example_quotes=example_quotes,
        )
