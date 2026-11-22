"""Conversations list + run details, in the agent's own Postgres database."""

from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from schemas.agent_result import AgentResult

# One statement per string: the pool runs each statement on its own
SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS conversations (
                                                 thread_id  TEXT PRIMARY KEY,
                                                 user_id    TEXT NOT NULL,
                                                 title      TEXT NOT NULL DEFAULT 'New chat',
                                                 created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """,
    "CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations (user_id, updated_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS agent_runs (
                                              run_id     TEXT PRIMARY KEY,
                                              thread_id  TEXT NOT NULL REFERENCES conversations (thread_id) ON DELETE CASCADE,
        user_id    TEXT NOT NULL,
        turn       INTEGER NOT NULL,
        status     TEXT NOT NULL,
        source     TEXT NOT NULL,
        started_at TIMESTAMPTZ NOT NULL,
        result     JSONB NOT NULL  -- AgentResult WITHOUT the answer text
        )
    """,
    "CREATE INDEX IF NOT EXISTS idx_runs_thread ON agent_runs (thread_id, turn)",
]

COLUMNS = "thread_id, title, created_at, updated_at"

# A pool whose connections return rows as dictionaries ({"turn": 0, ...})
Pool = AsyncConnectionPool[AsyncConnection[DictRow]]


class AgentStore:
    def __init__(self, pool: Pool):
        self.pool = pool

    async def setup(self) -> None:
        """Create the tables if they don't exist yet (safe to run on every start)."""
        async with self.pool.connection() as conn:
            for statement in SCHEMA:
                await conn.execute(statement)

    async def create(self, thread_id: str, user_id: str) -> dict:
        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                f"INSERT INTO conversations (thread_id, user_id) VALUES (%s, %s) RETURNING {COLUMNS}",
                (thread_id, user_id),
            )
            row = await cursor.fetchone()
            if row is None:
                raise RuntimeError("INSERT ... RETURNING returned no row")
            return row

    async def get(self, thread_id: str, user_id: str) -> dict | None:
        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                f"SELECT {COLUMNS} FROM conversations WHERE thread_id = %s AND user_id = %s",
                (thread_id, user_id),
            )
            return await cursor.fetchone()

    async def list(self, user_id: str) -> list[dict]:
        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                f"SELECT {COLUMNS} FROM conversations WHERE user_id = %s ORDER BY updated_at DESC",
                (user_id,),
            )
            return await cursor.fetchall()

    async def touch(self, thread_id: str, first_question: str | None = None) -> None:
        """Move the chat to the top of the list; name it after its first question."""
        async with self.pool.connection() as conn:
            if first_question:
                await conn.execute(
                    "UPDATE conversations SET title = %s WHERE thread_id = %s AND title = 'New chat'",
                    (first_question.strip()[:60], thread_id),
                )
            await conn.execute("UPDATE conversations SET updated_at = now() WHERE thread_id = %s", (thread_id,))

    async def save_run(self, result: AgentResult) -> None:
        meta = result.agent_meta
        async with self.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO agent_runs (run_id, thread_id, user_id, turn, status, source, started_at, result) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    meta.run_id,
                    meta.thread_id,
                    meta.user_id,
                    meta.turn,
                    result.agent_output.status,
                    meta.source,
                    meta.started_at,
                    Jsonb(result.model_dump(mode="json", exclude={"agent_output": {"output"}})),
                ),
            )

    async def runs_by_turn(self, thread_id: str) -> dict[int, dict]:
        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                "SELECT turn, result FROM agent_runs WHERE thread_id = %s ORDER BY started_at", (thread_id,)
            )
            return {row["turn"]: row["result"] for row in await cursor.fetchall()}
