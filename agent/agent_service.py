"""The agent as a reusable service: build it once, then call answer() for each question."""

import asyncio
import logging
import os
import time
import uuid
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.mcp import MCPAdapter
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from agent.run_log import log_run
from agent.run_result import build_agent_result, build_cached_result, build_error_result
from cache.semantic_question_cache import QuestionCache
from guardrails.id_redaction_middleware import IdRedactionAgentMiddleware
from guardrails.security_middleware import LLMSecurityMiddleware
from llm.llm import get_llm, get_security_llm
from prompts.system_prompt import SYSTEM_PROMPT
from schemas.agent_result import AgentResult
from store.agent_store import AgentStore, Pool

log = logging.getLogger("agent_service")


class ConversationNotFound(Exception):
    """No conversation with this ID for this user."""


class ConversationBusy(Exception):
    """The conversation is still answering an earlier question."""


class AgenticService:
    def __init__(self) -> None:
        self.agent: Any = None
        self.cache: QuestionCache | None = None
        self.store: AgentStore | None = None
        self._stack = AsyncExitStack()  # everything to close in stop()
        self._locks: dict[str, asyncio.Lock] = {}  # one lock per conversation

    # ---------- conversations ----------
    def _store(self) -> AgentStore:
        if self.store is None:
            raise RuntimeError("AgenticService.start() was not called")
        return self.store

    async def new_conversation(self, user_id: str) -> dict:
        """Create and save a new conversation (a new thread ID)."""
        return await self._store().create(str(uuid.uuid4()), user_id)

    async def list_conversations(self, user_id: str) -> list[dict]:
        return await self._store().list(user_id)

    async def _require(self, thread_id: str, user_id: str) -> dict:
        """The conversation, if it exists AND belongs to this user. Otherwise ConversationNotFound."""
        conversation = await self._store().get(thread_id, user_id)
        if conversation is None:
            raise ConversationNotFound(thread_id)
        return conversation

    async def messages(self, thread_id: str, user_id: str) -> list[dict]:
        """Questions and final answers only (no tool calls), each answer with its run details."""
        await self._require(thread_id, user_id)
        state = await self.agent.aget_state({"configurable": {"thread_id": thread_id}})
        runs = await self._store().runs_by_turn(thread_id)
        out: list[dict] = []
        turn = -1
        answered: set[int] = set()
        for message in state.values.get("messages", []):
            if isinstance(message, HumanMessage):
                turn += 1
                out.append({"role": "user", "content": message.content, "turn": turn})
            elif isinstance(message, AIMessage) and not message.tool_calls and isinstance(message.content, str):
                answered.add(turn)
                out.append({"role": "assistant", "content": message.content, "turn": turn, "run": runs.get(turn)})
        for t in range(turn + 1):
            if t not in answered and t in runs:
                run_id = runs[t]["agent_meta"]["run_id"]
                note = {
                    "role": "assistant",
                    "content": f"This question failed (run {run_id[:8]}).",
                    "turn": t,
                    "run": runs[t],
                }
                index = next(i for i, m in enumerate(out) if m["role"] == "user" and m["turn"] == t)
                out.insert(index + 1, note)
        return out

    async def answer(self, thread_id: str, question: str, user_id: str) -> AgentResult:
        """One question in, one AgentResult out.

        Raises only ConversationNotFound / ConversationBusy; every other failure comes back as a FAILED result.
        """
        await self._require(thread_id, user_id)
        lock = self._locks.setdefault(thread_id, asyncio.Lock())
        if lock.locked():
            raise ConversationBusy(thread_id)
        async with lock:
            result = await self._answer(thread_id, question, user_id)
        try:
            await self._store().save_run(result)
            await self._store().touch(thread_id, first_question=question if result.agent_meta.turn == 0 else None)
        except Exception:
            log.exception("Could not save run details (thread %s)", thread_id)
        return result

    async def _turn(self, config: dict) -> tuple[int, int]:
        """(turn, number of saved messages), read from the conversation memory."""
        messages = (await self.agent.aget_state(config)).values.get("messages", [])
        turn = sum(isinstance(m, HumanMessage) for m in messages)
        return turn, len(messages)

    async def _answer(self, thread_id: str, question: str, user_id: str) -> AgentResult:
        started_at, started = datetime.now(timezone.utc), time.perf_counter()
        config = {"configurable": {"thread_id": thread_id}}
        turn = 0
        try:
            turn, before = await self._turn(config)

            # Cache only for the first question of a conversation
            if turn == 0 and self.cache is not None:
                cached_answer = None
                try:
                    cached_answer = await self.cache.get(question)
                except Exception as e:
                    log.warning("Cache unavailable, continuing without it: %s", type(e).__name__)
                if cached_answer:
                    result = build_cached_result(
                        output=cached_answer,
                        t_id=thread_id,
                        user_id=user_id,
                        t=turn,
                        source="cache",
                        start_at=started_at,
                        start=started,
                    )
                    await self.agent.aupdate_state(
                        config, {"messages": [HumanMessage(question), AIMessage(cached_answer)]}
                    )
                    log_run(result)
                    return result
            # invoke the agent
            with get_usage_metadata_callback() as usage:
                response = await self.agent.ainvoke({"messages": [HumanMessage(question)]}, config=config)
            result = build_agent_result(
                output_messages=response["messages"][before:],
                usage_meta=usage.usage_metadata,
                t_id=thread_id,
                user_id=user_id,
                t=turn,
                source="agent",
                start_at=started_at,
                start=started,
            )
        except Exception:
            log.exception("Run failed (thread %s, turn %d)", thread_id, turn)
            result = build_error_result(t_id=thread_id, user_id=user_id, t=turn, start_at=started_at, start=started)
            log_run(result)
            return result

        # Log, then cache first-turn successful answers backed by real rows
        log_run(result)
        output = result.agent_output
        found_rows = any((t.returned or 0) > 0 for t in result.agent_meta.tools)
        if turn == 0 and output.status == "SUCCESS" and found_rows and self.cache is not None:
            try:
                await self.cache.put(question, output.output or "")
            except Exception as e:
                log.warning("Could not save to cache: %s", type(e).__name__)
        return result

    async def start(self) -> None:
        """Build everything once: MCP tools, database, saved memory, conversation store, cache, agent."""
        async with MCPAdapter(os.getenv("MCP_URL", "http://127.0.0.1:9000/mcp")) as adapter:
            tools = await adapter.list_tools()
        print(f"Loaded {len(tools)} tools from the MCP server.")

        # One pool of connections to the agent's own database, shared by memory and the store
        db_url = os.getenv("AGENT_DB_URL")
        if not db_url:
            raise RuntimeError("AGENT_DB_URL is not set (see .env.example)")
        pool: Pool = AsyncConnectionPool(
            conninfo=db_url,
            connection_class=AsyncConnection[DictRow],
            max_size=int(os.getenv("AGENT_DB_POOL_SIZE", "5")),
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            open=False,
        )
        await pool.open(wait=True, timeout=10)
        self._stack.push_async_callback(pool.close)

        # Conversation memory (survives restarts). setup() creates its tables once.
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()

        # Conversation list + run details
        self.store = AgentStore(pool)
        await self.store.setup()

        try:
            self.cache = QuestionCache(0.2, 600)
        except Exception as e:  # Redis down at start-up: carry on without the cache
            print(f"Cache disabled: {type(e).__name__}")
            self.cache = None

        self.agent = create_agent(
            model=get_llm(),
            tools=tools,
            middleware=[
                IdRedactionAgentMiddleware(),
                LLMSecurityMiddleware(get_security_llm()),
                PIIMiddleware(
                    "email", strategy="redact", apply_to_input=True, apply_to_output=True, apply_to_tool_results=True
                ),
                PIIMiddleware(
                    "credit_card",
                    strategy="mask",
                    apply_to_input=True,
                    apply_to_output=True,
                    apply_to_tool_results=True,
                ),
            ],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )

    async def stop(self) -> None:
        """Close the database connection pool."""
        await self._stack.aclose()
