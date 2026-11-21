"""The agent as a reusable service: build it once, then call answer() for each question."""

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.mcp import MCPAdapter
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.run_log import log_run
from agent.run_result import build_agent_result, build_cached_result, build_error_result
from cache.semantic_question_cache import QuestionCache
from guardrails.id_redaction_middleware import IdRedactionAgentMiddleware
from guardrails.security_middleware import LLMSecurityMiddleware
from llm.llm import get_llm, get_security_llm
from prompts.system_prompt import SYSTEM_PROMPT
from schemas.agent_result import AgentResult

log = logging.getLogger("agent_service")


class AgentService:
    def __init__(self) -> None:
        self.agent: Any = None
        self.cache: QuestionCache | None = None

    @staticmethod
    def new_conversation() -> str:
        """A new conversation is just a new thread ID."""
        return str(uuid.uuid4())

    async def _turn(self, config: dict) -> tuple[int, int]:
        """(turn, number of saved messages), read from the conversation memory."""
        messages = (await self.agent.aget_state(config)).values.get("messages", [])
        turn = sum(isinstance(m, HumanMessage) for m in messages)
        return turn, len(messages)

    async def answer(self, thread_id: str, question: str) -> AgentResult:
        """One question in, one AgentResult out. Never raises exception; failures and exceptions come back as FAILED results."""
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
                    print(f"Cache unavailable, continuing without it: {type(e).__name__}")
                if cached_answer:
                    result = build_cached_result(
                        output=cached_answer, t_id=thread_id, t=turn, source="cache", start_at=started_at, start=started
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
                t=turn,
                source="agent",
                start_at=started_at,
                start=started,
            )
        except Exception:
            log.exception("Run failed (thread %s, turn %d)", thread_id, turn)
            result = build_error_result(t_id=thread_id, t=turn, start_at=started_at, start=started)
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
                print(f"Could not save to cache: {type(e).__name__}")
        return result

    async def start(self) -> None:
        """Build everything once: MCP tools, cache, agent."""
        async with MCPAdapter(os.getenv("MCP_URL", "http://127.0.0.1:9000/mcp")) as adapter:
            tools = await adapter.list_tools()
        print(f"Loaded {len(tools)} tools from the MCP server.")

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
            checkpointer=InMemorySaver(),
        )
