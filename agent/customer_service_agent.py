import time
import uuid
from datetime import datetime, timezone

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.mcp import MCPAdapter
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver

from cache.semantic_question_cache import QuestionCache
from guardrails.id_redaction_middleware import IdRedactionAgentMiddleware
from guardrails.security_middleware import LLMSecurityMiddleware
from llm.llm import get_llm, get_security_llm
from prompts.system_prompt import SYSTEM_PROMPT
import os
from schemas.agent_result import AgentResult, AgentRunMeta, AgentRunOutput, Source, ToolCallInfo, Status, TokenUsageInfo
from schemas.records import Order, Customer, sensitive_fields, Sensitivity


async def run():

    run_id=str(uuid.uuid4())
    # Modern MCP Configuration Dictionary Structure
    # Note: Transport key is no longer needed; it is inferred automatically from 'command'
    print(f"Connecting to local MCP server via http transport... {os.getenv('MCP_URL')}")
    # Load tools dynamically from the running MCP server
    # This automatically manages the async stdio connection lifetime
    # Use MCPAdapter context manager to automatically start/stop the server
    # async with MCPAdapter(Path(__file__).parent.parent / "server.py") as adapter:
    async with MCPAdapter(os.getenv("MCP_URL","http://127.0.0.1:9000/mcp")) as adapter:
        try:
            # Retrieve tools using the modern API method
            mcp_tools = await adapter.list_tools()
            print(f"Successfully loaded {len(mcp_tools)} tools from MCP server.")

            for tool in mcp_tools:
                print(f"{tool.name} : {tool.description}")

        except Exception as e:
            print(f"Failed to extract tools from adapter: {e}")
            return

    # configure_llm_cache()
    def new_conversation():
        return 0, {"configurable": {"thread_id":str(uuid.uuid4())}}

    ID_FIELDS = set()

    for model in (Customer, Order):
        fields = sensitive_fields(model)
        for name, tag in fields.items():
            if tag == Sensitivity.IDENTIFIER.value:
                ID_FIELDS.add(name)

    def _tool_payload(message: ToolMessage) -> dict | None:

        artifact = message.artifact
        if artifact and isinstance(artifact, dict):
            if artifact["structured_content"] and isinstance(artifact["structured_content"], dict):
                return artifact["structured_content"]

        return None

    def _status(message: BaseMessage | None, tool_calls: list[ToolCallInfo]) -> Status:
        if message is None:
            return "FAILED"
        for toolcall in reversed(tool_calls):
            if toolcall.status == "FAILED":
                return "FAILED"
        return "SUCCESS"

    def _ids(records: list[dict]) -> list[str]:
        ids = []
        for record in records:
            for field, value in record.items():
                if field in ID_FIELDS:
                    if isinstance(field, str):
                        ids.append(value)
                    if isinstance(field, list):
                        ids.extend(_ids(v) for v in value)
        return ids

    def _usage_by_model(usage_meta: dict) -> dict[str,TokenUsageInfo]:
        usage_token_by_model: dict[str, TokenUsageInfo] = {}
        for key, value in usage_meta.items():
            if isinstance(value, dict):
                usage_token_by_model[key] = TokenUsageInfo(input_tokens=value.get("input_tokens", 0), output_tokens=value.get("output_tokens", 0), total_tokens=value.get("total_tokens", 0))
        return usage_token_by_model

    def build_cached_result(*, output: str, t_id: str, t: int, source:Source, start_at: datetime, start: float):
        print("inside Build cache result")
        return AgentResult(
            agent_meta=AgentRunMeta(
                run_id=run_id,
                thread_id=t_id,
                turn=t,
                source=source,
                started_at=start_at,
                duration_ms=int((time.perf_counter() - start) * 1000),
            ),
            agent_output=AgentRunOutput(
                output=output,
                status="SUCCESS",       # only successful results are stored in cache
            )
        )

    def build_agent_result(*, output_messages: list[BaseMessage], usage_meta: dict, t_id: str, t: int, source:Source, start_at: datetime, start: float):
        print("inside Build agent result")
        tool_list: list[ToolCallInfo] = []
        data = []
        # Gather tool call info for each tool call
        for message in output_messages:
            print(f"Reading {type(message)}")
            if not isinstance(message, ToolMessage):
                continue

            tool_payload = _tool_payload(message)
            if tool_payload:
                meta = tool_payload.get("meta", {})
                data = tool_payload.get("data", [])
                status = tool_payload.get("status", "FAILED")

                tool_list.append(ToolCallInfo(tool_id=message.tool_call_id,
                                              tool_name=meta.get("tool_name",""),
                                              duration_ms=meta.get("duration_ms", 0),
                                              error_code=meta.get("error_code", ""),
                                              returned=meta.get("returned", ""),
                                              total=meta.get("total", 0),
                                              has_more=meta.get("has_more", False),
                                              status=status,))

        # Capture the final message to retrieve the output
        last_message: AIMessage | None = None
        for message in reversed(output_messages):
            if isinstance(message, AIMessage):
                last_message = message
                break
        text = last_message.content if last_message is not None and isinstance(last_message.content, str) else ""
        status = _status(message=last_message, tool_calls=tool_list)
        record_ids = _ids(data)



        return AgentResult(
            agent_meta=AgentRunMeta(
                run_id=run_id,
                thread_id=t_id,
                turn=t,
                source=source,
                started_at=start_at,
                duration_ms=int((time.perf_counter() - start) * 1000),
                tools=tool_list,
                usage_by_model=_usage_by_model(usage_meta),
                total_token=sum(value.total_tokens for key, value in _usage_by_model(usage_meta).items())
            ),
            agent_output=AgentRunOutput(
                output=text,
                status=status,
                record_ids=record_ids,
                has_more=any(t.has_more for t in tool_list)
            )
        )

    question_semantic_cache = QuestionCache(0.2, 600)
    agent = create_agent(
        model = get_llm(),
        tools = mcp_tools,
        middleware = [
            IdRedactionAgentMiddleware(),
            LLMSecurityMiddleware(get_security_llm()),
            PIIMiddleware("email", strategy="redact", apply_to_input=True, apply_to_output=True, apply_to_tool_results=True),
            PIIMiddleware("credit_card",strategy="mask", apply_to_input=True, apply_to_output=True, apply_to_tool_results=True)
        ],
        system_prompt = SYSTEM_PROMPT,
        checkpointer= InMemorySaver()
    )

    started_at, started = datetime.now(timezone.utc), time.perf_counter()
    turn, config = new_conversation()
    thread_id = config["configurable"]["thread_id"]
    while True:
        user_query = input("Enter your query (type 'new' to start a new conversation or type 'exit' to quit): ").strip()

        if user_query.lower() in ('exit', 'quit'):
            return

        if user_query.lower() == 'new':
            turn, config = new_conversation()
            print("Started a new conversation")
            continue

        try:
            #Look up cache to check if we have answered anything like this before?
            start_time = time.perf_counter()
            cached_answer: str | None = None
            if turn == 0:
                try:
                    cached_answer = await question_semantic_cache.get(user_query)
                except Exception as e:
                    print(f"Cache unavailable, continuing without it: {type(e).__name__}")
                    cached_answer = None

                if cached_answer:
                    agent_run_result = build_cached_result(output=cached_answer, t_id=thread_id, t=turn, source="cache", start_at=started_at, start=started)
                    await agent.aupdate_state(config, {"messages": [HumanMessage(user_query), AIMessage(cached_answer)]})
                    turn += 1
                    continue

            # If no cache hit, run the full agent
            # start_time = time.perf_counter()
            before = len((await agent.aget_state(config=config)).values.get("messages", []))
            with get_usage_metadata_callback() as usage:
                response = await agent.ainvoke({"messages": [
                    ("human", user_query)
                ]}, config=config)

            agent_run_result = build_agent_result(output_messages=response["messages"][before:], usage_meta=usage.usage_metadata, t_id=thread_id, t=turn, source="agent", start_at=started_at, start=started)

            print(f"\n\nAgent Result: {agent_run_result}")
            # time_taken = time.perf_counter() - start_time
            # print(f"Time taken to process the query: {time_taken:.2f} seconds")
            result = response["messages"][-1].content
            print(f"\n\nResponse: {result}")

            tool_used = any(isinstance(m, ToolMessage) for m in response["messages"])

            if turn ==0 and tool_used and isinstance(result, str) and result.strip():
                try:
                    await question_semantic_cache.put(user_query, result)
                except Exception as e:
                    print(f"Could not save to cache: {type(e).__name__}")
            turn = turn + 1
        except Exception as e:
            print(f"Error: {e}")



