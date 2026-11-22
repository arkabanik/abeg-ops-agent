"""Terminal chat: reads questions, calls AgentService.answer(), prints the result."""
from agent.agent_service import AgenticService
from schemas.agent_result import AgentResult


def show(result: AgentResult) -> None:
    out, meta = result.agent_output, result.agent_meta
    tools = ", ".join(t.tool_name or "?" for t in meta.tools) or "none"
    print(f"\nResponse: {out.output}")
    print(f"[{out.status} · {meta.source} · {meta.duration_ms} ms · {meta.total_token} tokens · tools: {tools}]")


USER_ID = "local"


async def run() -> None:
    service = AgenticService()
    await service.start()
    try:
        await chat(service)
    finally:
        await service.stop()


async def chat(service: AgenticService) -> None:
    thread_id = (await service.new_conversation(USER_ID))["thread_id"]

    while True:
        user_query = input("Enter your query (type 'new' to start a new conversation or type 'exit' to quit): ").strip()

        if user_query.lower() in ("exit", "quit"):
            return

        if user_query.lower() == "new":
            thread_id = (await service.new_conversation(USER_ID))["thread_id"]
            print("Started a new conversation")
            continue

        if user_query:
            show(await service.answer(thread_id, user_query, USER_ID))