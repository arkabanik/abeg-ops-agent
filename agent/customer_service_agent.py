"""Terminal chat: reads questions, calls AgentService.answer(), prints the result."""

from agent.service import AgentService
from schemas.agent_result import AgentResult


def show(result: AgentResult) -> None:
    out, meta = result.agent_output, result.agent_meta
    tools = ", ".join(t.tool_name or "?" for t in meta.tools) or "none"
    print(f"\nResponse: {out.output}")
    print(f"[{out.status} · {meta.source} · {meta.duration_ms} ms · {meta.total_token} tokens · tools: {tools}]")


async def run() -> None:
    service = AgentService()
    await service.start()
    thread_id = service.new_conversation()

    while True:
        user_query = input("Enter your query (type 'new' to start a new conversation or type 'exit' to quit): ").strip()

        if user_query.lower() in ("exit", "quit"):
            return

        if user_query.lower() == "new":
            thread_id = service.new_conversation()
            print("Started a new conversation")
            continue

        if user_query:
            show(await service.answer(thread_id, user_query))
