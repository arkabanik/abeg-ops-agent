import time
import uuid

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.mcp import MCPAdapter
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from cache.semantic_question_cache import QuestionCache
from guardrails.id_redaction_middleware import IdRedactionAgentMiddleware
from guardrails.security_middleware import LLMSecurityMiddleware
from llm.llm import get_llm, get_security_llm
from prompts.system_prompt import SYSTEM_PROMPT
import os


async def run():

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
    question_semantic_cache = QuestionCache(0.2, 600)
    agent = create_agent(
        model = get_llm(),
        tools = mcp_tools,
        middleware = [
            LLMSecurityMiddleware(get_security_llm()),
            PIIMiddleware("email", strategy="redact", apply_to_input=True, apply_to_output=True, apply_to_tool_results=True),
            PIIMiddleware("credit_card",strategy="mask", apply_to_input=True, apply_to_output=True, apply_to_tool_results=True),
            IdRedactionAgentMiddleware()
        ],
        system_prompt = SYSTEM_PROMPT,
        checkpointer= InMemorySaver()
    )

    def new_conversation():
        return 0, {"configurable": {"thread_id":str(uuid.uuid4())}}

    turn, config = new_conversation()
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
            if turn == 0:
                try:
                    cached_answer = await question_semantic_cache.get(user_query)
                except Exception as e:
                    print(f"Cache unavailable, continuing without it: {type(e).__name__}")
                    cached_answer = None

                if cached_answer:
                    time_taken = time.perf_counter() - start_time
                    print(f"Time taken to process the query: {time_taken:.2f} seconds")
                    print(f"Response: {cached_answer}")
                    await agent.aupdate_state(config, {"messages": [HumanMessage(user_query), AIMessage(cached_answer)]})
                    turn += 1
                    continue

            # If no cache hit, run the full agent
            start_time = time.perf_counter()
            response = await agent.ainvoke({"messages": [
                ("human", user_query)
            ]}, config=config)
            time_taken = time.perf_counter() - start_time
            print(f"Time taken to process the query: {time_taken:.2f} seconds")
            result = response["messages"][-1].content
            print(f"Response: {result}")

            tool_used = any(isinstance(m, ToolMessage) for m in response["messages"])

            if turn ==0 and tool_used and isinstance(result, str) and result.strip():
                try:
                    await question_semantic_cache.put(user_query, result)
                except Exception as e:
                    print(f"Could not save to cache: {type(e).__name__}")
            turn = turn + 1
        except Exception as e:
            print(f"Error: {e}")



