import re
from typing import Callable, Awaitable, Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage, AIMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

ID_PATTERN = re.compile(r"\bCUS\d{4,12}\b", re.IGNORECASE)    # real IDs:     CUS00017
PLACEHOLDER_PATTERN = re.compile(r"\bCUSTOMER_\d+\b")         # placeholders: CUSTOMER_1

class IdRedactionAgentMiddleware(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.notebook: dict[str, str] = {}

    def hide(self, text: str) -> str:
        """Replace real IDs with placeholders."""
        def swap(match):
            match_id = match.group(0).upper()
            for placeholder, real_id in self.notebook.items():
                if real_id == match_id:
                    return placeholder

            placeholder = f"CUSTOMER_{len(self.notebook) + 1}"
            self.notebook[placeholder] = match_id
            return placeholder
        return ID_PATTERN.sub(swap, text)

    def reveal(self, text: str) -> str | Any:
        """Replace placeholders with real IDs."""
        def swap(match):
            match_id = match.group(0).upper()
            return self.notebook.get(match_id, match_id)
        return PLACEHOLDER_PATTERN.sub(swap, text)

    @staticmethod
    def on_content(content, fn) -> str | list:
        if isinstance(content, str):
            return fn(content)
        new_list = []
        if isinstance(content, list):
            for b in content:
                if isinstance(b, str):
                    new_list.append(fn(b))
                elif isinstance(b, dict) and "text" in b:
                    new_list.append({**b, "text":fn(b["text"])})
                else:
                    new_list.append(b)
            return new_list
        return content

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:

        messages = [message.model_copy(update ={"content": self.on_content(message.content, self.hide)}) for message in request.messages]

        return await handler(request.override(messages=messages))

    async def awrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]]):

        new_args = {}
        for key, value in request.tool_call["args"].items():
            if isinstance(value,str):
                new_args[key] = self.on_content(value, self.reveal)
            else:
                new_args[key] = value

        new_request = request.override(
            tool_call={**request.tool_call, "args": new_args}
        )


        return await handler(new_request)


    async def aafter_agent(self, state, runtime):

        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage):
            new_message = last_message.model_copy(update = {
                "content": self.on_content(state["messages"][-1].content, self.reveal)
            })
            return {"messages": new_message}
        return None
