from typing import Callable, Awaitable, Any
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, ExtendedModelResponse
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from security.security_classifier import LLMSecurityClassifier


class LLMSecurityMiddleware(AgentMiddleware):

    def __init__(self,security_model):
        super().__init__()
        self.classifier = LLMSecurityClassifier(security_model)

    @staticmethod
    def extract_text(message: BaseMessage) -> str:
        # print(f"Human message: {message}")

        content = message.content
        # print(f"Extracted content: {content}")

        if isinstance(content, str):
            return content

        return ""

    def extract_latest_human_message(self, messages: list[BaseMessage]) -> str :

        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return self.extract_text(message)

        return ""


    async def awrap_model_call(self, request: ModelRequest[Any], handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]]):

        user_message = self.extract_latest_human_message(request.messages)

        if not user_message.strip():
            return await handler(request)

        try:

            security_assessment = await self.classifier.classify(user_message)

            # read the assessment and pass the request accordingly
            if security_assessment.prompt_injection:
                return AIMessage(
                    content="Smart ass. I cannot process instructions that attempts to override system instructions or bypass my security policies",
                    response_metadata = {"security_block": "prompt_injection"}
                )

            if security_assessment.harmful_content:
                if security_assessment.action == "block":
                    return AIMessage(
                        content="I cannot process this request as it involves potentially unauthorised or harmful request",
                        response_metadata = {"security_block": "harmful_content"}
                    )

                if security_assessment.action == "review":
                    return AIMessage(
                        content="Need additional human review before proceeding with this request",
                        response_metadata = {"security_block": "harmful_review"}
                    )

            if security_assessment.out_of_scope:
                if security_assessment.action == "block":
                    return AIMessage(
                        content="I cannot process this request as it involves potentially unauthorised or harmful request",
                        response_metadata = {"security_block": "out_of_scope"}
                    )

            # Pass legitimate requests
            return await handler(request)

        except Exception as e:
            print(f"Exception is : {e}")











