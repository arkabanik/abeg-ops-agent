from langchain.chat_models import init_chat_model
from langchain_core.caches import InMemoryCache
import os

cache = InMemoryCache(maxsize= 1000)

def get_llm():
    """Main agent model. Must support tool calling."""
    return init_chat_model(os.getenv("LLM_MODEL", "groq:openai/gpt-oss-120b"))
    # llm = ChatGroq(
    #     model="openai/gpt-oss-120b",
    #     # cache = cache
    # )
    # return llm

def get_security_llm():
    """Security classifier model. Must support structured output (with_structured_output)."""
    return init_chat_model(os.getenv("SECURITY_LLM_MODEL", "groq:openai/gpt-oss-20b"), temperature=0, timeout=10, cache=False)
    # llm = ChatGroq(
    #     model="openai/gpt-oss-20b",
    #     temperature=0,
    #     timeout=10,
    #     cache = False
    # )
    # return llm