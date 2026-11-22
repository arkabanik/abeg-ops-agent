"""Single-user web API. Run: uv run uvicorn api.app:app --host 127.0.0.1 --port 8000"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent.agent_service import AgenticService, ConversationBusy, ConversationNotFound
from schemas.agent_result import AgentResult

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    load_dotenv()
    service = AgenticService()
    await service.start()  # built ONCE, shared by every request
    app.state.service = service
    yield
    await service.stop()


app = FastAPI(title="ABEG ops assistant", lifespan=lifespan)


# ---------- seams for multi-user later ----------
async def current_user() -> str:
    return "local"  # v2: read the logged-in user from the session / token


def get_service(request: Request) -> AgenticService:
    return request.app.state.service


User = Annotated[str, Depends(current_user)]
Service = Annotated[AgenticService, Depends(get_service)]


# ---------- request / response shapes ----------
class Conversation(BaseModel):
    thread_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Message(BaseModel):
    role: str
    content: str
    turn: int
    run: dict | None = None


# ---------- errors ----------
@app.exception_handler(ConversationNotFound)
async def not_found(request: Request, exc: ConversationNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": "Conversation not found"})


@app.exception_handler(ConversationBusy)
async def busy(request: Request, exc: ConversationBusy) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": "Still answering the previous question"})


# ---------- endpoints ----------
@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/conversations", response_model=Conversation, status_code=201)
async def create_conversation(user: User, service: Service) -> dict:
    return await service.new_conversation(user)


@app.get("/api/conversations", response_model=list[Conversation])
async def list_conversations(user: User, service: Service) -> list[dict]:
    return await service.list_conversations(user)


@app.get("/api/conversations/{thread_id}/messages", response_model=list[Message])
async def get_messages(thread_id: str, user: User, service: Service) -> list[dict]:
    return await service.messages(thread_id, user)


@app.post("/api/conversations/{thread_id}/messages", response_model=AgentResult)
async def ask(thread_id: str, body: Ask, user: User, service: Service) -> AgentResult:
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question is empty")
    return await service.answer(thread_id, question, user)
