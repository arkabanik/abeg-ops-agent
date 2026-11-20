from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

AGENT_SCHEMA_VERSION = "1.0"

# Outcome = Literal["ANSWERED", "NO_RESULTS", "NOT_FOUND", "UNAVAILABLE", "ERROR", "BLOCKED"]
Source = Literal["agent", "cache"]
Status = Literal["SUCCESS", "FAILED"]


class TokenUsageInfo(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ToolCallInfo(BaseModel):
    tool_id: str | None = None
    tool_name: str | None = None
    duration_ms: int | None = None
    error_code: str | None = None
    returned: int | None = None
    total: int | None = None
    has_more: bool | None = None
    status: Status | None = None


class AgentRunMeta(BaseModel):
    run_id: str
    thread_id: str
    schema_version: str = AGENT_SCHEMA_VERSION
    turn: int
    source: Source
    started_at: datetime
    duration_ms: int
    tools: list[ToolCallInfo] = Field(default_factory=list)
    usage_by_model: dict[str, TokenUsageInfo] = Field(default_factory=dict)
    total_token: int = 0


class AgentRunOutput(BaseModel):
    output: str | None
    status: Status
    record_ids: list[str] = Field(default_factory=list)
    has_more: bool = False


class AgentResult(BaseModel):
    agent_meta: AgentRunMeta
    agent_output: AgentRunOutput
