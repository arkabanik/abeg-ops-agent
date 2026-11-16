from datetime import datetime
from enum import Enum
from typing import TypeVar, Literal, Generic

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"
MAX_LIMIT = 10
T = TypeVar("T")

class ErrorCode(str, Enum):
    NOT_FOUND = "NOT_FOUND"                  # a lookup by ID matched nothing
    INVALID_INPUT = "INVALID_INPUT"          # bad arguments (e.g. malformed ID)
    PERMISSION_DENIED = "PERMISSION_DENIED"  # PostgREST returned 401/403
    UPSTREAM_ERROR = "UPSTREAM_ERROR"        # PostgREST returned another error
    UNAVAILABLE = "UNAVAILABLE"              # could not connect to PostgREST
    TIMEOUT = "TIMEOUT"                      # PostgREST took too long
    INVALID_RESPONSE = "INVALID_RESPONSE"    # data did not match the expected model

# RETRYABLE = {ErrorCode.UNAVAILABLE, ErrorCode.TIMEOUT, ErrorCode.UPSTREAM_ERROR}

class ToolMeta(BaseModel):
    tool: str
    schema_version: str = SCHEMA_VERSION
    fetched_at: datetime
    duration_ms: int
    limit: int
    offset: int
    returned: int
    total: int| None = None
    has_more: bool = False
    limit_capped: bool = False




class ToolError(BaseModel):
    code: ErrorCode
    message: str
    retryable: bool

class ToolResult(BaseModel, Generic[T]):
    meta: ToolMeta
    data: list[T] = Field(default_factory=list)
    error: ToolError | None
    status: Literal["SUCCESS", "FAILED"]






