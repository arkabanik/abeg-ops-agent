"""Builds an AgentResult from one agent run."""

import time
import uuid
from datetime import datetime

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from schemas.agent_result import AgentResult, AgentRunMeta, AgentRunOutput, Source, Status, TokenUsageInfo, ToolCallInfo
from schemas.records import Customer, Order, Sensitivity, sensitive_fields

ID_FIELDS: set[str] = set()

for model in (Customer, Order):
    fields = sensitive_fields(model)
    for name, tag in fields.items():
        if tag == Sensitivity.IDENTIFIER.value:
            ID_FIELDS.add(name)


def _tool_payload(message: ToolMessage) -> dict:
    artifact = message.artifact
    if isinstance(artifact, dict):
        content = artifact.get("structured_content")
        if isinstance(content, dict):
            return content
    return {}


def _status(message: BaseMessage | None, tool_calls: list[ToolCallInfo]) -> Status:
    if message is None:
        return "FAILED"
    if message.response_metadata.get("security_block"):
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
                if isinstance(value, str):
                    ids.append(value)
    return ids


def _usage_by_model(usage_meta: dict) -> dict[str, TokenUsageInfo]:
    usage_token_by_model: dict[str, TokenUsageInfo] = {}
    for key, value in usage_meta.items():
        if isinstance(value, dict):
            usage_token_by_model[key] = TokenUsageInfo(
                input_tokens=value.get("input_tokens", 0),
                output_tokens=value.get("output_tokens", 0),
                total_tokens=value.get("total_tokens", 0),
            )
    return usage_token_by_model


def build_cached_result(*, output: str, t_id: str, t: int, source: Source, start_at: datetime, start: float):
    return AgentResult(
        agent_meta=AgentRunMeta(
            run_id=str(uuid.uuid4()),
            thread_id=t_id,
            turn=t,
            source=source,
            started_at=start_at,
            duration_ms=int((time.perf_counter() - start) * 1000),
        ),
        agent_output=AgentRunOutput(
            output=output,
            status="SUCCESS",  # only successful results are stored in cache
        ),
    )


def build_agent_result(
    *,
    output_messages: list[BaseMessage],
    usage_meta: dict,
    t_id: str,
    t: int,
    source: Source,
    start_at: datetime,
    start: float,
):
    tool_list: list[ToolCallInfo] = []
    record_ids: list[str] = []
    # Gather tool call info for each tool call
    for message in output_messages:
        if not isinstance(message, ToolMessage):
            continue

        tool_payload = _tool_payload(message)
        if tool_payload:
            meta = tool_payload.get("meta", {})
            record_ids.extend(_ids(tool_payload.get("data") or []))
            status = tool_payload.get("status", "FAILED")
            tool_list.append(
                ToolCallInfo(
                    tool_id=message.tool_call_id or None,
                    tool_name=meta.get("tool_name", ""),
                    duration_ms=meta.get("duration_ms"),
                    error_code=(tool_payload.get("error") or {}).get("code"),
                    returned=meta.get("returned"),
                    total=meta.get("total"),
                    has_more=meta.get("has_more", False),
                    status=status,
                )
            )

    # Capture the final message to retrieve the output
    last_message: AIMessage | None = None
    for message in reversed(output_messages):
        if isinstance(message, AIMessage):
            last_message = message
            break
    text = last_message.content if last_message is not None and isinstance(last_message.content, str) else ""
    status = _status(message=last_message, tool_calls=tool_list)
    usage_by_model = _usage_by_model(usage_meta)
    return AgentResult(
        agent_meta=AgentRunMeta(
            run_id=str(uuid.uuid4()),
            thread_id=t_id,
            turn=t,
            source=source,
            started_at=start_at,
            duration_ms=int((time.perf_counter() - start) * 1000),
            tools=tool_list,
            usage_by_model=usage_by_model,
            total_token=sum(value.total_tokens for value in usage_by_model.values()),
        ),
        agent_output=AgentRunOutput(
            output=text,
            status=status,
            record_ids=list(dict.fromkeys(record_ids)),
            has_more=any(t.has_more for t in tool_list),
        ),
    )


def build_error_result(*, t_id: str, t: int, start_at: datetime, start: float) -> AgentResult:
    """Used when a run crashes: a FAILED result with a friendly message instead of a Python error."""
    run_id = str(uuid.uuid4())
    return AgentResult(
        agent_meta=AgentRunMeta(
            run_id=run_id,
            thread_id=t_id,
            turn=t,
            source="agent",
            started_at=start_at,
            duration_ms=int((time.perf_counter() - start) * 1000),
        ),
        agent_output=AgentRunOutput(
            output=f"Something went wrong while answering. Please try again. (run {run_id[:8]})",
            status="FAILED",
        ),
    )
