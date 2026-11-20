import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

from schemas.records import select_for
from schemas.tool_result import MAX_LIMIT, RETRYABLE, ErrorCode, ToolError, ToolMeta, ToolResult

POSTGREST_URL = os.getenv("POSTGREST_URL", "http://localhost:3000")
TIMEOUT_SECONDS = 10

log = logging.getLogger("postgrest_client")


def get_total_from_header(response: httpx.Response) -> int | None:
    match = re.search(r"/(\d+)$", response.headers.get("content-range", ""))
    return int(match.group(1)) if match else None


async def fetch(
    tool: str,
    model: type[BaseModel],
    table: str,
    filters: dict["str", "str"],
    limit: int = 5,
    offset: int = 0,
    not_found_if_empty: bool = False,
) -> ToolResult[Any]:
    """
    Fetch rows of model from table, caps the response limit, records the limit_capped to true/false, validates them, wrpas them in ToolResult[model].
    :param: limit
    :param: offset
    :param: filters to add to the postgrest query
    :param: model type[BaseModel]
    :param: name of the tool to call
    :return: ToolResult[model]
    """

    Result = ToolResult[model]  # type: ignore[valid-type]
    limit_capped: bool = False

    # Limit the size of the response
    limit_capped = limit > MAX_LIMIT
    limit = max(1, min(limit, MAX_LIMIT))
    offset = max(0, offset)

    params: dict[str, str | int] = {**(filters or {}), "select": select_for(model), "limit": limit, "offset": offset}

    fetched_at = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    duration: int = 0

    def get_meta(returned: int = 0, total_records: int | None = None):
        return ToolMeta(
            tool_name=tool,
            fetched_at=fetched_at,
            duration_ms=duration,
            limit=limit,
            offset=offset,
            returned=returned,
            total=total_records,
            has_more=total_records is not None and offset + returned < total_records,
            limit_capped=limit_capped,
        )

    def failed(error_code: ErrorCode, message: str):
        log.error(f"Error occurred: {message}")
        return Result(
            meta=get_meta(),
            error=ToolError(code=error_code, message=message, retryable=error_code in RETRYABLE),
            status="FAILED",
        )

    # Call PostgREST
    try:
        async with httpx.AsyncClient(base_url=POSTGREST_URL, timeout=TIMEOUT_SECONDS) as client:
            response = await client.get(f"/{table}", params=params, headers={"Prefer": "count=exact"})
    except httpx.TimeoutException:
        return failed(ErrorCode.TIMEOUT, f"PostgREST did not answer within {TIMEOUT_SECONDS}s")
    except httpx.RequestError:
        return failed(ErrorCode.UNAVAILABLE, "Could not connect to PostgREST")

    duration = round((time.perf_counter() - start_time) * 1000)
    fetched_at = datetime.now(timezone.utc)
    total = get_total_from_header(response)

    status_code = response.status_code
    if status_code == 416:
        return Result(
            meta=get_meta(returned=0, total_records=get_total_from_header(response)), data=[], status="SUCCESS"
        )
    if status_code in (401, 403):
        return failed(ErrorCode.PERMISSION_DENIED, "Not allowed to read this data")
    if status_code == 400:
        return failed(ErrorCode.INVALID_INPUT, "PostgREST rejected the request parameters")
    if status_code >= 400:
        return failed(ErrorCode.UPSTREAM_ERROR, f"PostgREST returned HTTP {status_code}")

    # validate every row against the given model
    try:
        rows = TypeAdapter(list[model]).validate_python(response.json())  # type: ignore[valid-type]
    except (ValidationError, ValueError):
        return failed(ErrorCode.INVALID_RESPONSE, f"Data from PostgREST did not match the {model.__name__} model")

    if not rows and not_found_if_empty:
        return failed(
            error_code=ErrorCode.NOT_FOUND, message=f"Couldn't find {model.__name__.lower()} with the given parameters"
        )

    # Return the result
    return Result(meta=get_meta(returned=len(rows), total_records=total), data=rows, status="SUCCESS")
