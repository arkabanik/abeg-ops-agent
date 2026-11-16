import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError, TypeAdapter

from schemas.records import select_for
from schemas.tool_result import ToolResult, MAX_LIMIT, ErrorCode, ToolError, ToolMeta

POSTGREST_URL = os.getenv("POSTGREST_URL", "http://localhost:3000")
TIMEOUT_SECONDS = 10

log = logging.getLogger("postgrest_client")

def get_total_from_header(response: httpx.Response) -> int | None:
    print(f"Response Header: {response.headers}")
    match = re.search(r"/(\d+)$", response.headers.get("content-range", ""))
    return int(match.group(1)) if match else None

async def fetch(tool: str, model: type[BaseModel], table: str, filters: dict["str", "str"], lim: int = 5, offset: int = 0, ) -> ToolResult[Any]:
    """
    Fetch rows of model from table, caps the response limit, records the limit_capped to true/false, validates them, wrpas them in ToolResult[model].
    :param: limit
    :param: offset
    :param: filters to add to the postgrest query
    :param: model type[BaseModel]
    :param: name of the tool to call
    :return: ToolResult[model]
    """

    Result = ToolResult[model]
    limit_capped: bool = False

    # Limit the size of the response
    limit =  min(lim, MAX_LIMIT)
    if limit < lim:
        limit_capped = True
    offset = max(0, offset)

    param = {
        "limit":0,
        "offset":0,
        "select": "*",
    }

    params = {**(filters or {}), "select": select_for(model), "limit": limit, "offset": offset}

    fetched_at = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    duration: int = 0
    def get_meta(returned: int = 0, total_records: int | None = None):
        return ToolMeta(
            tool= tool,
            fetched_at= fetched_at,
            duration_ms= duration,
            limit= limit,
            offset= offset,
            returned= returned,
            total= total_records,
            has_more= total_records is not None and offset + returned < total_records,
            limit_capped= limit_capped
        )

    def failed(error_code: ErrorCode, message: str):
        log.error(f"Error occurred: {message}")
        return ToolResult(meta=get_meta(), error=ToolError(code=error_code, message=message, retryable= error_code in ErrorCode ), status="FAILED")

    #Call PostgREST
    try:
        async with httpx.AsyncClient(base_url=POSTGREST_URL, timeout=TIMEOUT_SECONDS) as client:
            response = await client.get(f"/{table}", params=params, headers={"Prefer":"count=exact"})
            # response.raise_for_status()
    except httpx.TimeoutException as e:
        return failed(ErrorCode.TIMEOUT, f"PostgREST did not answer within {TIMEOUT_SECONDS}s")
    except httpx.RequestError as e:
        return failed(ErrorCode.UNAVAILABLE, "Could not connect to PostgREST")

    duration = int(time.perf_counter() - start_time)
    fetched_at = datetime.now(timezone.utc)
    total = get_total_from_header(response)

    status_code = response.status_code
    if status_code in (416):
        return ToolResult(meta= get_meta(returned=0, total_records= get_total_from_header(response)), data=[], error=None, status="SUCCESS")
    if status_code in (401, 403):
        return failed(ErrorCode.PERMISSION_DENIED, "Not allowed to read this data")
    if status_code == 400:
        return failed(ErrorCode.INVALID_INPUT, "PostgREST rejected the request parameters")
    if status_code >= 400:
        return failed(ErrorCode.UPSTREAM_ERROR, f"PostgREST returned HTTP {status_code}")

    # validate every row against the given model
    try:
        rows = TypeAdapter(list[model]).validate_python(response.json())
    except ValidationError as e:
        return failed(ErrorCode.INVALID_RESPONSE, f"Data from PostgREST did not match the {model.__name__} model")

    # Return the result
    return Result(meta= get_meta(returned=len(rows), total_records= total), data=rows, error= None, status = "SUCCESS")
