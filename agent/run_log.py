"""One JSON line per run in logs/agent_runs.jsonl (metadata only, never the answer text)."""

import json
import logging
import os

from schemas.agent_result import AgentResult

LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)
_run_logger = logging.getLogger("agent_run_logger")
if not _run_logger.handlers:
    _run_logger.addHandler(logging.FileHandler(filename="logs/agent_runs.jsonl", encoding="utf-8"))
_run_logger.setLevel(logging.INFO)
_run_logger.propagate = False


def log_run(result: AgentResult) -> None:
    agent_meta_json = result.model_dump(mode="json", exclude={"agent_output": {"output"}})
    _run_logger.info(json.dumps(agent_meta_json))
