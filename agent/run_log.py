"""One JSON line per run in logs/agent_runs.jsonl (metadata only, never the answer text)."""

import json
import logging
import os

from schemas.agent_result import AgentResult

_run_logger = logging.getLogger("agent_run_logger")


def _setup() -> logging.Logger:
    """Attach the log file once, on first use (after .env has been loaded)."""
    if not _run_logger.handlers:
        log_dir = os.getenv("LOG_DIR", "logs")
        os.makedirs(log_dir, exist_ok=True)
        _run_logger.addHandler(
            logging.FileHandler(filename=os.path.join(log_dir, "agent_runs.jsonl"), encoding="utf-8")
        )
        _run_logger.setLevel(logging.INFO)
        _run_logger.propagate = False
    return _run_logger


def log_run(result: AgentResult) -> None:
    agent_meta_json = result.model_dump(mode="json", exclude={"agent_output": {"output"}})
    _setup().info(json.dumps(agent_meta_json))
