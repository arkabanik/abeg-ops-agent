"""Summarise logs/agent_runs.jsonl. Run: uv run python scripts/run_report.py"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    log = Path(os.getenv("LOG_DIR", "logs")) / "agent_runs.jsonl"
    if not log.exists():
        print(f"No logs yet: {log}")
        return

    runs: list[dict] = []
    skipped = 0
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # half-written or not JSON
            skipped += 1
            continue
        if isinstance(record, dict) and "agent_meta" in record and "agent_output" in record:
            runs.append(record)
        else:
            skipped += 1
    if skipped:
        print(f"Skipped {skipped} unreadable line(s)")

    if not runs:
        print("Log is empty")
        return

    agent_runs = [r for r in runs if r["agent_meta"]["source"] == "agent"]
    failed = [r for r in runs if r["agent_output"]["status"] == "FAILED"]
    print(f"Runs: {len(runs)}  ·  agent: {len(agent_runs)}  ·  cache hits: {len(runs) - len(agent_runs)}")
    print(f"Failure rate: {len(failed) / len(runs):.0%}")

    # Does each turn of a conversation cost more? (the slowdown question)
    by_turn: dict[int, list[dict]] = defaultdict(list)
    for r in agent_runs:
        by_turn[r["agent_meta"]["turn"]].append(r)
    print("\nPer turn (agent runs only):")
    for turn, rs in sorted(by_turn.items()):
        tokens = sum(r["agent_meta"]["total_token"] for r in rs) // len(rs)
        ms = sum(r["agent_meta"]["duration_ms"] for r in rs) // len(rs)
        print(f"  turn {turn}: {len(rs):>3} runs · avg {tokens:>7,} tokens · avg {ms:>7,} ms")

    # Tokens per model (for cost)
    per_model: Counter[str] = Counter()
    for r in agent_runs:
        for model, usage in r["agent_meta"]["usage_by_model"].items():
            per_model[model] += usage["total_tokens"]
    print("\nTokens per model:")
    for model, total in per_model.most_common():
        print(f"  {model}: {total:,}")

    # Which tools fail, and how?
    tool_errors: Counter[tuple[str, str]] = Counter()
    for r in runs:
        for t in r["agent_meta"]["tools"]:
            if t["status"] != "SUCCESS":
                tool_errors[(t["tool_name"], t["error_code"] or "?")] += 1
    print("\nTool failures:")
    for (tool, code), count in tool_errors.most_common() or [(("none", ""), 0)]:
        print(f"  {tool} {code}: {count}" if count else "  none")


if __name__ == "__main__":
    main()
