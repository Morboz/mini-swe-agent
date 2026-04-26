#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class TaskStats:
    instance_id: str
    traj_path: str
    exit_status: str
    api_calls: int
    assistant_messages: int
    assistant_messages_with_usage: int
    tool_messages: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    instance_cost: float


def _safe_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def find_traj_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.traj.json"))


def load_task_stats(path: Path) -> TaskStats:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Unsupported trajectory format in {path}")

    messages = data.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError(f"Invalid messages field in {path}")

    info = data.get("info", {}) if isinstance(data.get("info"), dict) else {}
    model_stats = info.get("model_stats", {}) if isinstance(info.get("model_stats"), dict) else {}

    prompt_sum = 0
    completion_sum = 0
    total_sum = 0
    assistant_messages = 0
    tool_messages = 0
    assistant_messages_with_usage = 0

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            assistant_messages += 1
        elif role == "tool":
            tool_messages += 1

        extra = message.get("extra", {})
        usage = extra.get("usage", {}) if isinstance(extra, dict) else {}
        if role == "assistant" and isinstance(usage, dict):
            assistant_messages_with_usage += 1
            prompt_sum += _safe_int(usage.get("prompt_tokens"))
            completion_sum += _safe_int(usage.get("completion_tokens"))
            total_sum += _safe_int(usage.get("total_tokens"))

    instance_id = str(data.get("instance_id") or path.stem.removesuffix(".traj"))
    return TaskStats(
        instance_id=instance_id,
        traj_path=str(path),
        exit_status=str(info.get("exit_status", "")),
        api_calls=_safe_int(model_stats.get("api_calls")) or assistant_messages_with_usage,
        assistant_messages=assistant_messages,
        assistant_messages_with_usage=assistant_messages_with_usage,
        tool_messages=tool_messages,
        prompt_tokens=_safe_int(model_stats.get("prompt_tokens")) or prompt_sum,
        completion_tokens=_safe_int(model_stats.get("completion_tokens")) or completion_sum,
        total_tokens=_safe_int(model_stats.get("total_tokens")) or total_sum,
        instance_cost=_safe_float(model_stats.get("instance_cost")),
    )


def build_summary(rows: list[TaskStats]) -> dict[str, Any]:
    return {
        "tasks": len(rows),
        "api_calls": sum(row.api_calls for row in rows),
        "assistant_messages": sum(row.assistant_messages for row in rows),
        "assistant_messages_with_usage": sum(row.assistant_messages_with_usage for row in rows),
        "tool_messages": sum(row.tool_messages for row in rows),
        "prompt_tokens": sum(row.prompt_tokens for row in rows),
        "completion_tokens": sum(row.completion_tokens for row in rows),
        "total_tokens": sum(row.total_tokens for row in rows),
        "instance_cost": round(sum(row.instance_cost for row in rows), 6),
    }


def build_task_map(path: Path) -> dict[str, TaskStats]:
    return {row.instance_id: row for row in (load_task_stats(traj_path) for traj_path in find_traj_files(path))}


def build_comparison_rows(left_rows: dict[str, TaskStats], right_rows: dict[str, TaskStats]) -> list[dict[str, Any]]:
    shared_ids = sorted(set(left_rows) & set(right_rows))
    rows = []
    for instance_id in shared_ids:
        left = left_rows[instance_id]
        right = right_rows[instance_id]
        rows.append(
            {
                "instance_id": instance_id,
                "left_api_calls": left.api_calls,
                "right_api_calls": right.api_calls,
                "api_calls_diff": right.api_calls - left.api_calls,
                "left_prompt_tokens": left.prompt_tokens,
                "right_prompt_tokens": right.prompt_tokens,
                "prompt_tokens_diff": right.prompt_tokens - left.prompt_tokens,
                "left_completion_tokens": left.completion_tokens,
                "right_completion_tokens": right.completion_tokens,
                "completion_tokens_diff": right.completion_tokens - left.completion_tokens,
                "left_total_tokens": left.total_tokens,
                "right_total_tokens": right.total_tokens,
                "total_tokens_diff": right.total_tokens - left.total_tokens,
                "left_exit_status": left.exit_status,
                "right_exit_status": right.exit_status,
                "left_traj_path": left.traj_path,
                "right_traj_path": right.traj_path,
            }
        )
    return rows


def build_comparison_summary(
    left_rows: dict[str, TaskStats], right_rows: dict[str, TaskStats], comparison_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "left_tasks": len(left_rows),
        "right_tasks": len(right_rows),
        "shared_tasks": len(comparison_rows),
        "left_only_tasks": len(set(left_rows) - set(right_rows)),
        "right_only_tasks": len(set(right_rows) - set(left_rows)),
        "left_api_calls": sum(left_rows[row["instance_id"]].api_calls for row in comparison_rows),
        "right_api_calls": sum(right_rows[row["instance_id"]].api_calls for row in comparison_rows),
        "api_calls_diff": sum(row["api_calls_diff"] for row in comparison_rows),
        "left_prompt_tokens": sum(left_rows[row["instance_id"]].prompt_tokens for row in comparison_rows),
        "right_prompt_tokens": sum(right_rows[row["instance_id"]].prompt_tokens for row in comparison_rows),
        "prompt_tokens_diff": sum(row["prompt_tokens_diff"] for row in comparison_rows),
        "left_completion_tokens": sum(left_rows[row["instance_id"]].completion_tokens for row in comparison_rows),
        "right_completion_tokens": sum(right_rows[row["instance_id"]].completion_tokens for row in comparison_rows),
        "completion_tokens_diff": sum(row["completion_tokens_diff"] for row in comparison_rows),
        "left_total_tokens": sum(left_rows[row["instance_id"]].total_tokens for row in comparison_rows),
        "right_total_tokens": sum(right_rows[row["instance_id"]].total_tokens for row in comparison_rows),
        "total_tokens_diff": sum(row["total_tokens_diff"] for row in comparison_rows),
    }


def get_comparison_sort_value(row: dict[str, Any], sort_by: str) -> Any:
    sort_key_map = {
        "instance_id": "instance_id",
        "api_calls": "api_calls_diff",
        "prompt_tokens": "prompt_tokens_diff",
        "completion_tokens": "completion_tokens_diff",
        "total_tokens": "total_tokens_diff",
        "instance_cost": "total_tokens_diff",
    }
    return row[sort_key_map[sort_by]]


def render_table(rows: list[TaskStats], summary: dict[str, Any]) -> str:
    headers = [
        "instance_id",
        "api_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "exit_status",
    ]
    body = [
        [
            row.instance_id,
            str(row.api_calls),
            str(row.prompt_tokens),
            str(row.completion_tokens),
            str(row.total_tokens),
            row.exit_status,
        ]
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for line in body:
        for i, cell in enumerate(line):
            widths[i] = max(widths[i], len(cell))

    table_lines = [
        "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    table_lines.extend("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)) for line in body)
    table_lines.append("")
    table_lines.append(
        "SUMMARY "
        f"tasks={summary['tasks']} "
        f"api_calls={summary['api_calls']} "
        f"prompt_tokens={summary['prompt_tokens']} "
        f"completion_tokens={summary['completion_tokens']} "
        f"total_tokens={summary['total_tokens']} "
        f"instance_cost={summary['instance_cost']}"
    )
    return "\n".join(table_lines)


def render_csv(rows: list[TaskStats]) -> str:
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(TaskStats.__dataclass_fields__.keys())
    output_lines: list[str] = []

    class _ListWriter:
        def write(self, value: str) -> int:
            output_lines.append(value)
            return len(value)

    writer = csv.DictWriter(_ListWriter(), fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(asdict(row))
    return "".join(output_lines)


def render_comparison_table(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    headers = [
        "instance_id",
        "left_total_tokens",
        "right_total_tokens",
        "total_tokens_diff",
        "left_api_calls",
        "right_api_calls",
        "api_calls_diff",
    ]
    body = [[str(row[header]) for header in headers] for row in rows]
    widths = [len(header) for header in headers]
    for line in body:
        for i, cell in enumerate(line):
            widths[i] = max(widths[i], len(cell))

    table_lines = [
        "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    table_lines.extend("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)) for line in body)
    table_lines.append("")
    table_lines.append(
        "SUMMARY "
        f"shared_tasks={summary['shared_tasks']} "
        f"left_only_tasks={summary['left_only_tasks']} "
        f"right_only_tasks={summary['right_only_tasks']} "
        f"left_total_tokens={summary['left_total_tokens']} "
        f"right_total_tokens={summary['right_total_tokens']} "
        f"total_tokens_diff={summary['total_tokens_diff']} "
        f"left_api_calls={summary['left_api_calls']} "
        f"right_api_calls={summary['right_api_calls']} "
        f"api_calls_diff={summary['api_calls_diff']}"
    )
    return "\n".join(table_lines)


def render_comparison_csv(rows: list[dict[str, Any]]) -> str:
    fieldnames = list(rows[0].keys()) if rows else [
        "instance_id",
        "left_api_calls",
        "right_api_calls",
        "api_calls_diff",
        "left_prompt_tokens",
        "right_prompt_tokens",
        "prompt_tokens_diff",
        "left_completion_tokens",
        "right_completion_tokens",
        "completion_tokens_diff",
        "left_total_tokens",
        "right_total_tokens",
        "total_tokens_diff",
        "left_exit_status",
        "right_exit_status",
        "left_traj_path",
        "right_traj_path",
    ]
    output_lines: list[str] = []

    class _ListWriter:
        def write(self, value: str) -> int:
            output_lines.append(value)
            return len(value)

    writer = csv.DictWriter(_ListWriter(), fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return "".join(output_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize mini-SWE-agent bench trajectories by task, token usage, and API call count."
    )
    parser.add_argument("path", type=Path, help="Bench directory or a single .traj.json file")
    parser.add_argument("compare_path", type=Path, nargs="?", help="Optional second bench directory for intersection-only comparison")
    parser.add_argument(
        "--format",
        choices=("table", "json", "csv"),
        default="table",
        help="Output format",
    )
    parser.add_argument(
        "--sort-by",
        choices=("instance_id", "api_calls", "prompt_tokens", "completion_tokens", "total_tokens", "instance_cost"),
        default="total_tokens",
        help="Sort key for per-task rows",
    )
    parser.add_argument("--descending", action="store_true", help="Sort descending")
    parser.add_argument("-o", "--output", type=Path, help="Write result to a file instead of stdout")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.compare_path is not None:
        left_rows = build_task_map(args.path)
        right_rows = build_task_map(args.compare_path)
        if not left_rows:
            print(f"No .traj.json files found under {args.path}", file=sys.stderr)
            return 1
        if not right_rows:
            print(f"No .traj.json files found under {args.compare_path}", file=sys.stderr)
            return 1

        rows = build_comparison_rows(left_rows, right_rows)
        rows.sort(key=lambda row: get_comparison_sort_value(row, args.sort_by), reverse=args.descending)
        summary = build_comparison_summary(left_rows, right_rows, rows)

        if args.format == "json":
            result = json.dumps({"summary": summary, "tasks": rows}, ensure_ascii=False, indent=2)
        elif args.format == "csv":
            result = render_comparison_csv(rows)
        else:
            result = render_comparison_table(rows, summary)
    else:
        traj_files = find_traj_files(args.path)
        if not traj_files:
            print(f"No .traj.json files found under {args.path}", file=sys.stderr)
            return 1

        rows = [load_task_stats(path) for path in traj_files]
        rows.sort(key=lambda row: getattr(row, args.sort_by), reverse=args.descending)
        summary = build_summary(rows)

        if args.format == "json":
            result = json.dumps(
                {"summary": summary, "tasks": [asdict(row) for row in rows]},
                ensure_ascii=False,
                indent=2,
            )
        elif args.format == "csv":
            result = render_csv(rows)
        else:
            result = render_table(rows, summary)

    if args.output:
        args.output.write_text(result)
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
