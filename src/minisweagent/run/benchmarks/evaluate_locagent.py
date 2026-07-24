#!/usr/bin/env python3
"""Evaluate LocAgent-format output (produced by mini-swe-agent's locagent runner)
using LocAgent's evaluation metrics.

This wraps LocAgent's ``eval_metric.evaluate_results`` so you can run::

    python -m minisweagent.run.benchmarks.evaluate_locagent \\
        --loc-file ./results/loc_outputs.jsonl \\
        --dataset czlll/SWE-bench_Lite
"""

import sys
from pathlib import Path

import typer

APP = typer.Typer(rich_markup_mode="rich", add_completion=False)

LEVEL2KEY_DICT = {
    "file": "found_files",
    "module": "found_modules",
    "function": "found_entities",
}


@APP.command()
def main(
    loc_file: str = typer.Option(..., "--loc-file", help="Path to loc_outputs.jsonl (LocAgent format)"),
    dataset: str = typer.Option("czlll/SWE-bench_Lite", "--dataset", help="Dataset name"),
    split: str = typer.Option("test", "--split", help="Dataset split"),
    locagent_eval_path: str = typer.Option(
        "", "--locagent-eval-path",
        help="Path to LocAgent's evaluation directory (if not using installed package)",
    ),
    selected_instances: str = typer.Option(
        "", "--select",
        help="Comma-separated list of instance IDs to evaluate (default: all)",
    ),
) -> None:
    """Evaluate LocAgent-format output using LocAgent metrics."""

    # Add LocAgent evaluator to path if needed
    if locagent_eval_path:
        sys.path.insert(0, str(Path(locagent_eval_path).parent))

    # Import LocAgent's evaluation module (must be importable)
    try:
        from evaluation.eval_metric import evaluate_results
    except ImportError:
        print(
            "Error: Could not import LocAgent's evaluate_results. "
            "Install LocAgent or set --locagent-eval-path to its root directory.",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)

    selected_list = None
    if selected_instances:
        selected_list = [s.strip() for s in selected_instances.split(",")]

    result_df = evaluate_results(
        loc_file,
        LEVEL2KEY_DICT,
        dataset=dataset,
        split=split,
        selected_list=selected_list,
        metrics=["acc"],
    )

    print(result_df.to_string())


if __name__ == "__main__":
    APP()