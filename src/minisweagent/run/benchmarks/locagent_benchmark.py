#!/usr/bin/env python3
"""LocAgent benchmark runner — reuse mini-swe-agent's infrastructure to produce
LocAgent-format outputs for SWE-bench_Lite dataset validation.

This runner loads instances from ``czlll/SWE-bench_Lite`` (which includes the
``edit_functions`` ground-truth field), runs the mini-swe-agent agent with a
*localization-only* prompt, and writes results in the LocAgent JSONL format so
they can be evaluated with LocAgent's ``eval_metric.py``.
"""

import concurrent.futures
import json
import logging
import os
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import typer
from datasets import load_dataset
from rich.live import Live

from minisweagent.agents import get_agent
from minisweagent.agents.context_tool_agent import ContextToolAgent
from minisweagent.agents.memory_bootstrap import MemoryBootstrapAgent
from minisweagent.agents.pycodegraph_agent import PycodeGraphAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import (
    get_sb_environment,
    load_swebench_dataset,
)
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "locagent.yaml"
APP = typer.Typer(rich_markup_mode="rich", add_completion=False)
_OUTPUT_FILE_LOCK = threading.Lock()

# ── LocAgent output parsing ────────────────────────────────────────────────

# Reuse LocAgent's "czlll/SWE-bench_Lite" dataset which has edit_functions
LOCAGENT_DATASET = "czlll/SWE-bench_Lite"


def parse_loc_agent_output(raw_text: str) -> dict[str, list[str]]:
    """Parse the raw loc.txt content into found_files, found_modules, found_entities.

    Follows LocAgent's conventions (see ``util/process_output.py``):
    - Lines ending with ``.py`` are treated as file paths.
    - Lines starting with ``function:``, ``class:``, or ``method:`` (followed by a
      qualified name) are treated as entity locations. The prefix is stripped and
      the name is stored as ``file.py:QualifiedName``.
    - ``line:`` / ``lines:`` entries identify the file but do NOT produce entity
      records; they only confirm the file is relevant.
    - If a ``function`` or ``class`` name contains a dot (e.g. ``MyClass.my_method``),
      it is a method and the module is the parent class.
    """
    found_files: list[str] = []
    found_entities: list[str] = []
    current_file: str | None = None

    lines = raw_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect file path (ends with .py)
        if line.endswith(".py"):
            fn = _extract_python_path(line)
            if fn:
                current_file = fn
                if fn not in found_files:
                    found_files.append(fn)
            else:
                current_file = None
            continue

        # Handle function:/class:/method: lines — these produce entities
        for prefix in ("function:", "class:", "method:"):
            if line.startswith(prefix):
                name = line[len(prefix):].strip().split()[0] if line[len(prefix):].strip() else ""
                if current_file and name:
                    entity = f"{current_file}:{name}"
                    if entity not in found_entities:
                        found_entities.append(entity)
                break  # matched, skip remaining prefixes

    # Derive modules from entities (strip method suffix for class methods)
    found_modules = _derive_modules(found_entities)

    return {
        "found_files": found_files,
        "found_modules": found_modules,
        "found_entities": found_entities,
    }


def _extract_python_path(line: str) -> str | None:
    """Extract a Python file path from a line of text."""
    # Remove leading non-path content (e.g. line numbers or markers)
    match = re.search(r"([\w./-]+\.py)", line)
    if match:
        path = match.group(1)
        # Strip leading Docker container working dir prefixes (/app/, /testbed/)
        for prefix in ("/app/", "/testbed/"):
            if path.startswith(prefix):
                path = path[len(prefix):]
                break
        return path
    return None


def _derive_modules(entities: list[str]) -> list[str]:
    """Derive module names from entities.

    A module is the top-level entity name (class or function) without method suffixes.
    e.g. ``file.py:MyClass.my_method`` → module is ``file.py:MyClass``
         ``file.py:my_function`` → module is ``file.py:my_function``
    """
    modules: list[str] = []
    seen: set[str] = set()
    for entity in entities:
        parts = entity.split(":", 1)
        if len(parts) != 2:
            continue
        file_part, name_part = parts
        # If name contains a dot, it's a method — strip to class name
        if "." in name_part:
            module_name = name_part.rsplit(".", 1)[0]
        else:
            module_name = name_part
        full = f"{file_part}:{module_name}"
        if full not in seen:
            seen.add(full)
            modules.append(full)
    return modules


def extract_submission(agent_result: dict) -> str:
    """Extract the loc.txt content from the agent's submission."""
    return agent_result.get("submission", "")


# ── Progress-tracking agent wrappers ───────────────────────────────────────


class LocAgentProgressTracking(MemoryBootstrapAgent):
    """Wraps DefaultAgent with progress reporting for LocAgent runs."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager | None = None, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        if self.progress_manager is not None:
            self.progress_manager.update_instance_status(self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})")
        return super().step()


class LocAgentContextToolTracking(ContextToolAgent):
    """Wraps ContextToolAgent with progress reporting for LocAgent runs."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager | None = None, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        if self.progress_manager is not None:
            self.progress_manager.update_instance_status(self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})")
        return super().step()


class LocAgentPycodeGraphTracking(PycodeGraphAgent):
    """Wraps PycodeGraphAgent with progress reporting for LocAgent runs."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager | None = None, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        if self.progress_manager is not None:
            self.progress_manager.update_instance_status(self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})")
        return super().step()


# ── Process a single instance ──────────────────────────────────────────────


def _filter_entity_list(
    entities: list[str],
    valid_prefixes: list[str] = ("function:", "class:", "method:"),
) -> list[str]:
    """Filter entities to only include those with recognized prefixes."""
    return [e for e in entities if any(e.split(":", 1)[-1].startswith(p) for p in valid_prefixes)]


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
) -> None:
    """Process a single SWE-bench instance and produce a LocAgent-format output."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    model = get_model(config=config.get("model", {}))
    task = instance["problem_statement"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting environment")

    agent = None
    exit_status = None
    submission = ""
    extra_info: dict[str, Any] = {}

    try:
        env = get_sb_environment(config, instance)
        agent_class_name = config.get("agent", {}).get("agent_class", "")
        if agent_class_name == "context_tool":
            agent = LocAgentContextToolTracking(
                model,
                env,
                progress_manager=progress_manager,
                instance_id=instance_id,
                memory=config.get("memory", {}),
                **config.get("agent", {}),
            )
        elif agent_class_name == "pycodegraph":
            agent = LocAgentPycodeGraphTracking(
                model,
                env,
                progress_manager=progress_manager,
                instance_id=instance_id,
                pycodegraph=config.get("pycodegraph", {}),
                **config.get("agent", {}),
            )
        else:
            agent = LocAgentProgressTracking(
                model,
                env,
                progress_manager=progress_manager,
                instance_id=instance_id,
                memory=config.get("memory", {}),
                **config.get("agent", {}),
            )
        info = agent.run(task, instance_id=instance_id)
        exit_status = info.get("exit_status")
        submission = info.get("submission", "")
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status = type(e).__name__
        submission = ""
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    finally:
        # Save trajectory
        if agent is not None:
            traj_path = instance_dir / f"{instance_id}.traj.json"
            agent.save(
                traj_path,
                {
                    "info": {
                        "exit_status": exit_status,
                        "submission": submission,
                        **extra_info,
                    },
                    "instance_id": instance_id,
                },
            )
            logger.info(f"Saved trajectory to '{traj_path}'")

        # Parse loc.txt submission into LocAgent format
        loc_output = parse_loc_agent_output(submission)

        # Build LocAgent-format record
        # Store raw output in the format LocAgent expects
        raw_output_loc = [submission] if submission else []

        loc_record = {
            "instance_id": instance_id,
            "found_files": loc_output["found_files"],
            "found_modules": loc_output["found_modules"],
            "found_entities": loc_output["found_entities"],
            "raw_output_loc": raw_output_loc,
            "meta_data": {
                "repo": instance.get("repo", ""),
                "base_commit": instance.get("base_commit", ""),
                "problem_statement": instance.get("problem_statement", ""),
                "patch": instance.get("patch", ""),
            },
        }

        # Write to LocAgent JSONL
        locagent_path = output_dir / "loc_outputs.jsonl"
        with _OUTPUT_FILE_LOCK:
            with open(locagent_path, "a") as f:
                f.write(json.dumps(loc_record) + "\n")

        progress_manager.on_instance_end(instance_id, exit_status)

        # Clean up Docker container + image to free disk for the next case
        # (each case pulls a different ~1.5GB image; without this, 30 cases fill the disk)
        try:
            import subprocess
            # Stop and remove any minisweagent containers
            subprocess.run(
                "docker ps -q --filter name=minisweagent | xargs -r docker rm -f",
                shell=True, capture_output=True, timeout=30,
            )
            # Remove sweap-images that are no longer in use
            subprocess.run(
                "docker images --format '{{.Repository}}:{{.Tag}}' | grep sweap-images | "
                "while read img; do docker rmi \"$img\" 2>/dev/null || true; done",
                shell=True, capture_output=True, timeout=60,
            )
            logger.debug(f"Cleaned up Docker resources after {instance_id}")
        except Exception as cleanup_err:
            logger.debug(f"Docker cleanup skipped: {cleanup_err}")


# ── CLI ────────────────────────────────────────────────────────────────────


@APP.command()
def main(
    subset: str = typer.Option("czlll/SWE-bench_Lite", "--subset", help="Dataset to use (default: czlll/SWE-bench_Lite which has edit_functions)"),
    split: str = typer.Option("test", "--split", help="Dataset split"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads", rich_help_panel="Basic"),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help="Config specs"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type (docker, etc.)"),
) -> None:
    """Run mini-swe-agent in localization mode and produce LocAgent-format output.

    This loads a SWE-bench dataset (default: czlll/SWE-bench_Lite which includes
    the edit_functions ground-truth field), runs the agent to LOCATE (not fix)
    the bug, and outputs in the LocAgent JSONL format compatible with
    LocAgent's eval_metric.py.
    """
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    logger.info(f"Loading dataset {subset}, split {split}...")
    instances = list(load_swebench_dataset(subset, split=split))

    # Filtering
    if filter_spec:
        before = len(instances)
        instances = [i for i in instances if re.match(filter_spec, i["instance_id"])]
        logger.info(f"Filter: {before} → {len(instances)} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
    if shuffle:
        import random

        random.seed(42)
        random.shuffle(instances)

    if not redo_existing and (output_path / "loc_outputs.jsonl").exists():
        existing = set()
        with open(output_path / "loc_outputs.jsonl") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    existing.add(rec["instance_id"])
        logger.info(f"Skipping {len(existing)} existing instances")
        instances = [i for i in instances if i["instance_id"] not in existing]

    logger.info(f"Running on {len(instances)} instances...")

    # Build config
    configs = [get_config_from_spec(s) for s in config_spec]
    configs.append({
        "environment": {"environment_class": environment_class or UNSET},
        "model": {"model_name": model_name or UNSET, "model_class": model_class or UNSET},
    })
    config = recursive_merge(*configs)

    progress_manager = RunBatchProgressManager(
        len(instances), output_path / f"exit_statuses_{time.time()}.yaml"
    )

    def process_futures(futures):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                iid = futures[future]
                logger.error(f"Error in future for instance {iid}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(iid, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance, inst, output_path, config, progress_manager): inst[
                    "instance_id"
                ]
                for inst in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling. Press ^C again to exit immediately.")
                for f in futures:
                    if not f.running() and not f.done():
                        f.cancel()
                process_futures(futures)

    logger.info("Done! LocAgent output saved to: %s", output_path / "loc_outputs.jsonl")


if __name__ == "__main__":
    APP()