#!/usr/bin/env python3
"""Run a single SWE-bench instance in LocAgent mode (localization only).

Usage::

    # Using the registered mini-extra subcommand (after registering)
    mini-extra locagent-single \\
        -i django__django-14787 \\
        -m openai/glm-5.1 \\
        -c locagent.yaml

    # Or directly
    python -m minisweagent.run.benchmarks.locagent_single \\
        -i django__django-14787 \\
        -m openai/glm-5.1
"""

from pathlib import Path

import typer
from datasets import load_dataset

from minisweagent import global_config_dir
from minisweagent.agents import get_agent
from minisweagent.agents.context_tool_agent import ContextToolAgent
from minisweagent.agents.memory_bootstrap import MemoryBootstrapAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.models import get_model
from minisweagent.run.benchmarks.locagent_benchmark import (
    LocAgentContextToolTracking,
    LocAgentProgressTracking,
    parse_loc_agent_output,
)
from minisweagent.run.benchmarks.swebench import (
    get_sb_environment,
    load_swebench_dataset,
)
from minisweagent.utils.log import logger
from minisweagent.utils.serialize import UNSET, recursive_merge

DEFAULT_OUTPUT_FILE = global_config_dir / "last_locagent_run.traj.json"
DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "locagent.yaml"

APP = typer.Typer(rich_markup_mode="rich", add_completion=False)


@APP.command()
def main(
    subset: str = typer.Option("czlll/SWE-bench_Lite", "--subset", help="Dataset to load from"),
    split: str = typer.Option("test", "--split", help="Dataset split"),
    instance_spec: str = typer.Option(0, "-i", "--instance", help="Instance ID or index"),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class"),
    agent_class: str | None = typer.Option(None, "--agent-class", help="Agent class"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment class"),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
    cost_limit: float | None = typer.Option(None, "-l", "--cost-limit", help="Cost limit"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help="Config specs"),
    exit_immediately: bool = typer.Option(False, "--exit-immediately", help="Exit immediately when finished"),
    output: Path | None = typer.Option(DEFAULT_OUTPUT_FILE, "-o", "--output", help="Output trajectory file"),
) -> None:
    """Run a single SWE-bench instance in localization (LocAgent) mode."""
    logger.info(f"Loading dataset from {subset}, split {split}...")
    instances = {inst["instance_id"]: inst for inst in load_swebench_dataset(subset, split=split)}
    if instance_spec.isnumeric():
        instance_spec = sorted(instances.keys())[int(instance_spec)]
    instance = instances[instance_spec]

    logger.info(f"Instance: {instance['instance_id']}")
    logger.info(f"Repo: {instance.get('repo', 'unknown')}")

    # Build config
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "agent": {
            "agent_class": agent_class or UNSET,
            "mode": "yolo" if yolo else UNSET,
            "cost_limit": cost_limit or UNSET,
            "confirm_exit": False if exit_immediately else UNSET,
            "output_path": output or UNSET,
        },
        "model": {
            "model_class": model_class or UNSET,
            "model_name": model_name or UNSET,
        },
        "environment": {
            "environment_class": environment_class or UNSET,
        },
    })
    config = recursive_merge(*configs)

    env = get_sb_environment(config, instance)
    agent_config = config.get("agent", {}).copy()

    agent_class_name = agent_config.get("agent_class")
    if agent_class_name == "context_tool":
        agent_config.pop("agent_class", None)
        agent_config["memory"] = config.get("memory", {})
        agent = LocAgentContextToolTracking(
            get_model(config=config.get("model", {})),
            env,
            progress_manager=None,  # type: ignore
            instance_id=instance["instance_id"],
            **agent_config,
        )
    else:
        agent = LocAgentProgressTracking(
            get_model(config=config.get("model", {})),
            env,
            progress_manager=None,  # type: ignore
            instance_id=instance["instance_id"],
            **agent_config,
        )

    info = agent.run(instance["problem_statement"], instance_id=instance["instance_id"])
    exit_status = info.get("exit_status")
    submission = info.get("submission", "")

    logger.info(f"Exit status: {exit_status}")
    if submission:
        logger.info(f"Submission length: {len(submission)} chars")
        loc_output = parse_loc_agent_output(submission)
        logger.info(f"Found files: {loc_output['found_files']}")
        logger.info(f"Found modules: {loc_output['found_modules']}")
        logger.info(f"Found entities: {loc_output['found_entities']}")
    else:
        logger.warning("No submission (empty output)")

    # Save trajectory
    if output:
        traj = agent.save(output)
        logger.info(f"Saved trajectory to '{output}'")

    return agent


if __name__ == "__main__":
    APP()