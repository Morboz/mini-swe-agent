"""Prepare a local Python repo/venv and run a single SWE-bench case."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import typer

from minisweagent import package_dir

DEFAULT_CONFIG_PATH = package_dir / "config" / "benchmarks" / "swebench.yaml"
DEFAULT_SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
SWEBENCH_SINGLE_PATH = Path(__file__).with_name("swebench_single.py")

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


def run_checked(command: list[str]) -> None:
    """Run a command and raise on failure."""
    subprocess.run(command, check=True)


def guess_repo_url(repo_name: str) -> str:
    """Map an owner/repo name to the default HTTPS clone URL."""
    return f"https://github.com/{repo_name}.git"


def get_instance(dataset_path: Path, *, instance_id: str | None = None, index: int | None = None) -> dict:
    """Return one instance from a local JSONL dataset."""
    if (instance_id is None) == (index is None):
        raise ValueError("Specify exactly one of instance_id or index")

    with dataset_path.open(encoding="utf-8") as f:
        instances = [json.loads(line) for line in f if line.strip()]

    if instance_id is not None:
        for instance in instances:
            if instance.get("instance_id") == instance_id:
                return instance
        raise KeyError(f"Instance not found: {instance_id}")

    assert index is not None
    try:
        return instances[index]
    except IndexError as e:
        raise IndexError(f"Instance index out of range: {index}") from e


def ensure_repo_ready(*, repo_dir: Path, repo_url: str, base_commit: str) -> None:
    """Clone the repo if needed and reset it to the target commit."""
    if not repo_dir.exists():
        run_checked(["git", "clone", repo_url, str(repo_dir)])
    elif not (repo_dir / ".git").exists():
        raise ValueError(f"Repo dir exists but is not a git checkout: {repo_dir}")

    run_checked(["git", "-C", str(repo_dir), "reset", "--hard"])
    run_checked(["git", "-C", str(repo_dir), "clean", "-fdx"])
    run_checked(["git", "-C", str(repo_dir), "checkout", base_commit])


def ensure_venv_ready(*, env_dir: Path, repo_dir: Path, recreate: bool, skip_install: bool) -> None:
    """Create or reuse a uv-managed venv and install the repo editable package."""
    if recreate and env_dir.exists():
        shutil.rmtree(env_dir)

    if not (env_dir / "pyvenv.cfg").exists():
        run_checked(["uv", "venv", str(env_dir)])

    if skip_install:
        return

    venv_python = env_dir / "bin" / "python"
    run_checked(["uv", "pip", "install", "--python", str(venv_python), "-U", "pip", "setuptools", "wheel"])
    run_checked(["uv", "pip", "install", "--python", str(venv_python), "-e", str(repo_dir)])


def build_runner_command(
    *,
    agent_python: Path,
    dataset: Path,
    instance_id: str,
    repo_dir: Path,
    env_dir: Path,
    config_path: Path,
    yolo: bool,
    model: str | None,
    output_path: Path | None = None,
    extra_config: list[str] | None = None,
) -> list[str]:
    """Build the swebench_single.py invocation."""
    dataset = dataset.resolve()
    repo_dir = repo_dir.resolve()
    env_dir = env_dir.resolve()
    config_path = config_path.resolve()
    command = [
        str(agent_python),
        str(SWEBENCH_SINGLE_PATH),
        "--subset",
        str(dataset),
        "--split",
        "train",
        "--instance",
        instance_id,
        "--environment-class",
        "local",
    ]
    if yolo:
        command.append("-y")
    if model:
        command.extend(["-m", model])
    command.extend(
        [
            "-c",
            str(config_path),
            "-c",
            f"environment.cwd={repo_dir}",
            "-c",
            f"environment.env.VIRTUAL_ENV={env_dir}",
            "-c",
            f"environment.env.PATH={env_dir / 'bin'}:{DEFAULT_SYSTEM_PATH}",
            "--exit-immediately",
        ]
    )
    if output_path is not None:
        command.extend(["-o", str(output_path)])
    for spec in extra_config or []:
        command.extend(["-c", spec])
    return command


def run_single_case(
    *,
    dataset: Path,
    repo_dir: Path,
    env_dir: Path,
    instance_id: str | None,
    index: int | None,
    repo_url: str | None,
    agent_python: Path,
    config_path: Path,
    model: str | None,
    yolo: bool,
    recreate_venv: bool,
    skip_install: bool,
    output_path: Path | None,
    extra_config: list[str] | None,
) -> int:
    """Prepare local Python resources and invoke swebench_single.py."""
    instance = get_instance(dataset, instance_id=instance_id, index=index)
    resolved_repo_url = repo_url or guess_repo_url(instance["repo"])

    ensure_repo_ready(repo_dir=repo_dir, repo_url=resolved_repo_url, base_commit=instance["base_commit"])
    ensure_venv_ready(env_dir=env_dir, repo_dir=repo_dir, recreate=recreate_venv, skip_install=skip_install)

    command = build_runner_command(
        agent_python=agent_python,
        dataset=dataset,
        instance_id=instance["instance_id"],
        repo_dir=repo_dir,
        env_dir=env_dir,
        config_path=config_path,
        yolo=yolo,
        model=model,
        output_path=output_path,
        extra_config=extra_config,
    )
    return subprocess.run(command, check=False).returncode


@app.command()
def main(
    dataset: Path = typer.Option(..., "--dataset", exists=True, dir_okay=False, help="Local SWE-bench JSONL file."),
    repo_dir: Path = typer.Option(..., "--repo-dir", help="Fixed checkout path to clean and reuse."),
    env_dir: Path = typer.Option(..., "--env-dir", help="uv Python venv path to create or reuse."),
    instance_id: str | None = typer.Option(None, "--instance", help="Target instance_id from the JSONL dataset."),
    index: int | None = typer.Option(None, "--index", help="0-based index into the JSONL dataset."),
    repo_url: str | None = typer.Option(None, "--repo-url", help="Explicit git clone URL. Defaults from case repo."),
    agent_python: Path = typer.Option(Path(sys.executable), "--agent-python", help="Python used to run swebench_single."),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Base swebench config."),
    model: str | None = typer.Option(None, "--model", "-m", help="Optional model override."),
    yolo: bool = typer.Option(True, "--yolo/--no-yolo", help="Run swebench_single in auto-approve mode."),
    recreate_venv: bool = typer.Option(False, "--recreate-venv", help="Delete and recreate the target venv."),
    skip_install: bool = typer.Option(False, "--skip-install", help="Reuse the existing venv without reinstalling."),
    output_path: Path | None = typer.Option(None, "--output", help="Optional trajectory output path."),
    extra_config: list[str] = typer.Option([], "--extra-config", help="Additional -c config specs for swebench_single."),
) -> None:
    """Prepare a local Python repo/venv and run one SWE-bench case."""
    returncode = run_single_case(
        dataset=dataset,
        repo_dir=repo_dir,
        env_dir=env_dir,
        instance_id=instance_id,
        index=index,
        repo_url=repo_url,
        agent_python=agent_python,
        config_path=config_path,
        model=model,
        yolo=yolo,
        recreate_venv=recreate_venv,
        skip_install=skip_install,
        output_path=output_path,
        extra_config=extra_config,
    )
    raise typer.Exit(returncode)


if __name__ == "__main__":
    app()
