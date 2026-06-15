from pathlib import Path
from unittest.mock import call
from unittest.mock import patch

import pytest

from minisweagent.run.benchmarks.local_python_single import (
    build_runner_command,
    ensure_repo_ready,
    ensure_venv_ready,
    get_instance,
    guess_repo_url,
)


def test_guess_repo_url_uses_github_https():
    assert guess_repo_url("django/django") == "https://github.com/django/django.git"


def test_get_instance_supports_instance_id_and_index(tmp_path):
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                '{"instance_id":"django__django-1","repo":"django/django","base_commit":"abc"}',
                '{"instance_id":"django__django-2","repo":"django/django","base_commit":"def"}',
            ]
        )
        + "\n"
    )

    assert get_instance(dataset_path, instance_id="django__django-2")["base_commit"] == "def"
    assert get_instance(dataset_path, index=0)["instance_id"] == "django__django-1"


def test_ensure_repo_ready_clones_missing_repo_and_resets_existing_repo(tmp_path):
    repo_dir = tmp_path / "django"
    with patch("minisweagent.run.benchmarks.local_python_single.run_checked") as run:
        ensure_repo_ready(repo_dir=repo_dir, repo_url="https://github.com/django/django.git", base_commit="abc123")

        assert run.call_args_list == [
            call(["git", "clone", "https://github.com/django/django.git", str(repo_dir)]),
            call(["git", "-C", str(repo_dir), "reset", "--hard"]),
            call(["git", "-C", str(repo_dir), "clean", "-fdx"]),
            call(["git", "-C", str(repo_dir), "checkout", "abc123"]),
        ]


def test_ensure_venv_ready_reuses_existing_venv_when_skip_install_is_set(tmp_path):
    env_dir = tmp_path / ".venv"
    env_dir.mkdir()
    (env_dir / "pyvenv.cfg").write_text("home = /tmp/python\n")
    with patch("minisweagent.run.benchmarks.local_python_single.run_checked") as run:
        ensure_venv_ready(env_dir=env_dir, repo_dir=tmp_path / "repo", recreate=False, skip_install=True)

        run.assert_not_called()


def test_build_runner_command_passes_local_env_and_yolo(tmp_path):
    dataset = Path("/tmp/cases.jsonl").resolve()
    repo_dir = Path("/tmp/django").resolve()
    env_dir = Path("/tmp/django/.venv").resolve()
    config_path = Path("/tmp/swebench.yaml").resolve()
    command = build_runner_command(
        agent_python=Path("/tmp/agent/bin/python"),
        dataset=dataset,
        instance_id="django__django-11211",
        repo_dir=repo_dir,
        env_dir=env_dir,
        config_path=config_path,
        yolo=True,
        model="openai/glm-5.1",
    )

    assert command == [
        "/tmp/agent/bin/python",
        str(Path(__file__).parents[2] / "src/minisweagent/run/benchmarks/swebench_single.py"),
        "--subset",
        str(dataset),
        "--split",
        "train",
        "--instance",
        "django__django-11211",
        "--environment-class",
        "local",
        "-y",
        "-m",
        "openai/glm-5.1",
        "-c",
        str(config_path),
        "-c",
        f"environment.cwd={repo_dir}",
        "-c",
        f"environment.env.VIRTUAL_ENV={env_dir}",
        "-c",
        f"environment.env.PATH={env_dir}/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "--exit-immediately",
    ]


def test_build_runner_command_normalizes_repo_and_env_paths_to_absolute():
    command = build_runner_command(
        agent_python=Path("/tmp/agent/bin/python"),
        dataset=Path("/tmp/cases.jsonl"),
        instance_id="django__django-11211",
        repo_dir=Path("./runs/data/repos/django__django"),
        env_dir=Path("./runs/data/envs/django__django"),
        config_path=Path("/tmp/swebench.yaml"),
        yolo=False,
        model=None,
    )

    repo_dir = str(Path("./runs/data/repos/django__django").resolve())
    env_dir = str(Path("./runs/data/envs/django__django").resolve())

    assert f"environment.cwd={repo_dir}" in command
    assert f"environment.env.VIRTUAL_ENV={env_dir}" in command
    assert f"environment.env.PATH={env_dir}/bin:/usr/bin:/bin:/usr/sbin:/sbin" in command


def test_get_instance_requires_exactly_one_selector(tmp_path):
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text('{"instance_id":"django__django-1","repo":"django/django","base_commit":"abc"}\n')

    with pytest.raises(ValueError, match="exactly one"):
        get_instance(dataset_path)
