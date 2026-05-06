import logging
import re
from unittest.mock import patch

import pytest

from minisweagent import package_dir
from minisweagent.agents.memory_bootstrap import MemoryBootstrapAgent
from minisweagent.models.test_models import DeterministicModel, make_output
from minisweagent.run.benchmarks.swebench_single import SingleCaseProgressTrackingAgent, main


def _make_model_from_fixture(text_outputs: list[str], cost_per_call: float = 1.0, **kwargs) -> DeterministicModel:
    """Create a DeterministicModel from trajectory fixture data (raw text outputs)."""

    def parse_command(text: str) -> list[dict]:
        match = re.search(r"```mswea_bash_command\s*\n(.*?)\n```", text, re.DOTALL)
        return [{"command": match.group(1)}] if match else []

    return DeterministicModel(
        outputs=[make_output(text, parse_command(text), cost=cost_per_call) for text in text_outputs],
        cost_per_call=cost_per_call,
        **kwargs,
    )


@pytest.mark.slow
def test_swebench_single_end_to_end(github_test_data, tmp_path, container_executable):
    """Test the swebench_single script using the _test subset with deterministic model.
    This mostly tests that no exception occurs.
    """

    model_responses = github_test_data["model_responses"]

    with (
        patch("minisweagent.run.benchmarks.swebench_single.get_model") as mock_get_model,
        patch("minisweagent.agents.utils.prompt_user.prompt_session.prompt", side_effect=lambda *a, **kw: ""),
        patch(
            "minisweagent.agents.utils.prompt_user._multiline_prompt_session.prompt", side_effect=lambda *a, **kw: ""
        ),
        patch("builtins.input", return_value=""),  # For LimitsExceeded handling
    ):
        mock_get_model.return_value = _make_model_from_fixture(model_responses, cost_per_call=0.1)

        # Test with explicit instance ID
        output_path = tmp_path / "test_output.json"
        main(
            subset="_test",
            split="test",
            instance_spec="swe-agent__test-repo-1",
            model_name="deterministic",
            config_spec=[str(package_dir / "config" / "benchmarks" / "swebench.yaml")],
            environment_class="docker",
            exit_immediately=False,
            output=output_path,
            model_class=None,
            agent_class=None,
            yolo=False,
            cost_limit=None,
        )

        # Verify model was called with correct parameters
        mock_get_model.assert_called_once()
        assert output_path.exists()


@pytest.mark.slow
def test_swebench_single_end_to_end_exit_immediately(github_test_data, tmp_path, container_executable):
    """Test the swebench_single script using the _test subset with deterministic model.
    This mostly tests that no exception occurs.
    This test uses the --exit-immediately flag to exit immediately when the agent wants to finish instead of prompting.
    """

    model_responses = github_test_data["model_responses"]

    with (
        patch("minisweagent.run.benchmarks.swebench_single.get_model") as mock_get_model,
        patch("minisweagent.agents.utils.prompt_user.prompt_session.prompt", side_effect=lambda *a, **kw: ""),
        patch(
            "minisweagent.agents.utils.prompt_user._multiline_prompt_session.prompt", side_effect=lambda *a, **kw: ""
        ),
        patch("builtins.input", return_value=""),  # For LimitsExceeded handling
    ):
        mock_get_model.return_value = _make_model_from_fixture(model_responses, cost_per_call=0.1)

        # Test with explicit instance ID
        output_path = tmp_path / "test_output.json"
        main(
            subset="_test",
            split="test",
            instance_spec="swe-agent__test-repo-1",
            model_name="deterministic",
            config_spec=[str(package_dir / "config" / "benchmarks" / "swebench.yaml")],
            environment_class="docker",
            exit_immediately=True,
            output=output_path,
            model_class=None,
            agent_class=None,
            yolo=False,
            cost_limit=None,
        )

        # Verify model was called with correct parameters
        mock_get_model.assert_called_once()
        assert output_path.exists()


def test_swebench_single_uses_json_loader_for_jsonl_subset(tmp_path):
    """Test that swebench_single accepts a JSONL dataset path."""
    dataset_path = tmp_path / "custom.jsonl"
    dataset_path.write_text('{"instance_id":"demo__repo-1","problem_statement":"Fix it"}\n')

    with (
        patch("minisweagent.run.benchmarks.swebench_single.load_swebench_dataset") as mock_load_dataset,
        patch("minisweagent.run.benchmarks.swebench_single.get_sb_environment") as mock_get_env,
        patch("minisweagent.run.benchmarks.swebench_single.get_model") as mock_get_model,
        patch("minisweagent.run.benchmarks.swebench_single.get_agent") as mock_get_agent,
    ):
        mock_load_dataset.return_value = [{"instance_id": "demo__repo-1", "problem_statement": "Fix it"}]
        mock_agent = mock_get_agent.return_value

        main(
            subset=str(dataset_path),
            split="train",
            instance_spec="demo__repo-1",
            model_name="deterministic",
            config_spec=[str(package_dir / "config" / "benchmarks" / "swebench.yaml")],
            environment_class="docker",
            exit_immediately=True,
            output=tmp_path / "traj.json",
            model_class=None,
            agent_class="interactive",
            yolo=False,
            cost_limit=None,
        )

        mock_load_dataset.assert_called_once_with(str(dataset_path), split="train")
        mock_get_env.assert_called_once()
        mock_get_model.assert_called_once()
        mock_get_agent.assert_called_once()
        mock_agent.run.assert_called_once_with("Fix it", instance_id="demo__repo-1")


def test_swebench_single_uses_memory_bootstrap_agent_when_memory_enabled(tmp_path):
    """Test that swebench_single routes memory config to the memory bootstrap agent."""
    dataset_path = tmp_path / "custom.jsonl"
    dataset_path.write_text('{"instance_id":"demo__repo-1","problem_statement":"Fix it"}\n')

    with (
        patch("minisweagent.run.benchmarks.swebench_single.load_swebench_dataset") as mock_load_dataset,
        patch("minisweagent.run.benchmarks.swebench_single.get_sb_environment") as mock_get_env,
        patch("minisweagent.run.benchmarks.swebench_single.get_model") as mock_get_model,
        patch("minisweagent.run.benchmarks.swebench_single.SingleCaseProgressTrackingAgent") as mock_agent_cls,
    ):
        mock_load_dataset.return_value = [{"instance_id": "demo__repo-1", "problem_statement": "Fix it"}]
        mock_agent = mock_agent_cls.return_value

        main(
            subset=str(dataset_path),
            split="train",
            instance_spec="demo__repo-1",
            model_name="deterministic",
            config_spec=[
                str(package_dir / "config" / "benchmarks" / "swebench.yaml"),
                "memory.enabled=true",
                "memory.base_url=http://memory",
                "memory.query_budget=321",
            ],
            environment_class="docker",
            exit_immediately=True,
            output=tmp_path / "traj.json",
            model_class=None,
            agent_class=None,
            yolo=False,
            cost_limit=None,
        )

        mock_agent_cls.assert_called_once()
        args, kwargs = mock_agent_cls.call_args
        assert kwargs["memory"]["enabled"] is True
        assert kwargs["memory"]["base_url"] == "http://memory"
        assert kwargs["memory"]["query_budget"] == 321
        mock_get_env.assert_called_once()
        mock_get_model.assert_called_once()
        mock_agent.run.assert_called_once_with("Fix it", instance_id="demo__repo-1")


def test_swebench_single_routes_memory_query_agent_config(tmp_path):
    """Test that memory_query receives the top-level memory config."""
    dataset_path = tmp_path / "custom.jsonl"
    dataset_path.write_text('{"instance_id":"demo__repo-1","problem_statement":"Fix it"}\n')

    with (
        patch("minisweagent.run.benchmarks.swebench_single.load_swebench_dataset") as mock_load_dataset,
        patch("minisweagent.run.benchmarks.swebench_single.get_sb_environment") as mock_get_env,
        patch("minisweagent.run.benchmarks.swebench_single.get_model") as mock_get_model,
        patch("minisweagent.run.benchmarks.swebench_single.get_agent") as mock_get_agent,
    ):
        mock_load_dataset.return_value = [{"instance_id": "demo__repo-1", "problem_statement": "Fix it"}]
        mock_agent = mock_get_agent.return_value

        main(
            subset=str(dataset_path),
            split="train",
            instance_spec="demo__repo-1",
            model_name="deterministic",
            config_spec=[
                str(package_dir / "config" / "benchmarks" / "swebench.yaml"),
                "agent.agent_class=memory_query",
                "memory.enabled=true",
                "memory.base_url=http://memory",
                "memory.query_budget=321",
            ],
            environment_class="docker",
            exit_immediately=True,
            output=tmp_path / "traj.json",
            model_class=None,
            agent_class=None,
            yolo=False,
            cost_limit=None,
        )

        mock_get_agent.assert_called_once()
        _, _, agent_config = mock_get_agent.call_args.args
        assert agent_config["agent_class"] == "memory_query"
        assert agent_config["memory"]["base_url"] == "http://memory"
        assert agent_config["memory"]["query_budget"] == 321
        mock_get_env.assert_called_once()
        mock_get_model.assert_called_once()
        mock_agent.run.assert_called_once_with("Fix it", instance_id="demo__repo-1")


def test_single_case_progress_tracking_agent_logs_step_progress(caplog):
    model = DeterministicModel(outputs=[make_output("done", [], cost=0.25)], cost_per_call=0.25)

    class _Env:
        def execute(self, action):
            return {"output": "", "returncode": 0, "exception_info": ""}

        def get_template_vars(self):
            return {}

        def serialize(self):
            return {}

    with caplog.at_level(logging.INFO):
        agent = SingleCaseProgressTrackingAgent(
            model,
            _Env(),
            system_template="system",
            instance_template="{{ task }}",
        )
        with patch.object(MemoryBootstrapAgent, "step", return_value=[]):
            agent.step()

    assert "Step   1 ($0.00)" in caplog.text
