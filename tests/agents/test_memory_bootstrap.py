import httpx
import pytest

from minisweagent._vendor.formsy_sdk import SourceFile
from minisweagent.agents.memory_bootstrap import MemoryBootstrapAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output


class _FakeExploreResult:
    def __init__(self, content):
        self.content = content


class _EvidenceClient:
    """Stand-in for formsy_sdk.Client — records calls, returns shell objects."""

    def __init__(self):
        self.ingest_calls = []
        self.explore_calls = []

    def ingest(self, repo_id, revision, files, *, source_type="code"):
        self.ingest_calls.append({"repo_id": repo_id, "revision": revision, "files": list(files)})
        return None

    def explore(self, repo_id, revision, query, *, max_output_chars=None, **kwargs):
        self.explore_calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "query": query,
                "max_output_chars": max_output_chars,
            }
        )
        return _FakeExploreResult("### ctx")


_SUBMISSION = make_output(
    "finish",
    [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}],
)


def test_memory_bootstrap_agent_injects_context_once(monkeypatch):
    client = _EvidenceClient()
    monkeypatch.setattr(
        "minisweagent.agents.memory_bootstrap.extract_source_files",
        lambda env, cwd="": [SourceFile(path="pkg/mod.py", content="x=1")],
    )
    monkeypatch.setattr(
        "minisweagent.agents.memory_bootstrap.resolve_revision",
        lambda env, cwd="": "deadbeef",
    )

    agent = MemoryBootstrapAgent(
        model=DeterministicModel(outputs=[_SUBMISSION]),
        env=LocalEnvironment(),
        system_template="sys",
        instance_template="task={{task}}",
        memory={"enabled": True, "max_output_chars": 1234},
        evidence_client=client,
    )

    info = agent.run("issue text", instance_id="repo__1")

    assert info["exit_status"] == "Submitted"
    assert client.ingest_calls == [
        {
            "repo_id": "repo__1",
            "revision": "deadbeef",
            "files": [SourceFile(path="pkg/mod.py", content="x=1")],
        }
    ]
    assert client.explore_calls == [
        {
            "repo_id": "repo__1",
            "revision": "deadbeef",
            "query": "issue text",
            "max_output_chars": 1234,
        }
    ]
    assert agent.memory_state["bootstrapped"] is True
    assert [msg["role"] for msg in agent.messages[:4]] == ["system", "user", "assistant", "tool"]
    assert agent.messages[3]["content"] == "### ctx"


def test_memory_bootstrap_agent_degrades_on_explore_failure(monkeypatch):
    class _FailingClient(_EvidenceClient):
        def explore(self, *args, **kwargs):
            raise httpx.HTTPError("explore failed")

    monkeypatch.setattr(
        "minisweagent.agents.memory_bootstrap.extract_source_files", lambda env, cwd="": []
    )
    monkeypatch.setattr(
        "minisweagent.agents.memory_bootstrap.resolve_revision", lambda env, cwd="": "deadbeef"
    )

    agent = MemoryBootstrapAgent(
        model=DeterministicModel(outputs=[_SUBMISSION]),
        env=LocalEnvironment(),
        system_template="sys",
        instance_template="task={{task}}",
        memory={"enabled": True},
        evidence_client=_FailingClient(),
    )

    # Must NOT raise — degrades gracefully (no injected context), run completes.
    info = agent.run("issue text", instance_id="repo__1")
    assert info["exit_status"] == "Submitted"
    assert agent.memory_state["bootstrapped"] is True
