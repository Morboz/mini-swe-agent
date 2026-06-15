import pytest

from minisweagent.agents.memory_bootstrap import MemoryBootstrapAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output


class _MemoryClient:
    def __init__(self):
        self.compile_calls = []
        self.query_calls = []

    def compile_repo(self, repo_id, files, *, metadata=None, revision=None, enable_w2=False):
        self.compile_calls.append(
            {
                "repo_id": repo_id,
                "files": files,
                "metadata": metadata,
                "revision": revision,
                "enable_w2": enable_w2,
            }
        )
        return {"repo_id": repo_id, "revision": "rev1", "parsed_file_count": len(files), "_latency_ms": 1}

    def query_repo(self, repo_id, query, *, metadata=None, revision=None, budget=4000):
        self.query_calls.append(
            {
                "repo_id": repo_id,
                "query": query,
                "metadata": metadata,
                "revision": revision,
                "budget": budget,
            }
        )
        return {"repo_id": repo_id, "revision": "rev1", "query": query, "extra_context": "### ctx", "_latency_ms": 2}


def test_memory_bootstrap_agent_injects_context_once(monkeypatch):
    client = _MemoryClient()
    monkeypatch.setattr(
        "minisweagent.agents.memory_bootstrap.extract_memory_source_files",
        lambda env, cwd="": [{"path": "pkg/mod.py", "content": "x=1", "language": "python", "is_test": False}],
    )

    agent = MemoryBootstrapAgent(
        model=DeterministicModel(
            outputs=[
                make_output(
                    "finish",
                    [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}],
                )
            ]
        ),
        env=LocalEnvironment(),
        system_template="sys",
        instance_template="task={{task}}",
        memory={"enabled": True, "query_budget": 1234},
        memory_client=client,
    )

    info = agent.run("issue text", instance_id="repo__1")

    assert info["exit_status"] == "Submitted"
    assert client.compile_calls == [
        {
            "repo_id": "repo__1",
            "files": [{"path": "pkg/mod.py", "content": "x=1", "language": "python", "is_test": False}],
            "metadata": {"instance_id": "repo__1"},
            "revision": None,
            "enable_w2": False,
        }
    ]
    assert client.query_calls == [
        {
            "repo_id": "repo__1",
            "query": "issue text",
            "metadata": {"instance_id": "repo__1"},
            "revision": "rev1",
            "budget": 1234,
        }
    ]
    assert agent.memory_state["bootstrapped"] is True
    assert [msg["role"] for msg in agent.messages[:4]] == ["system", "user", "assistant", "tool"]
    assert agent.messages[3]["content"] == "### ctx"


def test_memory_bootstrap_agent_aborts_on_query_failure(monkeypatch):
    class _FailingClient(_MemoryClient):
        def query_repo(self, repo_id, query, *, metadata=None, revision=None, budget=4000):
            raise RuntimeError("query failed")

    monkeypatch.setattr("minisweagent.agents.memory_bootstrap.extract_memory_source_files", lambda env, cwd="": [])

    agent = MemoryBootstrapAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        system_template="sys",
        instance_template="task={{task}}",
        memory={"enabled": True},
        memory_client=_FailingClient(),
    )

    with pytest.raises(RuntimeError, match="query failed"):
        agent.run("issue text", instance_id="repo__1")
