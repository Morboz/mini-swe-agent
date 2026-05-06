from __future__ import annotations

from typing import Any

from minisweagent.agents.memory_query import MemoryQueryAgent
from minisweagent.models.test_models import DeterministicModel, make_output


class _Env:
    def __init__(self):
        self.actions = []
        self.config = type("Config", (), {"cwd": ""})()

    def execute(self, action: dict, cwd: str = "") -> dict[str, Any]:
        self.actions.append(action)
        return {"returncode": 0, "exception_info": "", "output": f"ran: {action['command']}"}

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return kwargs

    def serialize(self) -> dict:
        return {}


class _MemoryClient:
    def __init__(self):
        self.compile_calls = []
        self.query_calls = []

    def compile_repo(self, repo_id, files, *, metadata=None, revision=None, enable_w2=False):
        self.compile_calls.append({"repo_id": repo_id, "files": files, "revision": revision})
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
        return {
            "repo_id": repo_id,
            "revision": revision or "rev1",
            "query": query,
            "extra_context": f"ctx for {query}",
            "_latency_ms": 2,
        }


def _agent(command: str, monkeypatch) -> tuple[MemoryQueryAgent, _Env, _MemoryClient]:
    monkeypatch.setattr("minisweagent.agents.memory_bootstrap.extract_memory_source_files", lambda env, cwd="": [])
    env = _Env()
    client = _MemoryClient()
    agent = MemoryQueryAgent(
        model=DeterministicModel(outputs=[make_output("lookup", [{"command": command}])]),
        env=env,
        system_template="sys",
        instance_template="task={{task}}",
        memory={"enabled": True, "query_budget": 123},
        memory_client=client,
    )
    agent.extra_template_vars = {"task": "issue text", "instance_id": "repo__1"}
    agent.messages = [
        agent.model.format_message(role="system", content="sys"),
        agent.model.format_message(role="user", content="task=issue text"),
    ]
    return agent, env, client


def test_memory_query_agent_redirects_explicit_memory_search(monkeypatch):
    agent, env, client = _agent('memory_search "Parser parse src/parser.py"', monkeypatch)

    agent.step()

    assert env.actions == []
    assert [call["query"] for call in client.query_calls] == ["issue text", "Parser parse src/parser.py"]
    assert client.query_calls[-1]["revision"] == "rev1"
    assert agent.messages[-1]["extra"]["memory_redirected"] is True
    assert "ctx for Parser parse src/parser.py" in agent.messages[-1]["content"]


def test_memory_query_agent_redirects_source_content_lookup(monkeypatch):
    agent, env, client = _agent("rg Parser src", monkeypatch)

    agent.step()

    assert env.actions == []
    assert client.query_calls[-1]["query"] == "rg Parser src"
    assert agent.extra_template_vars["memory_info"]["redirect_count"] == 1


def test_memory_query_agent_allows_directory_exploration(monkeypatch):
    agent, env, client = _agent("find src -name '*.py'", monkeypatch)

    agent.step()

    assert env.actions == [{"command": "find src -name '*.py'"}]
    assert [call["query"] for call in client.query_calls] == ["issue text"]
    assert "ran: find src -name" in agent.messages[-1]["content"]


def test_memory_query_agent_allows_tree_and_filename_grep(monkeypatch):
    for command in ["tree src", "git ls-files | grep parser", "ls src | grep parser"]:
        agent, env, client = _agent(command, monkeypatch)

        agent.step()

        assert env.actions == [{"command": command}]
        assert [call["query"] for call in client.query_calls] == ["issue text"]


def test_memory_query_agent_allows_patch_and_submission_reads(monkeypatch):
    for command in ["cat patch.txt", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"]:
        agent, env, client = _agent(command, monkeypatch)

        agent.step()

        assert env.actions == [{"command": command}]
        assert [call["query"] for call in client.query_calls] == ["issue text"]


def test_memory_query_agent_redirects_source_file_reads(monkeypatch):
    for command in ["cat src/package/module.py", "sed -n '1,120p' src/package/module.py"]:
        agent, env, client = _agent(command, monkeypatch)

        agent.step()

        assert env.actions == []
        assert client.query_calls[-1]["query"] == command


def test_memory_query_agent_logs_step_progress(monkeypatch, caplog):
    agent, _, _ = _agent("find src -name '*.py'", monkeypatch)

    agent.step()

    assert "Step   1 ($0.00)" in caplog.text
