import pytest

from minisweagent.agents.memory_bootstrap import MemoryBootstrapAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, DeterministicToolcallModel, make_output, make_toolcall_output


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

    def read_repo(self, repo_id, path, *, revision=None, start_line=None, end_line=None, metadata=None):
        return {
            "repo_id": repo_id,
            "revision": revision or "rev1",
            "path": path,
            "content": "def target():\n    return 1\n",
            "start_line": start_line,
            "end_line": end_line,
            "total_lines": 2,
            "truncated": False,
            "_latency_ms": 3,
        }


def test_memory_bootstrap_agent_executes_context_tools_without_startup_bootstrap(monkeypatch):
    client = _MemoryClient()

    agent = MemoryBootstrapAgent(
        model=DeterministicToolcallModel(
            outputs=[
                make_toolcall_output(
                    None,
                    [
                        {
                            "id": "call_search",
                            "type": "function",
                            "function": {
                                "name": "context_search",
                                "arguments": '{"query": "find target", "budget": 1234}',
                            },
                        }
                    ],
                    [{"tool": "context_search", "query": "find target", "budget": 1234, "tool_call_id": "call_search"}],
                ),
                make_toolcall_output(
                    None,
                    [
                        {
                            "id": "call_read",
                            "type": "function",
                            "function": {
                                "name": "context_read",
                                "arguments": '{"path": "pkg/mod.py", "start_line": 1, "end_line": 2}',
                            },
                        }
                    ],
                    [
                        {
                            "tool": "context_read",
                            "path": "pkg/mod.py",
                            "start_line": 1,
                            "end_line": 2,
                            "tool_call_id": "call_read",
                        }
                    ],
                ),
                make_toolcall_output(
                    "finish",
                    [
                        {
                            "id": "call_submit",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\necho done"}',
                            },
                        }
                    ],
                    [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\necho done", "tool_call_id": "call_submit"}],
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
    assert client.compile_calls == []
    assert client.query_calls == [
        {
            "repo_id": "repo__1",
            "query": "find target",
            "metadata": {"instance_id": "repo__1"},
            "revision": None,
            "budget": 1234,
        }
    ]
    assert agent.memory_state["tool_calls"] == {"context_search": 1, "context_read": 1}
    observations = [msg for msg in agent.messages if msg.get("role") == "tool"]
    assert "### ctx" in observations[0]["extra"]["raw_output"]
    assert "def target()" in observations[1]["extra"]["raw_output"]


def test_memory_bootstrap_agent_aborts_on_query_failure(monkeypatch):
    class _FailingClient(_MemoryClient):
        def query_repo(self, repo_id, query, *, metadata=None, revision=None, budget=4000):
            raise RuntimeError("query failed")

    agent = MemoryBootstrapAgent(
        model=DeterministicToolcallModel(
            outputs=[
                make_toolcall_output(
                    None,
                    [
                        {
                            "id": "call_search",
                            "type": "function",
                            "function": {
                                "name": "context_search",
                                "arguments": '{"query": "find target"}',
                            },
                        }
                    ],
                    [{"tool": "context_search", "query": "find target", "tool_call_id": "call_search"}],
                )
            ]
        ),
        env=LocalEnvironment(),
        system_template="sys",
        instance_template="task={{task}}",
        memory={"enabled": True},
        memory_client=_FailingClient(),
    )

    with pytest.raises(RuntimeError, match="query failed"):
        agent.run("issue text", instance_id="repo__1")


def test_memory_bootstrap_agent_adds_context_tool_schemas_to_tool_models():
    class _Model:
        def __init__(self):
            self.tools = []

        def query(self, messages):
            return make_output("done", [])

        def format_message(self, **kwargs):
            return kwargs

        def format_observation_messages(self, message, outputs, template_vars=None):
            return []

        def get_template_vars(self):
            return {}

        def serialize(self):
            return {}

    model = _Model()
    MemoryBootstrapAgent(
        model=model,
        env=LocalEnvironment(),
        system_template="sys",
        instance_template="task={{task}}",
        memory={"enabled": True},
    )

    assert [tool["function"]["name"] for tool in model.tools] == ["context_search", "context_read"]


def test_memory_bootstrap_agent_adds_response_api_context_tool_schemas():
    class _Model:
        def __init__(self):
            self.tools = [{"type": "function", "name": "bash", "parameters": {}}]

        def query(self, messages):
            return make_output("done", [])

        def format_message(self, **kwargs):
            return kwargs

        def format_observation_messages(self, message, outputs, template_vars=None):
            return []

        def get_template_vars(self):
            return {}

        def serialize(self):
            return {}

    model = _Model()
    MemoryBootstrapAgent(
        model=model,
        env=LocalEnvironment(),
        system_template="sys",
        instance_template="task={{task}}",
        memory={"enabled": True},
    )

    assert [tool["name"] for tool in model.tools] == ["bash", "context_search", "context_read"]
