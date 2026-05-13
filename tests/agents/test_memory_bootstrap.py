import pytest

from minisweagent.agents.memory_bootstrap import MemoryBootstrapAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, DeterministicToolcallModel, make_output, make_toolcall_output


class _MemoryClient:
    def __init__(self):
        self.compile_calls = []
        self.query_calls = []
        self.query_results = []

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
        if self.query_results:
            return self.query_results.pop(0)
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
    monkeypatch.setattr(
        "minisweagent.agents.memory_bootstrap.extract_memory_source_files",
        lambda env, cwd="": [{"path": "pkg/mod.py", "content": "x=1", "language": "python", "is_test": False}],
    )

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
    assert client.compile_calls == [
        {
            "repo_id": "repo__1",
            "files": [{"path": "pkg/mod.py", "content": "x=1", "language": "python", "is_test": False}],
            "metadata": {"instance_id": "repo__1"},
            "revision": "latest",
            "enable_w2": False,
        }
    ]
    assert client.query_calls == [
        {
            "repo_id": "repo__1",
            "query": "find target",
            "metadata": {
                "instance_id": "repo__1",
                "case_id": "repo__1",
                "retrieval_mode": "symbolic",
                "grounding_phase": "seed",
                "response_format": "bundle",
            },
            "revision": "rev1",
            "budget": 1234,
        }
    ]
    assert agent.memory_state["tool_calls"] == {"context_search": 1, "context_read": 1}
    observations = [msg for msg in agent.messages if msg.get("role") == "tool"]
    assert "### ctx" in observations[0]["extra"]["raw_output"]
    assert "def target()" in observations[1]["extra"]["raw_output"]


def test_memory_bootstrap_agent_reuses_compile_revision_for_multiple_searches(monkeypatch):
    client = _MemoryClient()
    monkeypatch.setattr("minisweagent.agents.memory_bootstrap.extract_memory_source_files", lambda env, cwd="": [])

    agent = MemoryBootstrapAgent(
        model=DeterministicToolcallModel(
            outputs=[
                make_toolcall_output(
                    None,
                    [
                        {
                            "id": "call_search_1",
                            "type": "function",
                            "function": {"name": "context_search", "arguments": '{"query": "first"}'},
                        },
                        {
                            "id": "call_search_2",
                            "type": "function",
                            "function": {"name": "context_search", "arguments": '{"query": "second"}'},
                        },
                    ],
                    [
                        {"tool": "context_search", "query": "first", "tool_call_id": "call_search_1"},
                        {"tool": "context_search", "query": "second", "tool_call_id": "call_search_2"},
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
                ),
            ]
        ),
        env=LocalEnvironment(),
        system_template="sys",
        instance_template="task={{task}}",
        cost_limit=0,
        memory={"enabled": True},
        memory_client=client,
    )

    agent.run("issue text", instance_id="repo__1")

    assert len(client.compile_calls) == 1
    assert client.compile_calls[0]["revision"] == "latest"
    assert [call["revision"] for call in client.query_calls] == ["rev1", "rev1"]


def test_memory_bootstrap_agent_defaults_context_search_to_symbolic_seed_metadata(monkeypatch):
    client = _MemoryClient()
    monkeypatch.setattr("minisweagent.agents.memory_bootstrap.extract_memory_source_files", lambda env, cwd="": [])

    agent = MemoryBootstrapAgent(
        model=DeterministicToolcallModel(
            outputs=[
                make_toolcall_output(
                    "search",
                    [
                        {
                            "id": "call_search",
                            "type": "function",
                            "function": {"name": "context_search", "arguments": '{"query": "find target"}'},
                        }
                    ],
                    [{"tool": "context_search", "query": "find target", "tool_call_id": "call_search"}],
                )
            ]
        ),
        env=LocalEnvironment(),
        system_template="sys",
        instance_template="task={{task}}",
        cost_limit=0,
        memory={"enabled": True},
        memory_client=client,
    )

    agent.extra_template_vars |= {"task": "issue text", "instance_id": "repo__1"}
    agent.messages = [
        agent.model.format_message(role="system", content="sys"),
        agent.model.format_message(role="user", content="task=issue text"),
    ]
    agent.step()

    assert client.query_calls[0]["metadata"] == {
        "instance_id": "repo__1",
        "case_id": "repo__1",
        "retrieval_mode": "symbolic",
        "grounding_phase": "seed",
        "response_format": "bundle",
    }


def test_memory_bootstrap_agent_blocks_bash_until_initial_context_search():
    agent = MemoryBootstrapAgent(
        model=DeterministicToolcallModel(
            outputs=[
                make_toolcall_output(
                    "try bash",
                    [
                        {
                            "id": "call_bash",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                        }
                    ],
                    [{"command": "ls", "tool_call_id": "call_bash"}],
                )
            ]
        ),
        env=LocalEnvironment(),
        system_template="sys",
        instance_template="task={{task}}",
        memory={"enabled": True},
    )

    agent.extra_template_vars |= {"task": "issue text", "instance_id": "repo__1"}
    agent.messages = [
        agent.model.format_message(role="system", content="sys"),
        agent.model.format_message(role="user", content="task=issue text"),
    ]
    agent.step()

    observation = [msg for msg in agent.messages if msg.get("role") == "tool"][0]
    assert "Retrieval gate active: call context_search first" in observation["extra"]["raw_output"]


def test_memory_bootstrap_agent_requires_read_then_grounded_search_before_editing(monkeypatch):
    client = _MemoryClient()
    client.query_results = [
        {
            "repo_id": "repo__1",
            "revision": "rev1",
            "query": "seed",
            "extra_context": "Seed notes",
            "matches": [{"path": "pkg/mod.py"}],
            "coverage": "partial",
            "_latency_ms": 2,
        },
        {
            "repo_id": "repo__1",
            "revision": "rev1",
            "query": "grounded",
            "extra_context": "Grounded notes",
            "matches": [{"path": "pkg/mod.py"}],
            "coverage": "good",
            "grounded_files": ["pkg/mod.py"],
            "_latency_ms": 2,
        },
    ]
    monkeypatch.setattr("minisweagent.agents.memory_bootstrap.extract_memory_source_files", lambda env, cwd="": [])

    agent = MemoryBootstrapAgent(
        model=DeterministicToolcallModel(
            outputs=[
                make_toolcall_output(
                    "seed",
                    [
                        {
                            "id": "seed",
                            "type": "function",
                            "function": {"name": "context_search", "arguments": '{"query": "seed"}'},
                        }
                    ],
                    [{"tool": "context_search", "query": "seed", "tool_call_id": "seed"}],
                ),
                make_toolcall_output(
                    "blocked edit",
                    [
                        {
                            "id": "edit1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command": "python - <<PY\\nopen(\\\"pkg/mod.py\\\", \\\"w\\\").write(\\\"x\\\")\\nPY"}'},
                        }
                    ],
                    [
                        {
                            "command": 'python - <<PY\nopen("pkg/mod.py", "w").write("x")\nPY',
                            "tool_call_id": "edit1",
                        }
                    ],
                ),
                make_toolcall_output(
                    "read",
                    [
                        {
                            "id": "read",
                            "type": "function",
                            "function": {"name": "context_read", "arguments": '{"path": "pkg/mod.py"}'},
                        }
                    ],
                    [{"tool": "context_read", "path": "pkg/mod.py", "tool_call_id": "read"}],
                ),
                make_toolcall_output(
                    "blocked edit again",
                    [
                        {
                            "id": "edit2",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command": "python - <<PY\\nopen(\\\"pkg/mod.py\\\", \\\"w\\\").write(\\\"x\\\")\\nPY"}'},
                        }
                    ],
                    [
                        {
                            "command": 'python - <<PY\nopen("pkg/mod.py", "w").write("x")\nPY',
                            "tool_call_id": "edit2",
                        }
                    ],
                ),
                make_toolcall_output(
                    "grounded",
                    [
                        {
                            "id": "grounded",
                            "type": "function",
                            "function": {
                                "name": "context_search",
                                "arguments": '{"query": "grounded", "metadata": {"grounding_phase": "grounded"}}',
                            },
                        }
                    ],
                    [
                        {
                            "tool": "context_search",
                            "query": "grounded",
                            "metadata": {"grounding_phase": "grounded"},
                            "tool_call_id": "grounded",
                        }
                    ],
                ),
                make_toolcall_output(
                    "blocked broad search",
                    [
                        {
                            "id": "grep",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command": "grep -R target ."}'},
                        }
                    ],
                    [{"command": "grep -R target .", "tool_call_id": "grep"}],
                ),
                make_toolcall_output(
                    "finish",
                    [
                        {
                            "id": "submit",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\necho done"}',
                            },
                        }
                    ],
                    [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\necho done", "tool_call_id": "submit"}],
                ),
            ]
        ),
        env=LocalEnvironment(),
        system_template="sys",
        instance_template="task={{task}}",
        cost_limit=0,
        memory={"enabled": True},
        memory_client=client,
    )

    info = agent.run("issue text", instance_id="repo__1")

    assert info["exit_status"] == "Submitted"
    raw_outputs = [msg["extra"]["raw_output"] for msg in agent.messages if msg.get("role") == "tool"]
    assert "use context_read on a candidate file" in raw_outputs[1]
    assert "requires a grounded context_search" in raw_outputs[3]
    assert "broad grep/find/search commands are blocked" in raw_outputs[5]
    assert agent.memory_state["retrieval_state"] == "grounded"
    assert agent.memory_state["accepted_targets"] == ["pkg/mod.py"]


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
