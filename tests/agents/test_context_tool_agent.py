import httpx
import pytest

from minisweagent.agents.context_tool_agent import ContextToolAgent, _read_path_candidates
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel


# -- _read_path_candidates (pure) -----------------------------------------


def test_read_path_candidates_strips_leading_slash_then_components():
    assert _read_path_candidates("/app/lib/x.py") == ["app/lib/x.py", "lib/x.py", "x.py"]
    assert _read_path_candidates("lib/x.py") == ["lib/x.py", "x.py"]
    assert _read_path_candidates("/x.py") == ["x.py"]
    assert _read_path_candidates("  /a/b/c.py  ") == ["a/b/c.py", "b/c.py", "c.py"]
    assert _read_path_candidates("") == []
    assert _read_path_candidates(None) == []  # type: ignore[arg-type]


# -- _execute_context_read (agent) ----------------------------------------


class _FakeReadResult:
    def __init__(self, content, truncated=False):
        self.content = content
        self.truncated = truncated


def _http_status_error(status: int) -> httpx.HTTPStatusError:
    req = httpx.Request("GET", "http://formsy/evidence/r/v/files/x")
    resp = httpx.Response(status, request=req)
    return httpx.HTTPStatusError(f"http {status}", request=req, response=resp)


class _ReadClient:
    """Fake formsy_sdk.Client.read_file: serves known paths, 404s the rest."""

    def __init__(self, existing: dict[str, str]):
        self.existing = existing
        self.calls: list[str] = []

    def read_file(self, repo_id, revision, path, *, start_line=1, end_line=None, max_lines=400):
        self.calls.append(path)
        if path in self.existing:
            return _FakeReadResult(self.existing[path])
        raise _http_status_error(404)


def _make_agent(client) -> ContextToolAgent:
    agent = ContextToolAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        system_template="sys",
        instance_template="task={{task}}",
        memory={"enabled": False},
        evidence_client=client,
    )
    # Simulate a successful ingest without actually calling the server.
    agent._ingest_ok = True
    agent._repo_id = "r"
    agent._revision = "v"
    return agent


def test_context_read_resolves_absolute_path_by_stripping_components():
    # Index has the repo-relative path; the LLM passed an absolute container path.
    client = _ReadClient({"lib/ansible/plugins/loader.py": "CONTENT"})
    agent = _make_agent(client)

    out = agent._execute_context_read({"path": "/app/lib/ansible/plugins/loader.py"})

    assert out["returncode"] == 0
    assert "CONTENT" in out["output"]
    # Tried the slash-stripped form first, then matched the repo-relative form.
    assert client.calls == ["app/lib/ansible/plugins/loader.py", "lib/ansible/plugins/loader.py"]
    # The matched relative path is echoed in the header.
    assert out["output"].startswith("lib/ansible/plugins/loader.py\n\n")


def test_context_read_matches_relative_path_first_try():
    client = _ReadClient({"lib/x.py": "OK"})
    agent = _make_agent(client)

    out = agent._execute_context_read({"path": "lib/x.py"})

    assert out["returncode"] == 0
    assert "OK" in out["output"]
    assert client.calls == ["lib/x.py"]  # no retries needed


def test_context_read_reports_actionable_error_when_not_found():
    client = _ReadClient({})  # nothing indexed
    agent = _make_agent(client)

    out = agent._execute_context_read({"path": "/app/missing.py"})

    assert out["returncode"] == 1
    assert "not found in Evidence index" in out["output"]
    assert "context_search" in out["output"]


def test_context_read_when_not_ingested_degrades():
    agent = _make_agent(_ReadClient({}))
    agent._ingest_ok = False

    out = agent._execute_context_read({"path": "lib/x.py"})

    assert out["returncode"] == 1
    assert "unavailable" in out["output"]
