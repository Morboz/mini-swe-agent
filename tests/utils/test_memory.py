import pytest

from minisweagent.utils.memory import (
    MemoryClient,
    MemoryExtractError,
    MemoryQueryError,
    extract_memory_source_files,
)


class _DummyResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._payload


def test_memory_client_compile_returns_validated_payload():
    calls = []

    class _DummySession:
        def post(self, url, json, timeout, headers):
            calls.append((url, json, timeout, headers))
            return _DummyResponse({"repo_id": "repo", "revision": "rev1", "parsed_file_count": 2})

    client = MemoryClient(base_url="http://memory", timeout_seconds=5, session=_DummySession())
    result = client.compile_repo("repo", [{"path": "a.py", "content": "print(1)"}], metadata={"instance_id": "x"})

    assert result["repo_id"] == "repo"
    assert result["parsed_file_count"] == 2
    assert result["_latency_ms"] >= 0
    assert calls == [
        (
            "http://memory/api/v1/compile",
            {
                "repo_id": "repo",
                "files": [{"path": "a.py", "content": "print(1)"}],
                "revision": None,
                "enable_w2": False,
                "metadata": {"instance_id": "x"},
            },
            5,
            {},
        )
    ]


def test_memory_client_query_raises_on_invalid_payload():
    class _DummySession:
        def post(self, url, json, timeout, headers):
            return _DummyResponse({"repo_id": "repo", "revision": "rev1", "query": "issue text"}, status_code=200)

    client = MemoryClient(base_url="http://memory", timeout_seconds=5, session=_DummySession())

    with pytest.raises(MemoryQueryError, match="extra_context"):
        client.query_repo("repo", "issue text", metadata={})


class _DummyEnv:
    def __init__(self, payload, returncode=0):
        self.payload = payload
        self.returncode = returncode
        self.calls = []

    def execute(self, action, cwd=""):
        self.calls.append((action, cwd))
        return {"returncode": self.returncode, "output": self.payload, "exception_info": ""}


def test_extract_memory_source_files_marks_tests_and_preserves_relative_paths():
    payload = (
        '[{"path":"pkg/mod.py","content":"x=1","language":"python","is_test":false},'
        '{"path":"tests/test_mod.py","content":"def test_x(): pass","language":"python","is_test":true}]'
    )
    env = _DummyEnv(payload)

    files = extract_memory_source_files(env, cwd="/testbed")

    assert [f["path"] for f in files] == ["pkg/mod.py", "tests/test_mod.py"]
    assert files[0]["is_test"] is False
    assert files[1]["is_test"] is True
    assert env.calls[0][1] == "/testbed"
    assert "python - <<'PY'" in env.calls[0][0]["command"]


def test_extract_memory_source_files_raises_on_command_failure():
    env = _DummyEnv("boom", returncode=1)

    with pytest.raises(MemoryExtractError, match="boom"):
        extract_memory_source_files(env, cwd="/testbed")
