import pytest

from minisweagent._vendor.formsy_sdk import SourceFile
from minisweagent.utils import evidence
from minisweagent.utils.evidence import (
    FormsyEvidenceError,
    extract_source_files,
    resolve_revision,
)


@pytest.fixture(autouse=True)
def _clear_revision_cache():
    """The revision cache is module-global; isolate it between tests."""
    evidence._REVISION_CACHE.clear()
    yield
    evidence._REVISION_CACHE.clear()


class _DummyEnv:
    def __init__(self, output, returncode=0):
        self.output = output
        self.returncode = returncode
        self.calls = []

    def execute(self, action, cwd=""):
        self.calls.append((action, cwd))
        return {"returncode": self.returncode, "output": self.output, "exception_info": ""}


def test_extract_source_files_returns_sourcefiles():
    payload = (
        '[{"path":"pkg/mod.py","content":"x=1"},'
        '{"path":"tests/test_mod.py","content":"def test_x(): pass"}]'
    )
    env = _DummyEnv(payload)

    files = extract_source_files(env, cwd="/testbed")

    assert all(isinstance(f, SourceFile) for f in files)
    assert [f.path for f in files] == ["pkg/mod.py", "tests/test_mod.py"]
    assert files[0].content == "x=1"
    assert env.calls[0][1] == "/testbed"
    assert "python - <<'PY'" in env.calls[0][0]["command"]


def test_extract_source_files_raises_on_command_failure():
    env = _DummyEnv("boom", returncode=1)
    with pytest.raises(FormsyEvidenceError, match="boom"):
        extract_source_files(env, cwd="/testbed")


def test_resolve_revision_parses_git_sha():
    env = _DummyEnv("abc123def456\n")
    assert resolve_revision(env, cwd="/testbed") == "abc123def456"
    assert env.calls[0][0]["command"] == "git rev-parse HEAD"


def test_resolve_revision_caches_per_cwd():
    env = _DummyEnv("abc123\n")
    assert resolve_revision(env, cwd="/testbed") == "abc123"
    assert len(env.calls) == 1
    # Second call hits the cache — no further env execution.
    assert resolve_revision(env, cwd="/testbed") == "abc123"
    assert len(env.calls) == 1


def test_resolve_revision_raises_on_failure():
    env = _DummyEnv("not a repo", returncode=1)
    with pytest.raises(FormsyEvidenceError, match="resolve_revision"):
        resolve_revision(env, cwd="/testbed")
