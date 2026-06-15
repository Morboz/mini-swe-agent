# Memory SWE-Bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in memory bootstrap path for SWE-Bench runs that compiles repository source into an external backend, performs one startup query, injects the returned markdown as synthetic tool context, aborts the case on memory bootstrap failure, and records token usage for comparison.

**Architecture:** Add a small memory integration layer with a backend client and repository extractor, then wire it into a new agent class that bootstraps exactly once before the first model query. Keep runner outputs and patch submission unchanged, while enriching trajectories with memory metadata and aggregated token usage.

**Tech Stack:** Python, pytest, pydantic, requests/httpx-compatible HTTP client, existing mini-SWE-agent agent/model/environment abstractions

---

### Task 1: Add failing tests for memory client and repo extractor

**Files:**
- Create: `tests/agents/test_memory_bootstrap.py`
- Create: `tests/utils/test_memory.py`
- Create: `src/minisweagent/utils/memory.py`

- [ ] **Step 1: Write the failing tests for backend request/response handling**

```python
import pytest

from minisweagent.utils.memory import MemoryClient, MemoryCompileError, MemoryQueryError


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


def test_memory_client_compile_returns_validated_payload(monkeypatch):
    calls = []

    class _DummySession:
        def post(self, url, json, timeout, headers):
            calls.append((url, json, timeout, headers))
            return _DummyResponse({"repo_id": "repo", "revision": "rev1", "parsed_file_count": 2})

    client = MemoryClient(base_url="http://memory", timeout_seconds=5, session=_DummySession())
    result = client.compile_repo("repo", [{"path": "a.py", "content": "print(1)"}], metadata={"instance_id": "x"})

    assert result["repo_id"] == "repo"
    assert result["parsed_file_count"] == 2
    assert calls[0][0] == "http://memory/api/v1/compile"


def test_memory_client_query_raises_on_invalid_payload():
    class _DummySession:
        def post(self, url, json, timeout, headers):
            return _DummyResponse({"repo_id": "repo", "revision": "rev1"}, status_code=200)

    client = MemoryClient(base_url="http://memory", timeout_seconds=5, session=_DummySession())

    with pytest.raises(MemoryQueryError):
        client.query_repo("repo", "issue text", metadata={})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/utils/test_memory.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing `MemoryClient`

- [ ] **Step 3: Write the failing tests for repository extraction filtering**

```python
from minisweagent.utils.memory import extract_memory_source_files


class _DummyEnv:
    def __init__(self, payload):
        self.payload = payload
        self.commands = []

    def execute(self, action, cwd=""):
        self.commands.append((action, cwd))
        return {"returncode": 0, "output": self.payload, "exception_info": ""}


def test_extract_memory_source_files_marks_tests_and_preserves_relative_paths():
    payload = (
        '[{"path":"pkg/mod.py","content":"x=1","language":"python","is_test":false},'
        '{"path":"tests/test_mod.py","content":"def test_x(): pass","language":"python","is_test":true}]'
    )
    env = _DummyEnv(payload)

    files = extract_memory_source_files(env, cwd="/testbed")

    assert [f["path"] for f in files] == ["pkg/mod.py", "tests/test_mod.py"]
    assert files[1]["is_test"] is True
    assert env.commands
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/utils/test_memory.py -v`
Expected: FAIL with missing `extract_memory_source_files`

- [ ] **Step 5: Commit**

```bash
git add tests/utils/test_memory.py tests/agents/test_memory_bootstrap.py src/minisweagent/utils/memory.py
git commit -m "test: add memory client and extractor coverage"
```

### Task 2: Implement memory backend client and repository extractor

**Files:**
- Modify: `src/minisweagent/utils/memory.py`
- Test: `tests/utils/test_memory.py`

- [ ] **Step 1: Write the minimal memory client and exceptions**

```python
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import requests


class MemoryError(RuntimeError):
    pass


class MemoryCompileError(MemoryError):
    pass


class MemoryQueryError(MemoryError):
    pass


@dataclass
class MemoryCallResult:
    payload: dict
    latency_ms: int


class MemoryClient:
    def __init__(self, *, base_url: str, timeout_seconds: int, session=None, headers=None):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self.headers = headers or {}

    def compile_repo(self, repo_id: str, files: list[dict], metadata: dict | None = None, enable_w2: bool = False) -> dict:
        payload = {"repo_id": repo_id, "files": files, "enable_w2": enable_w2, "metadata": metadata or {}}
        result = self._post("/api/v1/compile", payload, MemoryCompileError)
        for key in ("repo_id", "revision", "parsed_file_count"):
            if key not in result.payload:
                raise MemoryCompileError(f"Missing compile field: {key}")
        return result.payload | {"_latency_ms": result.latency_ms}

    def query_repo(self, repo_id: str, query: str, metadata: dict | None = None, budget: int = 4000) -> dict:
        payload = {"repo_id": repo_id, "query": query, "metadata": metadata or {}, "budget": budget}
        result = self._post("/api/v1/query", payload, MemoryQueryError)
        for key in ("repo_id", "revision", "query"):
            if key not in result.payload:
                raise MemoryQueryError(f"Missing query field: {key}")
        if not isinstance(result.payload.get("extra_context", ""), str):
            raise MemoryQueryError("Query extra_context must be a string")
        return result.payload | {"_latency_ms": result.latency_ms}

    def _post(self, path: str, payload: dict, error_cls):
        started = time.time()
        try:
            response = self.session.post(
                f"{self.base_url}{path}", json=payload, timeout=self.timeout_seconds, headers=self.headers
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise error_cls(str(exc)) from exc
        return MemoryCallResult(payload=data, latency_ms=int((time.time() - started) * 1000))
```

- [ ] **Step 2: Write the minimal repository extractor**

```python
def extract_memory_source_files(env, cwd: str = "") -> list[dict]:
    command = r"""
python - <<'PY'
import json
from pathlib import Path

root = Path('.').resolve()
allowed = {'.py', '.js', '.ts', '.tsx', '.java', '.go', '.rb', '.rs', '.cpp', '.c', '.h'}
excluded = {'.git', 'node_modules', 'vendor', 'dist', 'build', '.venv', '__pycache__'}
files = []
for path in root.rglob('*'):
    if not path.is_file():
        continue
    if any(part in excluded for part in path.parts):
        continue
    if path.suffix.lower() not in allowed:
        continue
    rel = path.relative_to(root).as_posix()
    content = path.read_text(encoding='utf-8', errors='ignore')
    files.append({
        'path': rel,
        'content': content,
        'language': path.suffix.lower().lstrip('.') or 'text',
        'is_test': '/tests/' in f'/{rel}/' or rel.startswith('tests/') or rel.endswith('_test.py') or rel.startswith('test/'),
    })
print(json.dumps(files))
PY
""".strip()
    result = env.execute({"command": command}, cwd=cwd)
    if result["returncode"] != 0:
        raise MemoryError(result.get("output", "extract failed"))
    return json.loads(result["output"])
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `pytest tests/utils/test_memory.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/minisweagent/utils/memory.py tests/utils/test_memory.py
git commit -m "feat: add memory backend client and extractor"
```

### Task 3: Add failing tests for one-time agent bootstrap and failure behavior

**Files:**
- Modify: `tests/agents/test_memory_bootstrap.py`
- Create: `src/minisweagent/agents/memory_bootstrap.py`

- [ ] **Step 1: Write the failing tests**

```python
from minisweagent.agents.memory_bootstrap import MemoryBootstrapAgent
from minisweagent.models.test_models import DeterministicToolcallModel, make_toolcall_output


class _Env:
    def __init__(self):
        self.commands = []

    def execute(self, action, cwd=""):
        self.commands.append((action, cwd))
        return {"returncode": 0, "output": "ok", "exception_info": ""}

    def get_template_vars(self, **kwargs):
        return {}

    def serialize(self):
        return {}


class _MemoryClient:
    def __init__(self):
        self.compile_calls = []
        self.query_calls = []

    def compile_repo(self, repo_id, files, metadata=None, enable_w2=False):
        self.compile_calls.append((repo_id, files, metadata, enable_w2))
        return {"repo_id": repo_id, "revision": "rev1", "parsed_file_count": len(files), "_latency_ms": 1}

    def query_repo(self, repo_id, query, metadata=None, budget=4000):
        self.query_calls.append((repo_id, query, metadata, budget))
        return {"repo_id": repo_id, "revision": "rev1", "query": query, "extra_context": "### ctx", "_latency_ms": 2}


def test_memory_bootstrap_agent_injects_context_once(monkeypatch):
    monkeypatch.setattr("minisweagent.agents.memory_bootstrap.extract_memory_source_files", lambda env, cwd='': [
        {"path": "pkg/mod.py", "content": "x=1", "language": "py", "is_test": False}
    ])
    model = DeterministicToolcallModel(outputs=[make_toolcall_output(None, [], [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && printf diff"}])])
    env = _Env()
    agent = MemoryBootstrapAgent(model, env, system_template="sys", instance_template="user", memory={"enabled": True}, memory_client=_MemoryClient())
    agent.run("issue text", instance_id="repo__1")

    assert len(agent.messages) >= 4
    assert sum(1 for m in agent.messages if "### ctx" in str(m)) >= 1
    assert agent.memory_state["bootstrapped"] is True


def test_memory_bootstrap_agent_aborts_on_query_failure(monkeypatch):
    class _FailingClient(_MemoryClient):
        def query_repo(self, repo_id, query, metadata=None, budget=4000):
            raise RuntimeError("query failed")

    monkeypatch.setattr("minisweagent.agents.memory_bootstrap.extract_memory_source_files", lambda env, cwd='': [])
    model = DeterministicToolcallModel(outputs=[])
    env = _Env()
    agent = MemoryBootstrapAgent(model, env, system_template="sys", instance_template="user", memory={"enabled": True}, memory_client=_FailingClient())

    with pytest.raises(RuntimeError):
        agent.run("issue text", instance_id="repo__1")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/agents/test_memory_bootstrap.py -v`
Expected: FAIL with missing `MemoryBootstrapAgent`

- [ ] **Step 3: Commit**

```bash
git add tests/agents/test_memory_bootstrap.py src/minisweagent/agents/memory_bootstrap.py
git commit -m "test: add memory bootstrap agent coverage"
```

### Task 4: Implement memory bootstrap agent and runner wiring

**Files:**
- Modify: `src/minisweagent/agents/__init__.py`
- Create: `src/minisweagent/agents/memory_bootstrap.py`
- Modify: `src/minisweagent/agents/default.py`
- Modify: `src/minisweagent/run/benchmarks/swebench.py`
- Modify: `src/minisweagent/run/benchmarks/swebench_single.py`
- Modify: `src/minisweagent/config/benchmarks/swebench.yaml`
- Test: `tests/agents/test_memory_bootstrap.py`
- Test: `tests/run/test_swebench.py`

- [ ] **Step 1: Write the minimal agent implementation**

```python
from __future__ import annotations

from minisweagent.agents.default import DefaultAgent
from minisweagent.utils.memory import MemoryClient, MemoryCompileError, MemoryError, MemoryQueryError, extract_memory_source_files


class MemoryBootstrapAgent(DefaultAgent):
    def __init__(self, *args, memory=None, memory_client=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_config = memory or {}
        self.memory_client = memory_client
        self.memory_state = {"bootstrapped": False}

    def query(self) -> dict:
        if self.memory_config.get("enabled") and not self.memory_state["bootstrapped"]:
            self._bootstrap_memory()
        return super().query()

    def _bootstrap_memory(self) -> None:
        client = self.memory_client or MemoryClient(
            base_url=self.memory_config["base_url"],
            timeout_seconds=self.memory_config.get("timeout_seconds", 30),
        )
        repo_id = self.extra_template_vars.get("instance_id", "")
        files = extract_memory_source_files(self.env, cwd=getattr(self.env.config, "cwd", ""))
        compile_result = client.compile_repo(
            repo_id,
            files,
            metadata={"instance_id": repo_id},
            enable_w2=self.memory_config.get("enable_w2", False),
        )
        query_result = client.query_repo(
            repo_id,
            self.extra_template_vars["task"],
            metadata={"instance_id": repo_id},
            budget=self.memory_config.get("query_budget", 4000),
        )
        self.add_messages(
            {"role": "assistant", "content": None, "tool_calls": [{"id": "memory_bootstrap", "type": "function", "function": {"name": "memory_search", "arguments": '{"query":"startup"}'}}]},
            {"role": "tool", "tool_call_id": "memory_bootstrap", "content": query_result.get("extra_context", "")},
        )
        self.extra_template_vars["memory_info"] = {
            "enabled": True,
            "repo_id": repo_id,
            "compile_success": True,
            "query_success": True,
            "compile_latency_ms": compile_result["_latency_ms"],
            "query_latency_ms": query_result["_latency_ms"],
            "source_file_count": len(files),
            "source_total_bytes": sum(len(f["content"]) for f in files),
            "context_chars": len(query_result.get("extra_context", "")),
        }
        self.memory_state["bootstrapped"] = True
```

- [ ] **Step 2: Wire the agent into agent lookup and benchmark config**

```python
_AGENT_MAPPING = {
    "default": "minisweagent.agents.default.DefaultAgent",
    "interactive": "minisweagent.agents.interactive.InteractiveAgent",
    "memory_bootstrap": "minisweagent.agents.memory_bootstrap.MemoryBootstrapAgent",
}
```

```yaml
memory:
  enabled: false
  timeout_seconds: 30
  query_budget: 4000
  enable_w2: false
  allowed_extensions: [".py", ".js", ".ts", ".tsx", ".java", ".go", ".rb", ".rs", ".cpp", ".c", ".h"]
  excluded_dirs: [".git", "node_modules", "vendor", "dist", "build", ".venv", "__pycache__"]
```

- [ ] **Step 3: Extend benchmark runners to pass `instance_id` into `agent.run(...)`**

```python
agent.run(instance["problem_statement"], instance_id=instance["instance_id"])
```

- [ ] **Step 4: Run focused tests to verify they pass**

Run: `pytest tests/agents/test_memory_bootstrap.py tests/run/test_swebench.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/minisweagent/agents/__init__.py src/minisweagent/agents/memory_bootstrap.py src/minisweagent/agents/default.py src/minisweagent/run/benchmarks/swebench.py src/minisweagent/run/benchmarks/swebench_single.py src/minisweagent/config/benchmarks/swebench.yaml tests/agents/test_memory_bootstrap.py tests/run/test_swebench.py
git commit -m "feat: add memory bootstrap swebench agent"
```

### Task 5: Add failing tests for token aggregation and saved memory metadata

**Files:**
- Modify: `tests/agents/test_default.py`
- Modify: `tests/run/test_swebench.py`
- Modify: `src/minisweagent/agents/default.py`
- Modify: `src/minisweagent/models/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_agent_serialize_includes_usage_totals():
    ...
    assert data["info"]["model_stats"]["prompt_tokens"] == 10
    assert data["info"]["model_stats"]["completion_tokens"] == 4
    assert data["info"]["model_stats"]["total_tokens"] == 14


def test_swebench_traj_includes_memory_info(...):
    ...
    assert traj["info"]["memory"]["enabled"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/agents/test_default.py tests/run/test_swebench.py -v`
Expected: FAIL with missing usage totals and memory info

- [ ] **Step 3: Commit**

```bash
git add tests/agents/test_default.py tests/run/test_swebench.py src/minisweagent/agents/default.py src/minisweagent/models/__init__.py
git commit -m "test: add token aggregation and memory metadata coverage"
```

### Task 6: Implement token aggregation and trajectory metadata

**Files:**
- Modify: `src/minisweagent/agents/default.py`
- Modify: `src/minisweagent/models/__init__.py`
- Modify: `tests/agents/test_default.py`
- Modify: `tests/run/test_swebench.py`

- [ ] **Step 1: Add per-agent usage counters**

```python
self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
...
usage = message.get("extra", {}).get("usage", {})
for key in self.usage:
    self.usage[key] += int(usage.get(key, 0) or 0)
```

- [ ] **Step 2: Include usage and memory metadata in serialization**

```python
"model_stats": {
    "instance_cost": self.cost,
    "api_calls": self.n_calls,
    **self.usage,
},
...
return recursive_merge(agent_data, ..., {"info": {"memory": self.extra_template_vars.get("memory_info", {"enabled": False})}}, *extra_dicts)
```

- [ ] **Step 3: Populate `extra.usage` in model responses when provider data is available**

```python
def _normalize_usage(data: dict | None) -> dict[str, int]:
    data = data or {}
    return {
        "prompt_tokens": int(data.get("prompt_tokens") or data.get("input_tokens") or 0),
        "completion_tokens": int(data.get("completion_tokens") or data.get("output_tokens") or 0),
        "total_tokens": int(data.get("total_tokens") or 0),
    }
```

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/agents/test_default.py tests/agents/test_memory_bootstrap.py tests/utils/test_memory.py tests/run/test_swebench.py -v`
Expected: PASS

- [ ] **Step 5: Run broader verification**

Run: `pytest tests/run/test_swebench.py tests/agents/test_default.py tests/models/test_litellm_model.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/minisweagent/agents/default.py src/minisweagent/models/__init__.py tests/agents/test_default.py tests/run/test_swebench.py
git commit -m "feat: record usage stats for memory benchmark runs"
```
