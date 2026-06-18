# Design: ContextToolAgent

> **Update (2026-06):** the `context_read` tool has been **removed**. It offered
> no benefit over reading files with `bash` (sed/cat/head), so the agent now
> exposes only `context_search` + `bash`. The `context_read` sections below are
> kept as historical design record. See `ContextToolAgent` and
> `swebench-context-tool.yaml`.

## Goal

Create a new Agent class that injects `context_search` and `context_read` as LLM-callable tools, enabling dynamic Formsy context queries during the agent loop (not just a one-shot bootstrap). This is inspired by the OpenCode plugin's tool architecture in `plugin-opencode/src/plugin.ts` and `runtime.ts`.

## Architecture Decision

- **Base class:** `DefaultAgent` (not `MemoryBootstrapAgent`) — clean separation; the new agent handles compile-on-demand + per-step context queries itself.
- **Approach:** Agent-layer interception — the agent's `execute_actions()` dispatches `context_search`/`context_read` tool calls to `MemoryClient`, while `bash` tool calls go through `env.execute()` as before.

## File Changes

### 1. `models/utils/actions_toolcall.py`

**Add tool schemas:**

- `CONTEXT_SEARCH_TOOL` — function schema with `query` (required), `budget` (optional)
- `CONTEXT_READ_TOOL` — function schema with `path` (required), `start_line`, `end_line` (optional)

**Modify `parse_toolcall_actions`:**

Current behavior: hardcodes `tool_call.function.name != "bash"` as unknown tool.

New behavior: accept `bash`, `context_search`, `context_read` as known tools. Each action dict gains a `"tool"` key:

```python
{"tool": "bash", "command": "...", "tool_call_id": "..."}
{"tool": "context_search", "query": "...", "budget": 4000, "tool_call_id": "..."}
{"tool": "context_read", "path": "...", "start_line": 10, "end_line": 50, "tool_call_id": "..."}
```

Unknown tools still raise `FormatError`.

### 2. `models/litellm_model.py`

**Minimal change to `_query`:**

```python
def _query(self, messages, **kwargs):
    tools = [BASH_TOOL] + getattr(self.config, 'extra_tools', [])
    return litellm.completion(model=..., messages=messages, tools=tools, ...)
```

Add `extra_tools: list = []` to `LitellmModelConfig`.

### 3. `utils/memory.py`

**Add methods to `MemoryClient`:**

- `read_repo(repo_id, path, *, revision, start_line, end_line)` — POST `/api/v1/read`
- `compile_status(repo_id, revision)` — POST `/api/v1/compile/status`
- `ensure_compiled(repo_id, files, query, *, revision, ...)` — ported `ensureMemoryCompiled` logic:
  1. Check local `_compiled_state` cache (query signature match)
  2. Check server-side `compile_status` for existing valid compile
  3. If neither satisfies, run `compile_repo`
  4. Cache the result identity for subsequent calls

### 4. `agents/context_tool_agent.py` (NEW)

```python
class ContextToolAgentConfig(AgentConfig):
    memory: dict = {}

class ContextToolAgent(DefaultAgent):
    def __init__(self, model, env, *, memory=None, memory_client=None, **kwargs):
        super().__init__(model, env, **kwargs)
        self.memory_config = memory or {}
        self.memory_client = memory_client
        self._compiled_state = {"repo_id": None, "revision": None, "query_signature": None}

        # Inject extra tools into model config
        if hasattr(model.config, 'extra_tools'):
            model.config.extra_tools = [CONTEXT_SEARCH_TOOL, CONTEXT_READ_TOOL]

    def execute_actions(self, message):
        actions = message.get("extra", {}).get("actions", [])
        results = []
        for action in actions:
            tool = action.get("tool", "bash")
            if tool == "bash":
                results.append(self.env.execute(action))
            elif tool == "context_search":
                results.append(self._execute_context_search(action))
            elif tool == "context_read":
                results.append(self._execute_context_read(action))
            else:
                results.append({"output": f"Unknown tool: {tool}", "returncode": 1})
        return self.add_messages(
            *self.model.format_observation_messages(message, results, self.get_template_vars())
        )

    def _get_client(self) -> MemoryClient:
        if self.memory_client is None:
            self.memory_client = MemoryClient(
                base_url=self.memory_config["base_url"],
                timeout_seconds=self.memory_config.get("timeout_seconds", 300),
                api_key=self.memory_config.get("api_key"),
            )
        return self.memory_client

    def _resolve_repo_context(self):
        repo_id = self.extra_template_vars.get("instance_id", "")
        revision = self.memory_config.get("revision")
        return repo_id, revision

    def _execute_context_search(self, action):
        client = self._get_client()
        repo_id, revision = self._resolve_repo_context()
        query = action.get("query", "")
        budget = action.get("budget", self.memory_config.get("query_budget", 4000))

        # ensureCompiled — auto-compile if needed
        self._ensure_compiled(repo_id=repo_id, query=query, revision=revision)

        try:
            result = client.query_repo(
                repo_id=repo_id,
                query=query,
                revision=revision,
                budget=budget,
            )
            return {
                "output": result.get("extra_context", ""),
                "returncode": 0,
                "tool": "context_search",
            }
        except MemoryQueryError as e:
            return {
                "output": f"context_search failed: {e}",
                "returncode": 1,
                "tool": "context_search",
            }

    def _execute_context_read(self, action):
        client = self._get_client()
        repo_id, revision = self._resolve_repo_context()
        path = action.get("path", "")

        try:
            result = client.read_repo(
                repo_id=repo_id,
                path=path,
                revision=revision,
                start_line=action.get("start_line"),
                end_line=action.get("end_line"),
            )
            return {
                "output": result.get("content", ""),
                "returncode": 0,
                "tool": "context_read",
            }
        except MemoryError as e:
            return {
                "output": f"context_read failed: {e}",
                "returncode": 1,
                "tool": "context_read",
            }

    def _ensure_compiled(self, repo_id, query, revision=None):
        client = self._get_client()

        # Check local cache
        if self._compiled_state.get("repo_id") == repo_id:
            cached_sig = self._compiled_state.get("query_signature")
            current_sig = self._query_signature(query)
            if cached_sig == "*" or cached_sig == current_sig:
                return  # Already compiled for this query

        # Check server-side status
        try:
            status = client.compile_status(repo_id, revision)
            if status and self._existing_compile_satisfies(status, query):
                self._compiled_state = {
                    "repo_id": repo_id,
                    "revision": status.get("revision", revision),
                    "query_signature": self._compile_signature_from_status(status, query),
                }
                return
        except Exception:
            pass  # Fall through to compile

        # Compile
        files = extract_memory_source_files(
            self.env,
            cwd=getattr(getattr(self.env, "config", None), "cwd", ""),
        )
        result = client.compile_repo(
            repo_id,
            files,
            metadata={"instance_id": repo_id},
            revision=revision,
            enable_w2=self.memory_config.get("enable_w2", False),
        )
        self._compiled_state = {
            "repo_id": repo_id,
            "revision": result.get("revision", revision),
            "query_signature": "*",  # full compile covers all queries
        }

    @staticmethod
    def _query_signature(query: str) -> str:
        import hashlib
        normalized = " ".join(query.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    @staticmethod
    def _existing_compile_satisfies(status, query) -> bool:
        metadata = status.get("metadata", {})
        if str(metadata.get("source_scope", "")).strip().lower() == "full":
            return True
        parsed_count = int(status.get("parsed_file_count", 0) or 0)
        profile = str(metadata.get("compile_profile", "")).strip().lower()
        if profile != "interactive_context_search" and parsed_count > 260:
            return True
        sig = str(metadata.get("query_signature", "")).strip()
        if sig and sig == ContextToolAgent._query_signature(query):
            return True
        return False

    @staticmethod
    def _compile_signature_from_status(status, query) -> str:
        metadata = status.get("metadata", {})
        if str(metadata.get("source_scope", "")).strip().lower() == "full":
            return "*"
        sig = str(metadata.get("query_signature", "")).strip()
        return sig if sig else ContextToolAgent._query_signature(query)
```

### 5. `agents/__init__.py`

Add to `_AGENT_MAPPING`:
```python
"context_tool": "minisweagent.agents.context_tool_agent.ContextToolAgent",
```

### 6. Config integration

`swebench.yaml` (or a new `swebench-context-tool.yaml`) can set:
```yaml
agent:
  agent_class: context_tool
  memory:
    enabled: true
    base_url: "http://localhost:3001"
    timeout_seconds: 300
    query_budget: 4000
    api_key: "fsy_test_key_dev_only_12345678"
```

And the `swebench_single.py` runner needs a branch to handle `context_tool` agent class similarly to `memory_bootstrap`.

## Observation Formatting

The existing `format_toolcall_observation_messages` works by matching `tool_call_id` and rendering output via the `observation_template`. For context tools, the output dict has the same shape (`output`, `returncode`), so the same template works. The only difference is that context tool results don't have `exception_info`, which is fine (it's optional in the template).

## Implementation Order

1. Add tool schemas to `actions_toolcall.py` + modify `parse_toolcall_actions`
2. Add `extra_tools` to `LitellmModelConfig` + update `_query`
3. Add `read_repo`, `compile_status`, `ensure_compiled` to `MemoryClient`
4. Create `context_tool_agent.py`
5. Register in `agents/__init__.py`
6. Update `swebench_single.py` to handle `context_tool` agent class
7. Add config file
8. Test
