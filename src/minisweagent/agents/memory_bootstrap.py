from __future__ import annotations

import json
import re
from typing import Any

from minisweagent.agents.default import DefaultAgent
from minisweagent.utils.memory import MemoryClient, extract_memory_source_files


CONTEXT_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "context_search",
        "description": (
            "Search Formsy's compiled code memory/context for information relevant to a natural-language query. "
            "Use this before broad shell searching when memory is enabled. Prefer targeted queries about symbols, "
            "file paths, behavior, call flow, and edge cases. When it returns a relevant file or span, use "
            "context_read next to inspect exact source context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language query describing the code, behavior, or fact to find.",
                },
                "repo_id": {
                    "type": "string",
                    "description": "Repository identifier. Use the current SWE-bench instance_id; defaults to it when omitted.",
                },
                "revision": {
                    "type": "string",
                    "description": (
                        "Optional revision override. Usually omit this: the agent compiles revision 'latest' "
                        "before the first context_search and then automatically queries the compiled revision."
                    ),
                },
                "budget": {
                    "type": "integer",
                    "description": "Context token budget for the query.",
                    "default": 4000,
                    "minimum": 1,
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional server-side query metadata for retrieval mode or trace control.",
                },
            },
            "required": ["query"],
        },
    },
}

CONTEXT_READ_TOOL = {
    "type": "function",
    "function": {
        "name": "context_read",
        "description": (
            "Read exact source context from Formsy's compiled repository memory. Use after context_search returns "
            "a relevant file path or line range."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repository-relative file path to read."},
                "repo_id": {
                    "type": "string",
                    "description": "Repository identifier. Use the current SWE-bench instance_id; defaults to it when omitted.",
                },
                "revision": {
                    "type": "string",
                    "description": (
                        "Optional revision override. Usually omit this: context_read automatically uses the "
                        "compiled revision after context_search has compiled the repository."
                    ),
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional 1-indexed first source line to read.",
                    "minimum": 1,
                },
                "end_line": {
                    "type": "integer",
                    "description": "Optional inclusive 1-indexed last source line to read.",
                    "minimum": 1,
                },
            },
            "required": ["path"],
        },
    },
}

CONTEXT_SEARCH_RESPONSE_TOOL = {
    "type": "function",
    "name": CONTEXT_SEARCH_TOOL["function"]["name"],
    "description": CONTEXT_SEARCH_TOOL["function"]["description"],
    "parameters": CONTEXT_SEARCH_TOOL["function"]["parameters"],
}

CONTEXT_READ_RESPONSE_TOOL = {
    "type": "function",
    "name": CONTEXT_READ_TOOL["function"]["name"],
    "description": CONTEXT_READ_TOOL["function"]["description"],
    "parameters": CONTEXT_READ_TOOL["function"]["parameters"],
}


class MemoryBootstrapAgent(DefaultAgent):
    def __init__(
        self, *args, memory: dict[str, Any] | None = None, memory_client: MemoryClient | None = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.memory_config = {"enabled": False, **(memory or {})}
        self.memory_client = memory_client
        self.extra_template_vars["memory"] = self.memory_config
        self.memory_state = {
            "compiled": False,
            "compile_revision": None,
            "retrieval_state": "not_started",
            "candidate_files": [],
            "accepted_targets": [],
            "test_plan_files": [],
            "grounded_search_required": False,
            "tool_calls": {"context_search": 0, "context_read": 0},
        }
        if self.memory_config.get("enabled"):
            self._install_context_tools()

    def _install_context_tools(self) -> None:
        tools = getattr(self.model, "tools", None)
        if not isinstance(tools, list):
            return
        existing_names = {self._tool_name(tool) for tool in tools}
        context_tools = (
            (CONTEXT_SEARCH_RESPONSE_TOOL, CONTEXT_READ_RESPONSE_TOOL)
            if tools and "name" in tools[0]
            else (CONTEXT_SEARCH_TOOL, CONTEXT_READ_TOOL)
        )
        for tool in context_tools:
            if self._tool_name(tool) not in existing_names:
                tools.append(tool)

    @staticmethod
    def _tool_name(tool: dict[str, Any]) -> str:
        return str(tool.get("function", {}).get("name") or tool.get("name") or "")

    def execute_actions(self, message: dict) -> list[dict]:
        outputs = [self._execute_action(action) for action in message.get("extra", {}).get("actions", [])]
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def _execute_action(self, action: dict) -> dict:
        block_message = self._gate_block_message(action)
        if block_message:
            return self._tool_error(block_message)
        tool_name = action.get("tool")
        if tool_name == "context_search":
            return self._execute_context_search(action)
        if tool_name == "context_read":
            return self._execute_context_read(action)
        return self.env.execute(action)

    def _memory_client(self) -> MemoryClient:
        if self.memory_client is None:
            self.memory_client = MemoryClient(
                base_url=self.memory_config["base_url"],
                timeout_seconds=self.memory_config.get("timeout_seconds", 300),
            )
        return self.memory_client

    def _repo_id(self, action: dict) -> str:
        return str(action.get("repo_id") or self.extra_template_vars.get("instance_id", ""))

    def _revision(self, action: dict) -> str | None:
        revision = action.get(
            "revision",
            self.memory_state.get("compile_revision") or self.memory_config.get("revision") or "latest",
        )
        return str(revision) if revision else None

    def _metadata(self, action: dict) -> dict[str, Any]:
        metadata = action.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("case_id", self._repo_id(action))
        metadata.setdefault("retrieval_mode", "symbolic")
        metadata.setdefault("grounding_phase", "seed")
        metadata.setdefault("response_format", "bundle")
        return {"instance_id": self.extra_template_vars.get("instance_id", ""), **metadata}

    def _gate_block_message(self, action: dict) -> str | None:
        if not self.memory_config.get("enabled"):
            return None
        tool_name = action.get("tool")
        if tool_name == "context_search":
            if self.memory_state["grounded_search_required"] and self._metadata(action).get("grounding_phase") != "grounded":
                return (
                    "Retrieval gate active: context_read requires a grounded context_search next, "
                    "with metadata.grounding_phase='grounded'."
                )
            if self.memory_state["retrieval_state"] == "grounded":
                return (
                    "Retrieval gate active: accepted targets close exploration; additional "
                    "context_search calls are blocked. Continue with accepted-target reads, editing, or tests."
                )
            return None
        if tool_name == "context_read":
            path = str(action.get("path") or "").strip()
            if self._is_context_read_allowed(path):
                return None
            return (
                "Retrieval gate active: context_read is limited to accepted targets or test-plan files "
                "after a grounded target has been accepted."
            )
        if "command" in action:
            return self._bash_gate_block_message(str(action.get("command") or ""))
        return None

    def _bash_gate_block_message(self, command: str) -> str | None:
        state = self.memory_state["retrieval_state"]
        if state == "not_started":
            return (
                "Retrieval gate active: call context_search first with "
                "metadata.retrieval_mode='symbolic' and metadata.grounding_phase='seed'."
            )
        if state == "inspect_candidates":
            return (
                "Retrieval gate active: use context_read on a candidate file before "
                "shell exploration or editing."
            )
        if state == "context_read" and (self._is_edit_command(command) or self._is_broad_discovery_command(command)):
            return (
                "Retrieval gate active: context_read requires a grounded context_search "
                "before broad terminal exploration or editing."
            )
        if state == "grounded" and self._is_broad_discovery_command(command):
            return (
                "Retrieval gate active: accepted targets plus server test plan are available; "
                "broad grep/find/search commands are blocked."
            )
        return None

    def _execute_context_search(self, action: dict) -> dict:
        query = str(action.get("query") or "").strip()
        if not query:
            return self._tool_error("context_search requires a non-empty query")
        self._ensure_compiled(action)
        result = self._memory_client().query_repo(
            self._repo_id(action),
            query,
            metadata=self._metadata(action),
            revision=self._revision(action),
            budget=self._positive_int(action.get("budget"), self.memory_config.get("query_budget", 4000)),
        )
        self.memory_state["tool_calls"]["context_search"] += 1
        self._record_context_search_result(result, metadata=self._metadata(action))
        self._update_memory_info("context_search", result)
        return self._tool_output(json.dumps(result, ensure_ascii=False))

    def _ensure_compiled(self, action: dict) -> None:
        if self.memory_state["compiled"]:
            return
        repo_id = self._repo_id(action)
        files = extract_memory_source_files(self.env, cwd=getattr(getattr(self.env, "config", None), "cwd", ""))
        result = self._memory_client().compile_repo(
            repo_id,
            files,
            metadata={"instance_id": self.extra_template_vars.get("instance_id", repo_id)},
            revision=action.get("revision") or self.memory_config.get("revision") or "latest",
            enable_w2=self.memory_config.get("enable_w2", False),
        )
        self.memory_state["compiled"] = True
        self.memory_state["compile_revision"] = result.get("revision")
        self.extra_template_vars["memory_info"] = {
            "enabled": True,
            "repo_id": repo_id,
            "compile_success": True,
            "compile_latency_ms": result.get("_latency_ms", 0),
            "source_file_count": len(files),
            "source_total_bytes": sum(len(file.get("content", "")) for file in files),
            "tool_calls": dict(self.memory_state["tool_calls"]),
        }

    def _execute_context_read(self, action: dict) -> dict:
        path = str(action.get("path") or "").strip()
        if not path:
            return self._tool_error("context_read requires a non-empty path")
        result = self._memory_client().read_repo(
            self._repo_id(action),
            path,
            metadata=self._metadata(action),
            revision=self._revision(action),
            start_line=self._optional_positive_int(action.get("start_line")),
            end_line=self._optional_positive_int(action.get("end_line")),
        )
        self.memory_state["tool_calls"]["context_read"] += 1
        self.memory_state["retrieval_state"] = "context_read"
        self.memory_state["grounded_search_required"] = True
        path = str(result.get("path") or path)
        if path and path not in self.memory_state["candidate_files"]:
            self.memory_state["candidate_files"].append(path)
        self._update_memory_info("context_read", result)
        return self._tool_output(self._format_context_read(result, requested_path=path))

    def _record_context_search_result(self, result: dict[str, Any], *, metadata: dict[str, Any]) -> None:
        phase = str(metadata.get("grounding_phase") or "seed")
        coverage = str(result.get("coverage") or "").lower()
        files = self._extract_result_files(result)
        if phase == "grounded":
            accepted = self._coerce_string_list(result.get("accepted_targets")) or self._coerce_string_list(
                result.get("grounded_files")
            ) or files or list(self.memory_state["candidate_files"])
            self.memory_state["accepted_targets"] = accepted
            self.memory_state["retrieval_state"] = "grounded"
            self.memory_state["grounded_search_required"] = False
            for path in self._extract_test_plan_files(result.get("test_plan")):
                if path not in self.memory_state["test_plan_files"]:
                    self.memory_state["test_plan_files"].append(path)
            return
        if coverage == "poor" or not files:
            self.memory_state["retrieval_state"] = "retry"
            return
        for path in files:
            if path not in self.memory_state["candidate_files"]:
                self.memory_state["candidate_files"].append(path)
        self.memory_state["retrieval_state"] = "inspect_candidates"

    def _is_context_read_allowed(self, path: str) -> bool:
        if self.memory_state["retrieval_state"] != "grounded":
            return True
        accepted = set(self.memory_state["accepted_targets"])
        test_plan = set(self.memory_state["test_plan_files"])
        return path in accepted or path in test_plan or path.startswith("tests/")

    @staticmethod
    def _coerce_string_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in result:
                result.append(text)
        return result

    @staticmethod
    def _extract_result_files(result: dict[str, Any]) -> list[str]:
        files: list[str] = []
        for match in result.get("matches") or []:
            if isinstance(match, dict):
                path = match.get("path") or match.get("file") or match.get("filepath")
                if path and str(path) not in files:
                    files.append(str(path))
        for key in ("grounded_files", "accepted_targets"):
            for path in MemoryBootstrapAgent._coerce_string_list(result.get(key)):
                if path not in files:
                    files.append(path)
        return files

    @staticmethod
    def _extract_test_plan_files(test_plan: Any) -> list[str]:
        if not isinstance(test_plan, dict):
            return []
        paths: list[str] = []
        for key in ("files", "file_paths", "paths", "read_files", "targets", "target_files"):
            value = test_plan.get(key)
            items = value if isinstance(value, list) else [value]
            for item in items:
                if isinstance(item, str) and item.strip() and item not in paths:
                    paths.append(item)
                elif isinstance(item, dict):
                    path = item.get("path") or item.get("file") or item.get("filepath")
                    if path and str(path) not in paths:
                        paths.append(str(path))
        return paths

    @staticmethod
    def _is_broad_discovery_command(command: str) -> bool:
        text = " ".join(str(command or "").split()).lower()
        if not text:
            return False
        patterns = ("grep ", " find ", " fd ", " rg ", "git grep", "ack ", "ag ")
        return any(pattern in f" {text} " for pattern in patterns) or text.startswith(("find ", "grep ", "rg "))

    @staticmethod
    def _is_edit_command(command: str) -> bool:
        text = " ".join(str(command or "").split()).lower()
        if not text:
            return False
        return bool(
            re.search(r"\bopen\s*\(", text)
            or any(marker in text for marker in ("cat >", "tee ", "sed -i", "perl -pi", "apply_patch", "git apply"))
            or ">" in text
        )

    def _update_memory_info(self, tool_name: str, result: dict[str, Any]) -> None:
        info = self.extra_template_vars.setdefault("memory_info", {"enabled": bool(self.memory_config.get("enabled"))})
        info["enabled"] = bool(self.memory_config.get("enabled"))
        info["repo_id"] = result.get("repo_id", info.get("repo_id", self.extra_template_vars.get("instance_id", "")))
        info["last_tool"] = tool_name
        info["last_latency_ms"] = result.get("_latency_ms", 0)
        info["tool_calls"] = dict(self.memory_state["tool_calls"])

    @staticmethod
    def _format_context_read(result: dict[str, Any], *, requested_path: str) -> str:
        path = result.get("path") or requested_path
        content = result.get("content", "")
        start_line = result.get("start_line")
        end_line = result.get("end_line")
        lines = f"{start_line}-{end_line}" if start_line and end_line else "unknown"
        return f"ok: true\npath: {path}\nlines: {lines}\n\n```python\n{content}\n```"

    @staticmethod
    def _tool_output(output: str) -> dict:
        return {"output": output, "returncode": 0, "exception_info": ""}

    @staticmethod
    def _tool_error(error: str) -> dict:
        return {"output": error, "returncode": 2, "exception_info": error}

    @staticmethod
    def _positive_int(value: Any, default: Any) -> int:
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return max(1, int(default))

    @staticmethod
    def _optional_positive_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return None
