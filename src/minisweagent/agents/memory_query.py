from __future__ import annotations

import re
import shlex
from typing import Any

from minisweagent.agents.memory_bootstrap import MemoryBootstrapAgent
from minisweagent.utils.memory import MemoryClient


SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".m",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".scala",
    ".sh",
    ".swift",
    ".ts",
    ".tsx",
}

CONTENT_LOOKUP_COMMANDS = {"rg", "grep", "ack", "ag"}
SOURCE_READ_COMMANDS = {"cat", "bat", "less", "more", "sed", "head", "tail", "nl"}
PATCH_OR_OUTPUT_FILES = {"patch.txt"}


class MemoryQueryAgent(MemoryBootstrapAgent):
    """Redirect source-code content lookups to the memory query backend.

    Bash remains available for directory exploration, editing, testing, patch generation,
    and submission. Only commands that read or search source-code contents are intercepted.
    """

    def __init__(
        self, *args, memory: dict[str, Any] | None = None, memory_client: MemoryClient | None = None, **kwargs
    ):
        super().__init__(*args, memory=memory, memory_client=memory_client, **kwargs)
        self.memory_state["revision"] = None
        self.memory_state["redirect_count"] = 0

    def execute_actions(self, message: dict) -> list[dict]:
        outputs = []
        for action in message.get("extra", {}).get("actions", []):
            command = action.get("command", "")
            if self._should_redirect_to_memory(command):
                outputs.append(self._query_memory(command))
            else:
                outputs.append(self.env.execute(action))
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def _bootstrap_memory(self) -> None:
        super()._bootstrap_memory()
        self.memory_state["revision"] = self.extra_template_vars.get("memory_info", {}).get("revision")

    def _query_memory(self, command: str) -> dict[str, Any]:
        if self.memory_config.get("enabled") and not self.memory_state["bootstrapped"]:
            self._bootstrap_memory()
        client = self.memory_client or MemoryClient(
            base_url=self.memory_config["base_url"],
            timeout_seconds=self.memory_config.get("timeout_seconds", 300),
        )
        repo_id = self.extra_template_vars.get("instance_id", "")
        query = self._command_to_memory_query(command)
        result = client.query_repo(
            repo_id,
            query,
            metadata={"instance_id": repo_id, "source": "memory_query_agent", "command": command},
            revision=self.memory_state.get("revision") or self.memory_config.get("revision"),
            budget=self.memory_config.get("query_budget", 4000),
        )
        self.memory_state["redirect_count"] += 1
        info = self.extra_template_vars.setdefault("memory_info", {"enabled": True})
        info["redirect_count"] = self.memory_state["redirect_count"]
        return {
            "returncode": 0,
            "exception_info": "",
            "output": (
                "The source-code lookup command was redirected to memory_search.\n\n"
                f"<memory_query>{query}</memory_query>\n\n"
                f"{result.get('extra_context', '')}"
            ),
            "extra": {
                "memory_redirected": True,
                "memory_query": query,
                "memory_query_latency_ms": result.get("_latency_ms", 0),
            },
        }

    def _command_to_memory_query(self, command: str) -> str:
        command = command.strip()
        match = re.match(r"memory_search\s+(.+)", command, re.DOTALL)
        if match:
            try:
                args = shlex.split(match.group(1))
                return " ".join(args).strip() or command
            except ValueError:
                return match.group(1).strip().strip("\"'")
        return command

    def _should_redirect_to_memory(self, command: str) -> bool:
        command = command.strip()
        if not command:
            return False
        if command.startswith("memory_search "):
            return True
        if self._is_patch_or_submission_command(command):
            return False
        parts = self._split_command(command)
        if not parts:
            return False
        if self._is_directory_or_metadata_command(parts, command):
            return False
        if self._has_content_lookup(parts, command):
            return True
        return self._reads_source_file(parts)

    def _split_command(self, command: str) -> list[str]:
        try:
            return shlex.split(command)
        except ValueError:
            return command.split()

    def _is_patch_or_submission_command(self, command: str) -> bool:
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command:
            return True
        if re.search(r"\bpatch\.txt\b", command):
            return True
        if re.search(r"\bgit\s+(diff|status|show|ls-files)\b", command):
            return True
        return False

    def _is_directory_or_metadata_command(self, parts: list[str], command: str) -> bool:
        executable = _basename(parts[0])
        if executable in {"pwd", "ls", "tree", "fd"}:
            return True
        if executable == "git" and len(parts) > 1 and parts[1] in {"status", "diff", "ls-files"}:
            return True
        if executable == "find":
            return not re.search(r"\b(exec|ok)\b|xargs|grep|rg|cat|sed|head|tail", command)
        return False

    def _has_content_lookup(self, parts: list[str], command: str) -> bool:
        executables = {_basename(part) for part in parts}
        if executables & CONTENT_LOOKUP_COMMANDS:
            if re.search(r"\b(ls|find|git\s+ls-files)\b.*\|\s*(grep|rg)\b", command):
                return False
            return True
        if re.search(r"\|\s*(grep|rg|ack|ag)\b", command):
            return not re.search(r"\b(ls|find|git\s+ls-files)\b.*\|\s*(grep|rg|ack|ag)\b", command)
        return False

    def _reads_source_file(self, parts: list[str]) -> bool:
        executable = _basename(parts[0])
        if executable not in SOURCE_READ_COMMANDS:
            return False
        return any(_looks_like_source_path(part) for part in parts[1:])


def _basename(value: str) -> str:
    return value.rsplit("/", 1)[-1]


def _looks_like_source_path(value: str) -> bool:
    if value.startswith("-") or value in PATCH_OR_OUTPUT_FILES:
        return False
    cleaned = value.strip("\"'")
    return any(cleaned.endswith(extension) for extension in SOURCE_EXTENSIONS)
