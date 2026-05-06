from __future__ import annotations

import json
from typing import Any

from minisweagent.agents.default import DefaultAgent
from minisweagent.utils.memory import MemoryClient, extract_memory_source_files


class MemoryBootstrapAgent(DefaultAgent):
    def __init__(
        self, *args, memory: dict[str, Any] | None = None, memory_client: MemoryClient | None = None, **kwargs
    ):
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
            timeout_seconds=self.memory_config.get("timeout_seconds", 300),
        )
        repo_id = self.extra_template_vars.get("instance_id", "")
        files = extract_memory_source_files(self.env, cwd=getattr(getattr(self.env, "config", None), "cwd", ""))
        compile_result = client.compile_repo(
            repo_id,
            files,
            metadata={"instance_id": repo_id},
            revision=self.memory_config.get("revision"),
            enable_w2=self.memory_config.get("enable_w2", False),
        )
        query_result = client.query_repo(
            repo_id,
            self.extra_template_vars["task"],
            metadata={"instance_id": repo_id},
            revision=compile_result.get("revision"),
            budget=self.memory_config.get("query_budget", 4000),
        )
        tool_call_id = "memory_bootstrap"
        self.add_messages(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "memory_search",
                            "arguments": json.dumps({"query": self.extra_template_vars["task"]}),
                        },
                    }
                ],
                "extra": {"timestamp": 0.0},
            },
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": query_result.get("extra_context", ""),
                "extra": {"timestamp": 0.0},
            },
        )
        self.extra_template_vars["memory_info"] = {
            "enabled": True,
            "repo_id": repo_id,
            "compile_success": True,
            "query_success": True,
            "compile_latency_ms": compile_result.get("_latency_ms", 0),
            "query_latency_ms": query_result.get("_latency_ms", 0),
            "revision": compile_result.get("revision"),
            "source_file_count": len(files),
            "source_total_bytes": sum(len(file.get("content", "")) for file in files),
            "context_chars": len(query_result.get("extra_context", "")),
        }
        self.memory_state["bootstrapped"] = True
