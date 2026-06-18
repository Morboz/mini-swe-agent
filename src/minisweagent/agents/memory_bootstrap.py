from __future__ import annotations

import json
from typing import Any

import httpx

from minisweagent._vendor.formsy_sdk import Client as FormsyClient
from minisweagent.agents.default import DefaultAgent
from minisweagent.utils.evidence import (
    FormsyEvidenceError,
    extract_source_files,
    resolve_revision,
)


class MemoryBootstrapAgent(DefaultAgent):
    """One-shot Evidence bootstrap: ingest + Explore once, inject the result.

    On the agent's first step this ingests the repo under test
    (``repo_id = instance_id``, ``revision = git HEAD``) into the Formsy Evidence
    service and injects an Explore of the task as a ``memory_search`` tool result.
    Failure is non-fatal: we log and continue without injected context.
    """

    def __init__(
        self,
        *args,
        memory: dict[str, Any] | None = None,
        evidence_client: FormsyClient | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory_config = memory or {}
        self._client = evidence_client
        self.memory_state = {"bootstrapped": False}

    def query(self) -> dict:
        if self.memory_config.get("enabled") and not self.memory_state["bootstrapped"]:
            self._bootstrap_memory()
        return super().query()

    def _bootstrap_memory(self) -> None:
        # Mark first to avoid retry loops regardless of outcome.
        self.memory_state["bootstrapped"] = True
        cwd = getattr(getattr(self.env, "config", None), "cwd", "")
        try:
            client = self._client or FormsyClient(
                base_url=self.memory_config.get("base_url"),
                timeout=self.memory_config.get("timeout_seconds", 300),
            )
            repo_id = self.extra_template_vars.get("instance_id", "")
            revision = resolve_revision(self.env, cwd=cwd)
            files = extract_source_files(self.env, cwd=cwd)
            self.logger.info(
                "Evidence extract: %d files (%d bytes) from cwd=%s",
                len(files), sum(len(f.content) for f in files), cwd,
            )
            ingest_resp = client.ingest(repo_id, revision, files)
            self.logger.info(
                "Evidence ingest %s %s@%s: indexed=%d nodes_created=%d edges_created=%d "
                "refs=%d/%d duration=%dms (graph now: %d nodes / %d edges); success=%s",
                "created" if ingest_resp.created else "replaced",
                repo_id, revision,
                ingest_resp.files_indexed, ingest_resp.nodes_created, ingest_resp.edges_created,
                ingest_resp.refs_resolved, ingest_resp.refs_unresolved, ingest_resp.duration_ms,
                ingest_resp.node_count, ingest_resp.edge_count, ingest_resp.success,
            )
            result = client.explore(
                repo_id,
                revision,
                self.extra_template_vars["task"],
                max_output_chars=self.memory_config.get("max_output_chars", 4000),
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
                    "content": result.content,
                    "extra": {"timestamp": 0.0},
                },
            )
            self.extra_template_vars["memory_info"] = {
                "enabled": True,
                "repo_id": repo_id,
                "source_file_count": len(files),
                "source_total_bytes": sum(len(file.content) for file in files),
                "context_chars": len(result.content),
            }
        except (httpx.HTTPError, FormsyEvidenceError) as e:
            self.logger.warning("memory_bootstrap failed — continuing without context: %s", e)
