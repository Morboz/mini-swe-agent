"""Agent with context_search and context_read tools backed by the Formsy Evidence API.

This agent extends DefaultAgent by injecting two additional tools that the LLM
can call during the agent loop:

- ``context_search`` — Explore the Formsy Evidence for the repo under test
  (graph-ranked source for a natural-language query).
- ``context_read`` — Read a staged source file by path and optional line range.

The Evidence (``repo_id = instance_id``, ``revision = git HEAD``) is **ingested
once, eagerly, on the agent's first step** (see ADR-0004 in the Formsy repo);
``context_search`` / ``context_read`` then issue thin Explore / Read calls over
the vendored ``formsy_sdk.Client``. If ingestion or any call fails, the agent
degrades gracefully to bash-only rather than sinking the run.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from minisweagent._vendor.formsy_sdk import Client as FormsyClient
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.models.utils.actions_toolcall import CONTEXT_READ_TOOL, CONTEXT_SEARCH_TOOL
from minisweagent.utils.evidence import (
    FormsyEvidenceError,
    extract_source_files,
    resolve_revision,
)

# NOTE: the ``memory`` config key / kwarg is retained plumbing (threaded through
# the SWE-bench runners and the shared DefaultAgent config); it now holds Formsy
# Evidence connection settings (base_url, timeout_seconds, max_output_chars).


class ContextToolAgentConfig(AgentConfig):
    """Config for ContextToolAgent — includes Formsy Evidence connection settings."""

    memory: dict[str, Any] = {}
    """Formsy Evidence connection config (base_url, timeout_seconds, max_output_chars, ...)."""


class ContextToolAgent(DefaultAgent):
    """Agent with ``context_search`` and ``context_read`` tools.

    These tools let the LLM dynamically query the Formsy Evidence for the repo
    under test during the agent loop, instead of relying on a one-shot bootstrap
    injection.
    """

    def __init__(
        self,
        model,
        env,
        *,
        memory: dict[str, Any] | None = None,
        evidence_client: FormsyClient | None = None,
        config_class: type = ContextToolAgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, memory=memory or {}, **kwargs)
        self.memory_config = memory or {}
        self._client = evidence_client
        self.logger = logging.getLogger("context_tool_agent")

        # Eager-ingest state.
        self._ingested = False
        self._ingest_ok = False
        self._repo_id = ""
        self._revision = ""

        # Inject context tools into the model's tool list.
        if hasattr(model, "config") and hasattr(model.config, "extra_tools"):
            model.config.extra_tools = [CONTEXT_SEARCH_TOOL, CONTEXT_READ_TOOL]

    # -- Eager ingestion -----------------------------------------------------

    def query(self) -> dict:
        """Ingest the Evidence once on the first step, then query the model."""
        if self.memory_config.get("enabled") and not self._ingested:
            self._ensure_ingested()
        return super().query()

    def _ensure_ingested(self) -> None:
        """Ingest the repo under test into the Evidence service exactly once.

        Failure is non-fatal: we log and leave ``_ingest_ok`` False so the tools
        degrade to a "use bash" hint instead of crashing the run.
        """
        self._ingested = True  # guard against retry loops regardless of outcome
        if not self.memory_config.get("enabled"):
            return
        cwd = getattr(getattr(self.env, "config", None), "cwd", "")
        try:
            repo_id = self.extra_template_vars.get("instance_id", "")
            revision = resolve_revision(self.env, cwd=cwd)
            files = extract_source_files(self.env, cwd=cwd)
            self._get_client().ingest(repo_id, revision, files)
            self._repo_id = repo_id
            self._revision = revision
            self._ingest_ok = True
            self.extra_template_vars.setdefault("memory_info", {
                "enabled": True,
                "repo_id": repo_id,
                "context_search_calls": 0,
            })
            self.logger.info(
                "Ingested evidence %s@%s (%d files)", repo_id, revision, len(files)
            )
        except (httpx.HTTPError, FormsyEvidenceError) as e:
            self.logger.warning("Evidence ingest failed — degrading to bash-only: %s", e)

    def _get_client(self) -> FormsyClient:
        """Lazy-initialize the vendored formsy_sdk.Client from config."""
        if self._client is None:
            self._client = FormsyClient(
                base_url=self.memory_config.get("base_url"),
                timeout=self.memory_config.get("timeout_seconds", 300),
            )
        return self._client

    # -- Tool dispatch -------------------------------------------------------

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, dispatching by tool type."""
        actions = message.get("extra", {}).get("actions", [])
        outputs = []
        for action in actions:
            tool = action.get("tool", "bash")
            if tool == "bash":
                outputs.append(self.env.execute(action))
            elif tool == "context_search":
                outputs.append(self._execute_context_search(action))
            elif tool == "context_read":
                outputs.append(self._execute_context_read(action))
            else:
                outputs.append({
                    "output": f"Unknown tool: {tool}",
                    "returncode": 1,
                })
        return self.add_messages(
            *self.model.format_observation_messages(message, outputs, self.get_template_vars())
        )

    # -- context_search ------------------------------------------------------

    def _execute_context_search(self, action: dict) -> dict[str, Any]:
        """Explore the Evidence for a natural-language query."""
        if not self._ingest_ok:
            return {
                "output": "context_search unavailable (evidence not ingested); use bash instead",
                "returncode": 1,
            }
        query = action.get("query", "")
        max_output_chars = action.get("budget", self.memory_config.get("max_output_chars", 4000))
        try:
            result = self._get_client().explore(
                repo_id=self._repo_id,
                revision=self._revision,
                query=query,
                max_output_chars=max_output_chars,
            )
            memory_info = self.extra_template_vars.get("memory_info")
            if isinstance(memory_info, dict):
                memory_info["context_search_calls"] = memory_info.get("context_search_calls", 0) + 1
            return {
                "output": result.content,
                "returncode": 0,
            }
        except (httpx.HTTPError, FormsyEvidenceError) as e:
            self.logger.warning("context_search failed: %s", e)
            return {
                "output": f"context_search failed: {e}",
                "returncode": 1,
            }

    # -- context_read --------------------------------------------------------

    def _execute_context_read(self, action: dict) -> dict[str, Any]:
        """Read a staged source file by path and optional line range.

        The Evidence index stores **repo-relative** paths, but the LLM frequently
        passes absolute container paths it saw via bash (e.g. ``/app/lib/...``
        or ``/testbed/lib/...``). We normalize: strip the leading slash, and on
        404 retry with successive leading path components stripped (most-specific
        candidate first). So ``/app/lib/ansible/x.py`` resolves to the indexed
        ``lib/ansible/x.py``. The matched relative path is echoed in the header.
        """
        if not self._ingest_ok:
            return {
                "output": "context_read unavailable (evidence not ingested); use bash instead",
                "returncode": 1,
            }
        raw_path = action.get("path", "")
        start_line = action.get("start_line") or 1
        end_line = action.get("end_line")
        candidates = _read_path_candidates(raw_path)
        line_suffix = f":{start_line}" if action.get("start_line") else ""
        line_suffix += f"-{end_line}" if end_line else ""
        last_error: Exception | None = None
        for path in candidates:
            try:
                result = self._get_client().read_file(
                    repo_id=self._repo_id,
                    revision=self._revision,
                    path=path,
                    start_line=start_line,
                    end_line=end_line,
                )
                header = f"{path}{line_suffix}\n\n"
                truncation = "\n[truncated by Evidence server]" if result.truncated else ""
                return {"output": header + result.content + truncation, "returncode": 0}
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code != 404:
                    break  # non-404 HTTP error: stop retrying, degrade below
                # 404: try the next (shorter) candidate
            except (httpx.HTTPError, FormsyEvidenceError) as e:
                last_error = e
                break
        self.logger.warning(
            "context_read failed for %r (tried %s): %s", raw_path, candidates, last_error
        )
        return {
            "output": (
                f"context_read: '{raw_path}' not found in Evidence index "
                f"(tried {candidates}). The index uses repo-relative paths — "
                "use context_search to get the correct path."
            ),
            "returncode": 1,
        }


def _read_path_candidates(raw: str) -> list[str]:
    """Normalized path candidates for a context_read path.

    Strips the leading slash, then yields the path with 0, 1, 2 leading
    components stripped (most-specific first). ``/app/lib/x.py`` →
    ``["app/lib/x.py", "lib/x.py", "x.py"]``. The Evidence index is tried in
    that order on 404.
    """
    p = (raw or "").strip().lstrip("/")
    if not p:
        return []
    parts = p.split("/")
    candidates: list[str] = []
    seen: set[str] = set()
    for i in range(min(len(parts), 3)):
        cand = "/".join(parts[i:])
        if cand and cand not in seen:
            seen.add(cand)
            candidates.append(cand)
    return candidates
