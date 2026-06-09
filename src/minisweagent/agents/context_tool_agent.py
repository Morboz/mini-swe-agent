"""Agent with context_search and context_read tools for Formsy repository context.

This agent extends DefaultAgent by injecting two additional tools that the LLM
can call during the agent loop:

- ``context_search`` — query the Formsy repository context index for relevant
  code, symbols, tests, and observations.
- ``context_read`` — read indexed source content by path and optional line range.

An ``ensure_compiled`` guard automatically compiles the repository on the first
``context_search`` call (with caching and server-side status checks to avoid
redundant compiles), ported from the OpenCode plugin's ``ensureMemoryCompiled``.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.models.utils.actions_toolcall import CONTEXT_READ_TOOL, CONTEXT_SEARCH_TOOL
from minisweagent.utils.memory import FormsyMemoryError, MemoryClient, extract_memory_source_files


class ContextToolAgentConfig(AgentConfig):
    """Config for ContextToolAgent — includes Formsy memory/gateway settings."""

    memory: dict[str, Any] = {}
    """Formsy gateway configuration (base_url, timeout_seconds, api_key, etc.)."""


class ContextToolAgent(DefaultAgent):
    """Agent with ``context_search`` and ``context_read`` tools.

    These tools let the LLM dynamically query the Formsy repository context
    during the agent loop, instead of relying on a one-shot bootstrap injection.
    """

    def __init__(
        self,
        model,
        env,
        *,
        memory: dict[str, Any] | None = None,
        memory_client: MemoryClient | None = None,
        config_class: type = ContextToolAgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, memory=memory or {}, **kwargs)
        self.memory_config = memory or {}
        self._memory_client = memory_client
        self.logger = logging.getLogger("context_tool_agent")

        # Inject context tools into the model's tool list
        if hasattr(model, "config") and hasattr(model.config, "extra_tools"):
            model.config.extra_tools = [CONTEXT_SEARCH_TOOL, CONTEXT_READ_TOOL]

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
        """Execute a context_search tool call via the Formsy gateway."""
        client = self._get_client()
        repo_id, revision = self._resolve_repo_context()
        query = action.get("query", "")
        budget = action.get("budget", self.memory_config.get("query_budget", 4000))

        # Ensure the repository is compiled before querying
        try:
            self._ensure_compiled(repo_id=repo_id, query=query, revision=revision)
        except FormsyMemoryError as e:
            self.logger.warning("context_search: compile failed: %s", e)
            return {
                "output": f"context_search skipped (compile failed): {e}",
                "returncode": 1,
            }

        try:
            result = client.query_repo(
                repo_id=repo_id,
                query=query,
                revision=revision,
                budget=budget,
                metadata={"instance_id": repo_id},
            )
            extra_context = result.get("extra_context", "")
            self.extra_template_vars.setdefault("memory_info", {
                "enabled": True,
                "repo_id": repo_id,
                "context_search_calls": 0,
            })
            self.extra_template_vars["memory_info"]["context_search_calls"] += 1
            return {
                "output": extra_context,
                "returncode": 0,
            }
        except FormsyMemoryError as e:
            self.logger.warning("context_search: query failed: %s", e)
            return {
                "output": f"context_search failed: {e}",
                "returncode": 1,
            }

    # -- context_read --------------------------------------------------------

    def _execute_context_read(self, action: dict) -> dict[str, Any]:
        """Execute a context_read tool call via the Formsy gateway."""
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
            content = result.get("content", "")
            # Format with path and line info header
            start = action.get("start_line")
            end = action.get("end_line")
            line_suffix = f":{start}" if start else ""
            line_suffix += f"-{end}" if end else ""
            header = f"{path}{line_suffix}\n\n"
            return {
                "output": header + content,
                "returncode": 0,
            }
        except FormsyMemoryError as e:
            self.logger.warning("context_read: failed: %s", e)
            return {
                "output": f"context_read failed: {e}",
                "returncode": 1,
            }

    # -- ensure_compiled -----------------------------------------------------

    def _ensure_compiled(self, repo_id: str, query: str, revision: str | None = None) -> None:
        """Ensure the repository is compiled before querying.

        Delegates to MemoryClient.ensure_compiled, only extracting source files
        from the environment when a compile is actually needed.
        """
        client = self._get_client()

        # Fast path: already cached — no need to extract files
        if client.is_compile_cached(repo_id, query, revision):
            return

        # Need to compile — extract source files and delegate to client
        cwd = getattr(getattr(self.env, "config", None), "cwd", "")
        files = extract_memory_source_files(self.env, cwd=cwd)
        self.logger.info(
            "Compiling repo %s with %d source files (query: %.40s...)",
            repo_id,
            len(files),
            query,
        )
        client.ensure_compiled(
            repo_id=repo_id,
            files=files,
            query=query,
            revision=revision,
            enable_w2=self.memory_config.get("enable_w2", False),
            metadata={"instance_id": repo_id},
        )

    # -- Helpers -------------------------------------------------------------

    def _get_client(self) -> MemoryClient:
        """Lazy-initialize the MemoryClient from config."""
        if self._memory_client is None:
            self._memory_client = MemoryClient(
                base_url=self.memory_config["base_url"],
                timeout_seconds=self.memory_config.get("timeout_seconds", 300),
                api_key=self.memory_config.get("api_key"),
            )
        return self._memory_client

    def _resolve_repo_context(self) -> tuple[str, str | None]:
        """Resolve repo_id and revision from template vars and config."""
        repo_id = self.extra_template_vars.get("instance_id", "")
        revision = self.memory_config.get("revision")
        return repo_id, revision
