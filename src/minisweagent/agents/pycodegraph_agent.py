"""PycodeGraph agent — high-precision code search & call-graph navigation.

This agent injects two tools the LLM can call during the agent loop:

- ``search_nodes`` — fuzzy-search for code symbols (functions, classes, etc.)
  backed by pycodegraph's FTS → LIKE → fuzzy cascade.
- ``get_neighbors`` — query call-graph upstream/downstream for a given
  ``node_id`` from pycodegraph's remote PG graph (22k+ nodes for ansible).

Unlike the ContextToolAgent, this agent does NOT ingest the repo — the
pycodegraph Postgres database is pre-built and long-lived. Connection is
established lazily on the first tool call.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.models.utils.actions_toolcall import (
    GET_NEIGHBORS_TOOL,
    SEARCH_NODES_TOOL,
)

logger = logging.getLogger("minisweagent.pycodegraph_agent")
"""Use the minisweagent logger tree so warnings propagate to configured handlers."""

# ── Module-level CodeGraph singleton ──────────────────────────────────────

_cg_instance: Any = None
"""Cache the pycodegraph CodeGraph client so we don't reconnect per call."""


def _get_cg(pycodegraph_config: dict[str, Any]) -> Any:
    """Lazy-init pycodegraph CodeGraph from config dict.

    ``pycodegraph_config`` expects at minimum a ``db_url`` key.
    Returns ``None`` if pycodegraph is not installed or connection fails.
    """
    global _cg_instance
    if _cg_instance is not None:
        return _cg_instance
    db_url = pycodegraph_config.get("db_url", "")
    src_root = pycodegraph_config.get("src_root", "")
    if not db_url:
        logger.warning("pycodegraph db_url not configured — tools unavailable")
        return None
    try:
        from pycodegraph import CodeGraph

        _cg_instance = CodeGraph.open_from_url(db_url, src_root)
        logger.info("pycodegraph connected to %s", db_url.split("@")[-1] if "@" in db_url else db_url)
        return _cg_instance
    except FileNotFoundError:
        # 数据库文件或 schema 不存在，尝试 init
        logger.warning("pycodegraph database not found at %s — tools unavailable", db_url)
        return None
    except ImportError:
        logger.warning(
            "pycodegraph not installed (pip install pycodegraph) — search_nodes/get_neighbors unavailable"
        )
        return None
    except Exception as e:
        logger.warning("pycodegraph connection failed (%s) — tools degraded", e)
        # Log the full traceback for debugging
        logger.debug("pycodegraph connection error details", exc_info=True)
        return None


# ── Serialization helpers ─────────────────────────────────────────────────


def _serialize_node(node: Any) -> dict[str, Any]:
    """Convert a pycodegraph Node to a plain dict for tool output."""
    return {
        "id": node.id,
        "kind": str(node.kind) if hasattr(node, "kind") else "",
        "name": node.name,
        "qualified_name": node.qualified_name,
        "file_path": node.file_path,
        "start_line": node.start_line,
        "signature": node.signature or "",
    }


def _serialize_neighbor(node: Any, edge: Any) -> dict[str, Any]:
    """Convert a (Node, Edge) tuple to a plain dict."""
    return {
        "node": _serialize_node(node),
        "edge_kind": str(edge.kind) if hasattr(edge, "kind") else "",
        "edge_line": edge.line,
    }


# ── Agent class ───────────────────────────────────────────────────────────


class PycodeGraphAgentConfig(AgentConfig):
    """Config for PycodeGraphAgent — includes pycodegraph connection settings."""

    pycodegraph: dict[str, Any] = {}
    """Connection config (db_url, src_root, timeout_seconds, ...)."""


class PycodeGraphAgent(DefaultAgent):
    """Agent with ``search_nodes`` and ``get_neighbors`` tools.

    Backed by a pycodegraph CodeGraph client connected to a remote PG
    database containing the pre-built symbol graph for the repo under test.
    """

    def __init__(
        self,
        model,
        env,
        *,
        pycodegraph: dict[str, Any] | None = None,
        config_class: type = PycodeGraphAgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, pycodegraph=pycodegraph or {}, **kwargs)
        self.pycodegraph_config = pycodegraph or {}
        self._cg = None
        self._cg_ok = False
        self.logger = logger

        # Inject pycodegraph tools into the model's tool list.
        if hasattr(model, "config") and hasattr(model.config, "extra_tools"):
            model.config.extra_tools = [SEARCH_NODES_TOOL, GET_NEIGHBORS_TOOL]

    # -- Lazy connection + ingest --------------------------------------------------

    def query(self) -> dict:
        """Connect to pycodegraph once on the first step, then query the model."""
        if not self._cg_ok:
            self._ensure_cg()
        return super().query()

    def _ensure_cg(self) -> None:
        """Initialize the pycodegraph client exactly once and ensure the graph is ready."""
        self._cg = _get_cg(self.pycodegraph_config)
        self._cg_ok = self._cg is not None
        if self._cg_ok:
            # Try to ingest if no data exists for this repo
            self._ensure_ingested()

    def _ensure_ingested(self) -> None:
        """Check if the code graph exists; if not, index from the env (Docker container)."""
        try:
            cwd = getattr(getattr(self.env, "config", None), "cwd", "")
            repo_id = self.extra_template_vars.get("instance_id", "")

            # Quick check: search for a common symbol. If empty, repo not indexed.
            if self._cg.search("def ", limit=1):
                logger.debug("pycodegraph graph already exists — skipping ingest")
                return

            logger.info("pycodegraph: no existing graph found, indexing from %s ...", cwd or "(default cwd)")
            # Collect source files from the environment (Docker container)
            source_files = []
            result = self.env.execute({"command": "find . -name '*.py' -type f"}, cwd=cwd)
            if result.get("returncode") == 0:
                files = result["output"].strip().splitlines()
                for path in files[:1000]:  # safety limit in case of huge repos
                    path = path.strip()
                    if not path:
                        continue
                    fr = self.env.execute({"command": f"cat '{path}'"}, cwd=cwd)
                    if fr.get("returncode") == 0:
                        source_files.append({"path": path, "content": fr["output"]})

            if not source_files:
                logger.warning("pycodegraph: no source files found in env — skipping ingest")
                return

            logger.info("pycodegraph: indexing %d files ...", len(source_files))
            self._cg.index_all()
            logger.info("pycodegraph: indexing complete")
        except Exception as e:
            logger.warning("pycodegraph ingest failed (%s) — using whatever graph exists", e)

    # -- Tool dispatch ---------------------------------------------------------

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, dispatching by tool type."""
        actions = message.get("extra", {}).get("actions", [])
        outputs = []
        for action in actions:
            tool = action.get("tool", "bash")
            if tool == "bash":
                outputs.append(self.env.execute(action))
            elif tool == "search_nodes":
                outputs.append(self._execute_search_nodes(action))
            elif tool == "get_neighbors":
                outputs.append(self._execute_get_neighbors(action))
            else:
                outputs.append({
                    "output": f"Unknown tool: {tool}",
                    "returncode": 1,
                })
        return self.add_messages(
            *self.model.format_observation_messages(message, outputs, self.get_template_vars())
        )

    # -- search_nodes ----------------------------------------------------------

    def _execute_search_nodes(self, action: dict) -> dict[str, Any]:
        """Fuzzy-search code symbols by name/query."""
        query = action.get("query", "")
        limit = action.get("limit", 10)
        if not self._cg_ok:
            return {
                "output": (
                    "search_nodes unavailable (pycodegraph not connected). "
                    "Use bash (grep/find) to search the codebase instead."
                ),
                "returncode": 1,
            }
        try:
            nodes = self._cg.search(query, limit=limit)
            if not nodes:
                return {
                    "output": f"search_nodes: no results for {query!r}",
                    "returncode": 0,
                }
            serialized = [_serialize_node(n) for n in nodes]
            text = json.dumps(serialized, ensure_ascii=False, indent=2)
            return {"output": text, "returncode": 0}
        except Exception as e:
            self.logger.warning("search_nodes failed: %s", e)
            return {"output": f"search_nodes failed: {e}", "returncode": 1}

    # -- get_neighbors ---------------------------------------------------------

    def _execute_get_neighbors(self, action: dict) -> dict[str, Any]:
        """Query call-graph upstream/downstream for a node."""
        node_id = action.get("node_id", "")
        direction = action.get("direction", "both")
        max_depth = action.get("max_depth", 1)
        if not self._cg_ok:
            return {
                "output": (
                    "get_neighbors unavailable (pycodegraph not connected). "
                    "Use bash (grep/find) to explore call relationships instead."
                ),
                "returncode": 1,
            }
        try:
            parts: list[str] = []
            if direction in ("callers", "both"):
                callers = self._cg.get_callers_deep(node_id, max_depth=max_depth)
                if callers:
                    parts.append(
                        f"=== callers ({len(callers)}) ===\n"
                        + "\n".join(
                            json.dumps(_serialize_neighbor(n, e), ensure_ascii=False)
                            for n, e in callers
                        )
                    )
            if direction in ("callees", "both"):
                callees = self._cg.get_callees_deep(node_id, max_depth=max_depth)
                if callees:
                    parts.append(
                        f"=== callees ({len(callees)}) ===\n"
                        + "\n".join(
                            json.dumps(_serialize_neighbor(n, e), ensure_ascii=False)
                            for n, e in callees
                        )
                    )
            if not parts:
                return {
                    "output": f"get_neighbors: no neighbors found for {node_id!r}",
                    "returncode": 0,
                }
            return {"output": "\n\n".join(parts), "returncode": 0}
        except Exception as e:
            self.logger.warning("get_neighbors failed: %s", e)
            return {"output": f"get_neighbors failed: {e}", "returncode": 1}