"""Parse actions & format observations with toolcalls"""

import json
import time

from jinja2 import StrictUndefined, Template

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}

CONTEXT_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "context_search",
        "description": (
            "PRIMARY TOOL — call FIRST for almost any question OR before an edit: how does X work, "
            "architecture, a bug, where/what is X, surveying an area, or the symbols you are about to change. "
            "Returns the verbatim, line-numbered source of the relevant symbols grouped by file in ONE capped call "
            "(Read-equivalent — treat the shown source as already read; do NOT re-open those files with cat/sed/head), "
            "plus the call path among them and any prior observations relevant to the query. "
            "Query can be a natural-language question OR a bag of symbol/file names. "
            "Usually the ONLY call you need — more accurate context, in far fewer tokens and round-trips than a grep/Read loop."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Symbol names, file names, or short code terms to explore "
                        '(e.g., "AuthService loginUser session-manager", "GraphTraverser BFS impact traversal.ts"). '
                        'For a flow question, name the symbols spanning the flow (e.g. "mutateElement renderScene"). '
                        "A natural-language question works too — no prior search needed."
                    ),
                },
                "budget": {
                    "type": "integer",
                    "description": "Budget for the search (default 4000)",
                },
            },
            "required": ["query"],
        },
    },
}

SEARCH_NODES_TOOL = {
    "type": "function",
    "function": {
        "name": "search_nodes",
        "description": (
            "搜索代码库中的函数、类、变量等节点。支持模糊/自然语言查询。"
            "优先用此工具定位感兴趣的代码符号，再配合 get_neighbors 查调用上下游。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "搜索关键词，支持自然语言或部分符号名。"
                        '例如 "gzip decompress urls.py" 或 "fetch_url" 或 "Request.open"。'
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "最多返回结果数，默认 10。",
                },
            },
            "required": ["query"],
        },
    },
}

GET_NEIGHBORS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_neighbors",
        "description": (
            "查询指定函数/类的调用上下游（谁调用了我、我调用了谁）。"
            "先通过 search_nodes 拿到 node_id 后再调用此工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "节点的 ID（来自 search_nodes 返回的 id 字段）。",
                },
                "direction": {
                    "type": "string",
                    "enum": ["callers", "callees", "both"],
                    "description": "查询方向: callers（谁调用了我）、callees（我调用了谁）、both（两者）。",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "调用链深度，默认 1。深度越大返回信息越多。",
                },
            },
            "required": ["node_id", "direction"],
        },
    },
}

KNOWN_TOOLS = {"bash", "context_search", "search_nodes", "get_neighbors"}


def parse_toolcall_actions(tool_calls: list, *, format_error_template: str) -> list[dict]:
    """Parse tool calls from the response. Raises FormatError if unknown tool or invalid args."""
    if not tool_calls:
        raise FormatError(
            {
                "role": "user",
                "content": Template(format_error_template, undefined=StrictUndefined).render(
                    error="No tool calls found in the response. Every response MUST include at least one tool call.",
                    actions=[],
                ),
                "extra": {"interrupt_type": "FormatError"},
            }
        )
    actions = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        error_msg = ""
        args = {}
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as e:
            error_msg = f"Error parsing tool call arguments: {e}."

        if tool_name not in KNOWN_TOOLS:
            error_msg += f"Unknown tool '{tool_name}'."

        if not error_msg:
            if tool_name == "bash":
                if not isinstance(args, dict) or "command" not in args:
                    error_msg += "Missing 'command' argument in bash tool call."
            elif tool_name == "context_search":
                if not isinstance(args, dict) or "query" not in args:
                    error_msg += "Missing 'query' argument in context_search tool call."
            elif tool_name == "search_nodes":
                if not isinstance(args, dict) or "query" not in args:
                    error_msg += "Missing 'query' argument in search_nodes tool call."
            elif tool_name == "get_neighbors":
                if not isinstance(args, dict) or "node_id" not in args or "direction" not in args:
                    error_msg += "Missing 'node_id' or 'direction' argument in get_neighbors tool call."
                elif args.get("direction") not in ("callers", "callees", "both"):
                    error_msg += "Invalid 'direction' argument in get_neighbors tool call. Must be 'callers', 'callees', or 'both'."

        if error_msg:
            raise FormatError(
                {
                    "role": "user",
                    "content": Template(format_error_template, undefined=StrictUndefined).render(
                        actions=[], error=error_msg.strip()
                    ),
                    "extra": {"interrupt_type": "FormatError"},
                }
            )

        action: dict = {"tool": tool_name, "tool_call_id": tool_call.id}
        if tool_name == "bash":
            action["command"] = args["command"]
        elif tool_name == "context_search":
            action["query"] = args["query"]
            if "budget" in args:
                action["budget"] = args["budget"]
        elif tool_name == "search_nodes":
            action["query"] = args["query"]
            if "limit" in args:
                action["limit"] = int(args["limit"])
        elif tool_name == "get_neighbors":
            action["node_id"] = args["node_id"]
            action["direction"] = args["direction"]
            if "max_depth" in args:
                action["max_depth"] = int(args["max_depth"])
        actions.append(action)
    return actions


def format_toolcall_observation_messages(
    *,
    actions: list[dict],
    outputs: list[dict],
    observation_template: str,
    template_vars: dict | None = None,
    multimodal_regex: str = "",
) -> list[dict]:
    """Format execution outputs into tool result messages."""
    not_executed = {"output": "", "returncode": -1, "exception_info": "action was not executed"}
    padded_outputs = outputs + [not_executed] * (len(actions) - len(outputs))
    results = []
    for action, output in zip(actions, padded_outputs):
        # Ensure all fields referenced by observation templates exist.
        # Non-bash tools (e.g. context_search) may not set these.
        render_output = {
            "output": output.get("output", ""),
            "returncode": output.get("returncode", 0),
            "exception_info": output.get("exception_info", None),
        }
        content = Template(observation_template, undefined=StrictUndefined).render(
            output=render_output, **(template_vars or {})
        )
        msg = {
            "content": content,
            "extra": {
                "raw_output": output.get("output", ""),
                "returncode": output.get("returncode"),
                "timestamp": time.time(),
                "exception_info": output.get("exception_info"),
                **output.get("extra", {}),
            },
        }
        if "tool_call_id" in action:
            msg["tool_call_id"] = action["tool_call_id"]
            msg["role"] = "tool"
        else:
            msg["role"] = "user"  # human issued commands
        if multimodal_regex:
            msg = expand_multimodal_content(msg, pattern=multimodal_regex)
        results.append(msg)
    return results
