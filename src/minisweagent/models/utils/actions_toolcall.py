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

KNOWN_TOOLS = {"bash", "context_search"}


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
