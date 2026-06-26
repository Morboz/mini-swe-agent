"""Observability backend abstraction for mini-swe-agent.

Concrete implementations
------------------------
* ``NoopBackend`` — all methods are no-ops (used when observability is disabled).
* ``LangfuseBackend`` — Langfuse SDK v4 via ``get_client()`` and OTel context
  managers.
"""

from __future__ import annotations

import contextlib
import logging
from abc import ABC, abstractmethod
from typing import Any

import litellm

logger = logging.getLogger("observability")


# ---------------------------------------------------------------------------
# Public helpers (consumed by litellm_model.py)
# ---------------------------------------------------------------------------

_LANGFUSE_INITIALIZED = False


def init_langfuse() -> None:
    """One-time initialisation of the Langfuse client and the litellm OTel callback.

    Idempotent and fail-safe: if ``langfuse`` isn't installed or can't
    initialise, tracing is silently disabled and LLM calls proceed normally.
    """
    global _LANGFUSE_INITIALIZED
    if _LANGFUSE_INITIALIZED:
        return
    _LANGFUSE_INITIALIZED = True
    try:
        from langfuse import get_client

        get_client()  # initialise the singleton – reads LANGFUSE_* env vars
    except ImportError:
        logger.warning(
            "Langfuse tracing enabled (MSWEA_LANGFUSE set) but the 'langfuse' package is not "
            "installed. Install it (`uv pip install langfuse`) or unset MSWEA_LANGFUSE."
        )
        return
    except Exception as e:
        logger.warning(f"Failed to initialise Langfuse client: {e}")
        return

    # Wire up litellm so every ``query()`` call is auto-instrumented as a
    # ``generation`` observation nested under the current OTel context.
    litellm.callbacks = ["langfuse_otel"]


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class ObservabilityBackend(ABC):
    """Abstract interface for observability backends (Langfuse, etc.).

    The agent's run loop and tool executor call these methods via duck-typing
    (``getattr`` / ``hasattr``), so the interface is informal.  This ABC exists
    as documentation and to ensure new backends don't miss methods.
    """

    # -- Trace lifecycle ----------------------------------------------------------

    @abstractmethod
    def start_run_trace(self, *, tags: dict[str, Any] | None = None) -> None:
        """Begin a trace that spans one agent run."""

    @abstractmethod
    def end_run_trace(self, *, end_state: str = "Success") -> None:
        """End the current run trace."""

    # -- Agent span lifecycle -----------------------------------------------------

    @abstractmethod
    def start_run_span(
        self, *, name: str = "agent.run", attributes: dict[str, Any] | None = None
    ) -> Any:
        """Open a span around the run loop.  Returns an opaque handle for
        :meth:`end_run_span`, or ``None`` when the backend is disabled.

        The span must be set as the active OTel context so that child spans
        (LLM calls, tool calls) nest under it.
        """

    @abstractmethod
    def end_run_span(
        self, handle: Any, *, end_state: str = "Success", error: str | None = None
    ) -> None:
        """Close a span opened by :meth:`start_run_span`."""

    # -- Tool span lifecycle ------------------------------------------------------

    @contextlib.contextmanager
    @abstractmethod
    def tool_span(self, name: str, *, inputs: dict[str, Any] | None = None):
        """Context manager that yields a tool span.  No-op (yields ``None``)
        when the backend is disabled so callers don't branch."""

    @abstractmethod
    def start_tool_span(
        self, *, name: str, inputs: dict[str, Any] | None = None
    ) -> Any:
        """Open a tool span.  Returns ``(cm, span)`` or ``None``."""

    @abstractmethod
    def _end_tool_span(
        self,
        handle: Any,
        *,
        end_state: str = "Success",
        error: str | None = None,
        output: Any = None,
    ) -> None:
        """Close a tool span opened by :meth:`start_tool_span`."""


# ---------------------------------------------------------------------------
# Noop backend (default – observability disabled)
# ---------------------------------------------------------------------------


class NoopBackend(ObservabilityBackend):
    """Backend that does nothing.  Used when ``MSWEA_LANGFUSE`` is not set."""

    def start_run_trace(self, *, tags: dict[str, Any] | None = None) -> None:
        pass

    def end_run_trace(self, *, end_state: str = "Success") -> None:
        pass

    def start_run_span(
        self, *, name: str = "agent.run", attributes: dict[str, Any] | None = None
    ) -> Any:
        return None

    def end_run_span(
        self, handle: Any, *, end_state: str = "Success", error: str | None = None
    ) -> None:
        pass

    @contextlib.contextmanager
    def tool_span(self, name: str, *, inputs: dict[str, Any] | None = None):
        """Yield ``None`` so callers don't branch."""
        try:
            yield None
        finally:
            pass

    def start_tool_span(
        self, *, name: str, inputs: dict[str, Any] | None = None
    ) -> Any:
        return None

    def _end_tool_span(
        self,
        handle: Any,
        *,
        end_state: str = "Success",
        error: str | None = None,
        output: Any = None,
    ) -> None:
        pass


# ---------------------------------------------------------------------------
# Langfuse backend
# ---------------------------------------------------------------------------


class LangfuseBackend(ObservabilityBackend):
    """Langfuse SDK v4 backend using ``get_client()`` and OTel context managers.

    Expects ``LANGFUSE_SECRET_KEY``, ``LANGFUSE_PUBLIC_KEY``, and
    ``LANGFUSE_BASE_URL`` in the environment (read automatically by the SDK).

    Because Langfuse/OTel relies on **context propagation** for nesting, the
    backend uses manual ``start_observation()`` + ``.end()`` for tool spans
    (which outlive the call that creates them), and the context-manager-based
    ``start_as_current_observation()`` for the run trace / agent span.
    """

    def __init__(self) -> None:
        self._run_trace_cm: object | None = None
        self._run_span_handle: object | None = None

    # -- helpers ----------------------------------------------------------------

    @staticmethod
    def _client():
        from langfuse import get_client

        return get_client()

    # -- Trace lifecycle --------------------------------------------------------

    def start_run_trace(self, *, tags: dict[str, Any] | None = None) -> None:
        """Create the root observation of the agent run (a Langfuse trace).

        Tags such as ``instance_id`` are set via ``propagate_attributes`` so
        they appear on every child observation.
        """
        try:
            lf = self._client()
            from langfuse import propagate_attributes

            attrs = dict(tags) if tags else {}
            trace_name = attrs.pop("trace_name", "mini-swe-agent run")
            lf_attrs = {
                k: v
                for k, v in attrs.items()
                if k
                in (
                    "user_id",
                    "session_id",
                    "metadata",
                    "version",
                    "tags",
                    "trace_name",
                )
            }
            self._run_trace_cm = lf.start_as_current_observation(
                as_type="span",
                name=trace_name,
                input={"tags": tags} if tags else None,
            )
            self._run_trace_cm.__enter__()
            if lf_attrs:
                self._attr_cm = propagate_attributes(**lf_attrs)
                self._attr_cm.__enter__()
            else:
                self._attr_cm = None
        except Exception as e:
            logger.debug("Langfuse start_run_trace failed: %s", e)
            self._run_trace_cm = None

    def end_run_trace(self, *, end_state: str = "Success") -> None:
        if self._run_trace_cm is None:
            return
        try:
            if self._attr_cm is not None:
                self._attr_cm.__exit__(None, None, None)
            self._run_trace_cm.__exit__(None, None, None)
        except Exception as e:
            logger.debug("Langfuse end_run_trace failed: %s", e)
        finally:
            self._run_trace_cm = None
            self._attr_cm = None
            try:
                self._client().flush()
            except Exception:
                pass

    # -- Agent span lifecycle ----------------------------------------------------

    def start_run_span(
        self, *, name: str = "agent.run", attributes: dict[str, Any] | None = None
    ) -> Any:
        """Open a child span under the run trace using ``start_as_current_observation``.

        Returns the OTel context manager / span tuple so the caller can close it.
        """
        try:
            lf = self._client()
            cm = lf.start_as_current_observation(
                as_type="span",
                name=name,
                input=attributes,
            )
            span = cm.__enter__()
            return (cm, span)
        except Exception as e:
            logger.debug("Langfuse start_run_span failed: %s", e)
            return None

    def end_run_span(
        self, handle: Any, *, end_state: str = "Success", error: str | None = None
    ) -> None:
        if handle is None:
            return
        cm, span = handle
        try:
            if error is not None:
                span.update(output={"error": error, "end_state": end_state})
            else:
                span.update(output={"end_state": end_state})
            cm.__exit__(None, None, None)
        except Exception as e:
            logger.debug("Langfuse end_run_span failed: %s", e)
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass

    # -- Tool span lifecycle -----------------------------------------------------

    @contextlib.contextmanager
    def tool_span(self, name: str, *, inputs: dict[str, Any] | None = None):
        handle = self.start_tool_span(name=name, inputs=inputs)
        _span = handle[1] if handle is not None else None
        try:
            yield _span
        except Exception as e:
            if handle is not None:
                self._end_tool_span(handle, end_state="Error", error=str(e), output=None)
            raise
        else:
            if handle is not None:
                self._end_tool_span(handle, end_state="Success", error=None, output=None)

    def start_tool_span(
        self, *, name: str, inputs: dict[str, Any] | None = None
    ) -> Any:
        """Start a tool span using ``start_as_current_observation`` (manual style).

        Returns ``(cm, span)`` or ``None``.
        """
        try:
            lf = self._client()
            cm = lf.start_as_current_observation(
                as_type="tool",
                name=name,
                input=inputs,
            )
            span = cm.__enter__()
            return (cm, span)
        except Exception as e:
            logger.debug("Langfuse start_tool_span failed: %s", e)
            return None

    def _end_tool_span(
        self,
        handle: Any,
        *,
        end_state: str = "Success",
        error: str | None = None,
        output: Any = None,
    ) -> None:
        if handle is None:
            return
        cm, span = handle
        try:
            span.update(
                output={"end_state": end_state, "error": error, "result": output}
            )
            cm.__exit__(None, None, None)
        except Exception as e:
            logger.debug("Langfuse _end_tool_span failed: %s", e)
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass