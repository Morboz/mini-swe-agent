"""``formsy-sdk`` — Python SDK + CLI for the Evidence service.

A thin, typed HTTP client (sync ``Client`` + async ``AsyncClient``) over the
server's Evidence API, plus a ``formsy`` CLI. See ADR-0003.

Errors are raw ``httpx.HTTPStatusError`` (status code + body); the SDK does not
translate them into a typed hierarchy.

VENDORED into mini-swe-agent from the Formsy repo — see ``_vendor/README.md``.
"""

from __future__ import annotations

from minisweagent._vendor.formsy_contracts import (
    CodeSource,
    ExploreRequest,
    ExploreResponse,
    IngestRequest,
    IngestResponse,
    ReadResponse,
    SourceFile,
    StatusResponse,
)
from .client import AsyncClient, Client

__all__ = [
    "AsyncClient",
    "Client",
    # Re-exported wire contracts so consumers have one import surface.
    "CodeSource",
    "ExploreRequest",
    "ExploreResponse",
    "IngestRequest",
    "IngestResponse",
    "ReadResponse",
    "SourceFile",
    "StatusResponse",
]
