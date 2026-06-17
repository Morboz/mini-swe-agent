"""Public wire contracts for the Evidence API.

Every Pydantic type that crosses the HTTP wire as a public request or response
lives here, shared by the ``formsy`` server (its routes) and ``formsy-sdk`` (its
client). This is the single source of truth for the wire format — see ADR-0003.

Only ``code`` Evidence exists today; the ``Source`` seam widens here when issue /
PR Sources arrive. Depends on pydantic alone — **no FastAPI** — so both the server
and a lightweight SDK can import it.

VENDORED into mini-swe-agent from the Formsy repo — see ``../README.md``.
"""

from __future__ import annotations

from .models import (
    CodeSource,
    ExploreRequest,
    ExploreResponse,
    IngestRequest,
    IngestResponse,
    ReadResponse,
    SourceFile,
    StatusResponse,
)

__all__ = [
    "CodeSource",
    "ExploreRequest",
    "ExploreResponse",
    "IngestRequest",
    "IngestResponse",
    "ReadResponse",
    "SourceFile",
    "StatusResponse",
]
