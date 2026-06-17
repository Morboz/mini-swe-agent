"""Request/response models for the Evidence API (``/api/v1/evidence``).

These are the canonical wire shapes, addressed by ``(repo_id, revision)`` in the
path (revision is required — there is no "latest" resolution). See ``CONTEXT.md``
for the glossary and ADR-0002 for the resource-oriented API design.

VENDORED into mini-swe-agent from the Formsy repo — see ``_vendor/README.md``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Source — the input type of an ingest. Only ``code`` is implemented today;
# adding ``issue`` / ``pr`` later widens this (or turns ``source`` into a
# discriminated union). That evolution happens here, in the shared contract.
# ---------------------------------------------------------------------------


class SourceFile(BaseModel):
    path: str
    content: str


class CodeSource(BaseModel):
    type: Literal["code"] = "code"
    files: list[SourceFile]


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    repo_id: str
    revision: str
    source: CodeSource


class ExploreRequest(BaseModel):
    query: str
    max_files: int | None = None
    max_output_chars: int | None = None
    max_chars_per_file: int | None = None


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class IngestResponse(BaseModel):
    repo_id: str
    revision: str
    created: bool
    success: bool
    files_indexed: int
    nodes_created: int
    edges_created: int
    refs_resolved: int
    refs_unresolved: int
    duration_ms: int
    node_count: int
    edge_count: int


class StatusResponse(BaseModel):
    repo_id: str
    revision: str
    node_count: int
    edge_count: int
    file_count: int


class ExploreResponse(BaseModel):
    repo_id: str
    revision: str
    query: str
    content: str


class ReadResponse(BaseModel):
    repo_id: str
    revision: str
    path: str
    start_line: int
    end_line: int
    content: str
    truncated: bool
    total_lines: int
