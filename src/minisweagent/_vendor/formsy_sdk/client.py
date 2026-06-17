"""Sync and async HTTP clients for the Evidence API.

Both clients share request/response building and expose identical method names;
the only difference is the httpx transport (``httpx.Client`` vs
``httpx.AsyncClient``). Methods take plain kwargs and return typed
``formsy_contracts`` models.

Design (ADR-0003):

- ``base_url`` defaults from ``$FORMSY_BASE_URL`` then ``http://localhost:8000``.
- ``auth`` is pluggable (an httpx ``Auth`` / headers); defaults to none — the
  server has no auth today.
- ``timeout`` defaults to 300s: ``ingest`` runs pycodegraph's ``index_all`` and
  can take minutes; override per call with ``timeout=``.
- Errors are **raw** ``httpx.HTTPStatusError`` (``raise_for_status``); 409 (busy)
  / 404 (not found) surface as their status codes + the server's ``detail`` body.
- ``AsyncClient(app=...)`` mounts an ``httpx.ASGITransport`` so the client can be
  driven against an in-process FastAPI app in tests (no socket). The sync
  ``Client`` has no ``app=`` (ASGI is async); it accepts ``transport=`` (e.g.
  ``httpx.MockTransport``).

VENDORED into mini-swe-agent from the Formsy repo — see ``_vendor/README.md``.
Only the imports were adjusted (intra-package made relative; the cross-package
contracts import made absolute under ``minisweagent._vendor``).
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import httpx

from minisweagent._vendor.formsy_contracts import (
    ExploreResponse,
    IngestResponse,
    ReadResponse,
    SourceFile,
    StatusResponse,
)
from ._files import read_directory

_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_TIMEOUT = 300.0
_API = "/api/v1/evidence"


# ---------------------------------------------------------------------------
# Shared payload/path builders (pure).
# ---------------------------------------------------------------------------


def _resolve_base_url(base_url: str | None, app_provided: bool) -> str:
    if base_url is not None:
        return base_url
    if app_provided:
        return "http://test"
    return os.environ.get("FORMSY_BASE_URL", _DEFAULT_BASE_URL)


def _normalize_files(
    files: Iterable[SourceFile | tuple[str, str] | Mapping[str, Any]],
) -> list[SourceFile]:
    out: list[SourceFile] = []
    for f in files:
        if isinstance(f, SourceFile):
            out.append(f)
        elif isinstance(f, tuple):
            out.append(SourceFile(path=f[0], content=f[1]))
        else:
            out.append(SourceFile.model_validate(f))
    return out


def _ingest_body(
    repo_id: str,
    revision: str,
    files: Iterable[SourceFile | tuple[str, str] | Mapping[str, Any]],
    source_type: str,
) -> dict[str, Any]:
    return {
        "repo_id": repo_id,
        "revision": revision,
        "source": {
            "type": source_type,
            "files": [f.model_dump() for f in _normalize_files(files)],
        },
    }


def _explore_body(
    query: str,
    max_files: int | None,
    max_output_chars: int | None,
    max_chars_per_file: int | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"query": query}
    for key, val in (
        ("max_files", max_files),
        ("max_output_chars", max_output_chars),
        ("max_chars_per_file", max_chars_per_file),
    ):
        if val is not None:
            body[key] = val
    return body


def _read_params(
    start_line: int, end_line: int | None, max_lines: int
) -> dict[str, Any]:
    params: dict[str, Any] = {"start_line": start_line, "max_lines": max_lines}
    if end_line is not None:
        params["end_line"] = end_line
    return params


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


class AsyncClient:
    """Async HTTP client for the Evidence API. Use as an async context manager."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        auth: httpx.Auth | tuple[str, str] | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
        app: object | None = None,
    ) -> None:
        if app is not None:
            transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
        self._http = httpx.AsyncClient(
            base_url=_resolve_base_url(base_url, app is not None),
            auth=auth,
            timeout=timeout,
            transport=transport,
        )

    async def ingest(
        self,
        repo_id: str,
        revision: str,
        files: Iterable[SourceFile | tuple[str, str] | Mapping[str, Any]],
        *,
        source_type: str = "code",
    ) -> IngestResponse:
        resp = await self._http.post(
            _API, json=_ingest_body(repo_id, revision, files, source_type)
        )
        resp.raise_for_status()
        return IngestResponse.model_validate(resp.json())

    async def ingest_directory(
        self,
        repo_id: str,
        revision: str,
        dir_path: str | Path,
        *,
        suffixes: Iterable[str] | None = (".py",),
        ignore: Iterable[str] | None = None,
        source_type: str = "code",
    ) -> IngestResponse:
        files = read_directory(Path(dir_path), suffixes=suffixes, ignore=ignore)
        return await self.ingest(repo_id, revision, files, source_type=source_type)

    async def get_status(self, repo_id: str, revision: str) -> StatusResponse:
        resp = await self._http.get(f"{_API}/{repo_id}/{revision}")
        resp.raise_for_status()
        return StatusResponse.model_validate(resp.json())

    async def explore(
        self,
        repo_id: str,
        revision: str,
        query: str,
        *,
        max_files: int | None = None,
        max_output_chars: int | None = None,
        max_chars_per_file: int | None = None,
    ) -> ExploreResponse:
        resp = await self._http.post(
            f"{_API}/{repo_id}/{revision}/explore",
            json=_explore_body(query, max_files, max_output_chars, max_chars_per_file),
        )
        resp.raise_for_status()
        return ExploreResponse.model_validate(resp.json())

    async def read_file(
        self,
        repo_id: str,
        revision: str,
        path: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
        max_lines: int = 400,
    ) -> ReadResponse:
        resp = await self._http.get(
            f"{_API}/{repo_id}/{revision}/files/{path}",
            params=_read_params(start_line, end_line, max_lines),
        )
        resp.raise_for_status()
        return ReadResponse.model_validate(resp.json())

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------


class Client:
    """Sync HTTP client for the Evidence API. Use as a context manager.

    Unlike :class:`AsyncClient`, this has no ``app=`` (ASGI is async-only); pass
    ``transport=`` (e.g. ``httpx.MockTransport``) for tests.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        auth: httpx.Auth | tuple[str, str] | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = httpx.Client(
            base_url=_resolve_base_url(base_url, False),
            auth=auth,
            timeout=timeout,
            transport=transport,
        )

    def ingest(
        self,
        repo_id: str,
        revision: str,
        files: Iterable[SourceFile | tuple[str, str] | Mapping[str, Any]],
        *,
        source_type: str = "code",
    ) -> IngestResponse:
        resp = self._http.post(
            _API, json=_ingest_body(repo_id, revision, files, source_type)
        )
        resp.raise_for_status()
        return IngestResponse.model_validate(resp.json())

    def ingest_directory(
        self,
        repo_id: str,
        revision: str,
        dir_path: str | Path,
        *,
        suffixes: Iterable[str] | None = (".py",),
        ignore: Iterable[str] | None = None,
        source_type: str = "code",
    ) -> IngestResponse:
        files = read_directory(Path(dir_path), suffixes=suffixes, ignore=ignore)
        return self.ingest(repo_id, revision, files, source_type=source_type)

    def get_status(self, repo_id: str, revision: str) -> StatusResponse:
        resp = self._http.get(f"{_API}/{repo_id}/{revision}")
        resp.raise_for_status()
        return StatusResponse.model_validate(resp.json())

    def explore(
        self,
        repo_id: str,
        revision: str,
        query: str,
        *,
        max_files: int | None = None,
        max_output_chars: int | None = None,
        max_chars_per_file: int | None = None,
    ) -> ExploreResponse:
        resp = self._http.post(
            f"{_API}/{repo_id}/{revision}/explore",
            json=_explore_body(query, max_files, max_output_chars, max_chars_per_file),
        )
        resp.raise_for_status()
        return ExploreResponse.model_validate(resp.json())

    def read_file(
        self,
        repo_id: str,
        revision: str,
        path: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
        max_lines: int = 400,
    ) -> ReadResponse:
        resp = self._http.get(
            f"{_API}/{repo_id}/{revision}/files/{path}",
            params=_read_params(start_line, end_line, max_lines),
        )
        resp.raise_for_status()
        return ReadResponse.model_validate(resp.json())

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
