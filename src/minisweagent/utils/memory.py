from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests


class MemoryError(RuntimeError):
    """Base error for memory backend integration."""


class MemoryCompileError(MemoryError):
    """Raised when memory compile fails or returns invalid data."""


class MemoryQueryError(MemoryError):
    """Raised when memory query fails or returns invalid data."""


class MemoryExtractError(MemoryError):
    """Raised when repository extraction from the environment fails."""


@dataclass
class MemoryCallResult:
    payload: dict[str, Any]
    latency_ms: int


class MemoryClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: int,
        api_key: str | None = None,
        session: requests.Session | Any | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self.headers = dict(headers) if headers else {}
        if api_key:
            self.headers.setdefault("authorization", f"Bearer {api_key}")

    def compile_repo(
        self,
        repo_id: str,
        files: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
        revision: str | None = None,
        enable_w2: bool = False,
    ) -> dict[str, Any]:
        result = self._post(
            "/api/v1/compile",
            {
                "repo_id": repo_id,
                "files": files,
                "revision": revision,
                "enable_w2": enable_w2,
                "metadata": metadata or {},
            },
            MemoryCompileError,
        )
        self._require_fields(result.payload, ("repo_id", "revision", "parsed_file_count"), MemoryCompileError)
        return result.payload | {"_latency_ms": result.latency_ms}

    def query_repo(
        self,
        repo_id: str,
        query: str,
        *,
        metadata: dict[str, Any] | None = None,
        revision: str | None = None,
        budget: int = 4000,
    ) -> dict[str, Any]:
        result = self._post(
            "/api/v1/query",
            {
                "repo_id": repo_id,
                "query": query,
                "revision": revision,
                "budget": budget,
                "metadata": metadata or {},
            },
            MemoryQueryError,
        )
        self._require_fields(result.payload, ("repo_id", "revision", "query", "extra_context"), MemoryQueryError)
        if not isinstance(result.payload["extra_context"], str):
            raise MemoryQueryError("Field 'extra_context' must be a string")
        return result.payload | {"_latency_ms": result.latency_ms}

    def _post(self, path: str, payload: dict[str, Any], error_cls: type[MemoryError]) -> MemoryCallResult:
        started = time.time()
        try:
            response = self.session.post(
                f"{self.base_url}{path}",
                json=payload,
                timeout=self.timeout_seconds,
                headers=self.headers,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise error_cls(str(exc)) from exc
        return MemoryCallResult(payload=data, latency_ms=int((time.time() - started) * 1000))

    @staticmethod
    def _require_fields(payload: dict[str, Any], fields: tuple[str, ...], error_cls: type[MemoryError]) -> None:
        for field in fields:
            if field not in payload:
                raise error_cls(f"Missing required field '{field}'")


def extract_memory_source_files(env, cwd: str = "") -> list[dict[str, Any]]:
    command = r"""
python - <<'PY'
import json
from pathlib import Path

root = Path('.').resolve()
allowed = {'.py'}
excluded = {'.git', 'node_modules', 'vendor', 'dist', 'build', '.venv', '__pycache__'}
files = []

for path in root.rglob('*'):
    if not path.is_file():
        continue
    if any(part in excluded for part in path.parts):
        continue
    if path.suffix.lower() not in allowed:
        continue
    rel = path.relative_to(root).as_posix()
    with path.open(encoding='utf-8', errors='ignore') as handle:
        content = handle.read()
    files.append({
        'path': rel,
        'content': content,
        'language': path.suffix.lower().lstrip('.') or 'text',
        'is_test': (
            rel.startswith('tests/')
            or rel.startswith('test/')
            or '/tests/' in f'/{rel}/'
            or rel.endswith('_test.py')
            or rel.endswith('_spec.js')
            or rel.endswith('.test.js')
        ),
    })

print(json.dumps(files))
PY
""".strip()
    result = env.execute({"command": command}, cwd=cwd)
    if result.get("returncode") != 0:
        raise MemoryExtractError(result.get("output", "repository extraction failed"))
    try:
        payload = json.loads(result.get("output", "[]"))
    except json.JSONDecodeError as exc:
        raise MemoryExtractError(f"Invalid extraction payload: {exc}") from exc
    if not isinstance(payload, list):
        raise MemoryExtractError("Extraction payload must be a JSON list")
    return payload
