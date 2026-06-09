from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests


class FormsyMemoryError(RuntimeError):
    """Base error for Formsy memory backend integration."""


class MemoryCompileError(FormsyMemoryError):
    """Raised when memory compile fails or returns invalid data."""


class MemoryQueryError(FormsyMemoryError):
    """Raised when memory query fails or returns invalid data."""


class MemoryExtractError(FormsyMemoryError):
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
        # Compile cache keyed by (repo_id, revision) to support multi-repo usage
        self._compile_cache: dict[tuple[str, str | None], dict[str, Any]] = {}
        self._compile_lock = threading.Lock()
        # Per-key events for serializing concurrent compiles
        self._compile_events: dict[str, threading.Event] = {}

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

    def read_repo(
        self,
        repo_id: str,
        path: str,
        *,
        revision: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """Read source content from the Formsy context store by path and optional line range."""
        payload: dict[str, Any] = {
            "repo_id": repo_id,
            "revision": revision or "latest",
            "path": path,
        }
        if start_line is not None:
            payload["start_line"] = start_line
        if end_line is not None:
            payload["end_line"] = end_line
        result = self._post("/api/v1/read", payload, MemoryQueryError)
        return result.payload | {"_latency_ms": result.latency_ms}

    def compile_status(
        self,
        repo_id: str,
        revision: str | None = None,
    ) -> dict[str, Any] | None:
        """Check the compile status for a repo. Returns status dict or None if unavailable."""
        try:
            result = self._post(
                "/api/v1/compile/status",
                {"repo_id": repo_id, "revision": revision or "latest"},
                MemoryCompileError,
            )
            data = result.payload
            return data if data and len(data) > 0 else None
        except Exception:
            return None

    def ensure_compiled(
        self,
        repo_id: str,
        files: list[dict[str, Any]],
        query: str,
        *,
        revision: str | None = None,
        enable_w2: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ensure the repository is compiled before querying.

        Ported from plugin-opencode runtime.ts ensureMemoryCompiled logic.
        Checks local cache -> server status -> compiles if needed.
        Thread-safe: concurrent calls for the same repo+revision are serialized
        via per-key events.

        Returns compile result metadata dict.
        """
        query_signature = self._query_signature(query)
        cache_key = (repo_id, revision)

        # 1. Check local cache (fast path, no lock needed for read-only dict access)
        cached = self._compile_cache.get(cache_key)
        if cached:
            cached_sig = cached.get("query_signature")
            if cached_sig == "*" or cached_sig == query_signature:
                return cached

        # 2. Check server-side status
        status = self.compile_status(repo_id, revision)
        if status and self._existing_compile_satisfies(status, query):
            entry = self._compile_cache_from_status(status, repo_id, revision, query_signature)
            self._compile_cache[cache_key] = entry
            return entry

        # 3. Serialize concurrent compiles for the same repo+revision
        lock_key = f"{repo_id}:{revision or 'latest'}"
        our_event = threading.Event()

        with self._compile_lock:
            existing_event = self._compile_events.get(lock_key)
            if existing_event and not existing_event.is_set():
                # Another compile is in progress — wait for it
                pass
            else:
                # We are the one to compile
                self._compile_events[lock_key] = our_event

        # If another compile was in progress, wait for it then re-check
        if existing_event and not existing_event.is_set():
            existing_event.wait(timeout=self.timeout_seconds)
            # Re-check cache after waiting
            cached = self._compile_cache.get(cache_key)
            if cached:
                cached_sig = cached.get("query_signature")
                if cached_sig == "*" or cached_sig == query_signature:
                    return cached
            # If still not satisfied, fall through to compile ourselves
            with self._compile_lock:
                self._compile_events[lock_key] = our_event

        try:
            # Re-check after acquiring slot — another thread may have compiled
            cached = self._compile_cache.get(cache_key)
            if cached:
                cached_sig = cached.get("query_signature")
                if cached_sig == "*" or cached_sig == query_signature:
                    return cached

            server_status = self.compile_status(repo_id, revision)
            if server_status and self._existing_compile_satisfies(server_status, query):
                entry = self._compile_cache_from_status(
                    server_status, repo_id, revision, query_signature
                )
                self._compile_cache[cache_key] = entry
                return entry

            # 4. Actually compile
            result = self.compile_repo(
                repo_id,
                files,
                metadata=metadata or {},
                revision=revision,
                enable_w2=enable_w2,
            )
            entry = {
                "repo_id": repo_id,
                "revision": result.get("revision", revision),
                "query_signature": "*",  # full compile covers all queries
                "compile_success": True,
                "_latency_ms": result.get("_latency_ms", 0),
            }
            self._compile_cache[cache_key] = entry
            return entry
        finally:
            our_event.set()
            with self._compile_lock:
                if self._compile_events.get(lock_key) is our_event:
                    self._compile_events.pop(lock_key, None)

    def is_compile_cached(self, repo_id: str, query: str, revision: str | None = None) -> bool:
        """Public accessor: check if a compile is already cached for this repo+query."""
        cache_key = (repo_id, revision)
        cached = self._compile_cache.get(cache_key)
        if not cached:
            return False
        cached_sig = cached.get("query_signature")
        query_sig = self._query_signature(query)
        return cached_sig == "*" or cached_sig == query_sig

    @staticmethod
    def _query_signature(query: str) -> str:
        """Compute a SHA-256 signature for a normalized query string."""
        normalized = " ".join(query.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    @staticmethod
    def _existing_compile_satisfies(status: dict[str, Any], query: str) -> bool:
        """Check whether an existing server compile satisfies the given query."""
        metadata = status.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        # Full-scope compile satisfies everything
        if str(metadata.get("source_scope", "")).strip().lower() == "full":
            return True

        # Large non-query-bounded compile also satisfies
        parsed_count = int(status.get("parsed_file_count", 0) or 0)
        profile = str(metadata.get("compile_profile", "")).strip().lower()
        looks_query_bounded = bool(
            profile == "interactive_context_search"
            or metadata.get("query")
            or metadata.get("source_file_count")
        )
        if not looks_query_bounded and parsed_count > 260:
            return True

        # Query signature match
        sig = str(metadata.get("query_signature", "")).strip()
        if sig and sig == MemoryClient._query_signature(query):
            return True

        # Exact query text match
        previous_query = str(metadata.get("query", "")).lower().split()
        current_query = query.lower().split()
        if previous_query and previous_query == current_query:
            return True

        return False

    @staticmethod
    def _compile_cache_from_status(
        status: dict[str, Any],
        repo_id: str,
        revision: str | None,
        fallback_query_signature: str,
    ) -> dict[str, Any]:
        """Build a compile cache entry from a server status response."""
        metadata = status.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        status_revision = str(status.get("revision", "")).strip() or revision

        if str(metadata.get("source_scope", "")).strip().lower() == "full":
            query_signature = "*"
        else:
            profile = str(metadata.get("compile_profile", "")).strip().lower()
            parsed_count = int(status.get("parsed_file_count", 0) or 0)
            if profile != "interactive_context_search" and parsed_count > 260:
                query_signature = "*"
            else:
                sig = str(metadata.get("query_signature", "")).strip()
                query_signature = sig if sig else fallback_query_signature

        return {
            "repo_id": repo_id,
            "revision": status_revision,
            "query_signature": query_signature,
            "compile_success": True,
        }

    def _post(self, path: str, payload: dict[str, Any], error_cls: type[FormsyMemoryError]) -> MemoryCallResult:
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
    def _require_fields(payload: dict[str, Any], fields: tuple[str, ...], error_cls: type[FormsyMemoryError]) -> None:
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
