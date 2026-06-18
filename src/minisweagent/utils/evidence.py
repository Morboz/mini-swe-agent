"""Helpers for the Formsy Evidence integration.

These are the only Formsy-specific pieces that survived the migration from the
legacy ``/compile`` ``/query`` ``/read`` surface to the Evidence API (see
ADR-0004 in the Formsy repo). The agents call the vendored ``formsy_sdk.Client``
directly — there is **no wrapper client here**, only two pure helpers plus the
error type both agents catch for graceful degradation.

Why env-side extraction: the repo under test lives inside the Docker eval
container (``/testbed``); the agent process runs on the host and reaches it only
via ``env.execute`` (``docker exec``). The SDK's host-side ``ingest_directory``
therefore cannot be used — files must be extracted through the environment and
passed to ``Client.ingest(...)``.

Why a resolved ``revision``: the Evidence API requires ``revision`` in every
request and has no "latest" fallback (ADR-0002). For a SWE-bench case the
checked-out HEAD is the instance's ``base_commit`` — a stable, per-case version
label, so ``(repo_id=instance_id, revision=HEAD)`` is unique and idempotent.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from minisweagent._vendor.formsy_sdk import SourceFile

if TYPE_CHECKING:
    from minisweagent import Environment


class FormsyEvidenceError(RuntimeError):
    """Raised when source extraction or revision resolution fails.

    Agents catch this alongside ``httpx.HTTPError`` (raised by SDK calls) to
    degrade gracefully: a flaky Evidence backend must not sink a 90-minute
    SWE-bench run.
    """


# Per-cwd revision cache so we don't ``git rev-parse`` on every tool call.
# One repo per agent run; keyed by the environment's cwd.
_REVISION_CACHE: dict[str, str] = {}

_LOGGER = logging.getLogger(__name__)
# Fallback revision when /testbed is not a git checkout — some SWE-bench Pro
# images install the repo rather than checking it out, so ``git rev-parse HEAD``
# fails with "not a git repository". This is safe: repo_id (instance_id) is
# already unique per case, so ``(repo_id, "unknown")`` is still a unique Evidence.
_FALLBACK_REVISION = "unknown"


def resolve_revision(env: "Environment", *, cwd: str = "") -> str:
    """Resolve the Evidence ``revision`` for the repo under test.

    Runs ``git rev-parse HEAD`` inside the environment and caches it per ``cwd``.
    If the repo under test is not a git checkout (no ``.git``), falls back to
    :data:`_FALLBACK_REVISION` and logs a warning — never raises.
    """
    if cwd and cwd in _REVISION_CACHE:
        return _REVISION_CACHE[cwd]
    result = env.execute({"command": "git rev-parse HEAD"}, cwd=cwd)
    output = (result.get("output", "") or "").strip()
    if result.get("returncode") != 0 or not output:
        _LOGGER.warning(
            "resolve_revision: git rev-parse HEAD failed in %s (%s); "
            "falling back to revision=%r",
            cwd or "(default cwd)",
            output or "no output",
            _FALLBACK_REVISION,
        )
        revision = _FALLBACK_REVISION
    else:
        # ``git rev-parse HEAD`` prints the sha (plus any stderr warnings, which
        # land elsewhere); take the first whitespace token as the revision.
        revision = output.split()[0]
    if cwd:
        _REVISION_CACHE[cwd] = revision
    return revision


def extract_source_files(env: "Environment", *, cwd: str = "") -> list[SourceFile]:
    """Extract ``.py`` source from the repo under test as :class:`SourceFile`.

    Runs a Python snippet inside the environment (``docker exec`` into
    ``/testbed``) that walks the tree and emits ``{path, content}``. The legacy
    ``language`` / ``is_test`` fields are dropped — the Evidence ``SourceFile``
    carries only ``path`` and ``content``.
    """
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
    files.append({'path': rel, 'content': content})
print(json.dumps(files))
PY
""".strip()
    result = env.execute({"command": command}, cwd=cwd)
    if result.get("returncode") != 0:
        raise FormsyEvidenceError(
            f"extract_source_files failed: {result.get('output', 'repository extraction failed')}"
        )
    try:
        payload = json.loads(result.get("output", "[]"))
    except json.JSONDecodeError as exc:
        raise FormsyEvidenceError(f"Invalid extraction payload: {exc}") from exc
    if not isinstance(payload, list):
        raise FormsyEvidenceError("Extraction payload must be a JSON list")
    return [
        SourceFile(path=str(item["path"]), content=str(item["content"])) for item in payload
    ]
