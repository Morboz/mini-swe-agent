"""Filesystem helpers — read a repo directory into ``(path, content)`` pairs.

Used by ``Client.ingest_directory`` / ``AsyncClient.ingest_directory`` and by the
``formsy ingest`` CLI subcommand.

VENDORED into mini-swe-agent from the Formsy repo — see ``_vendor/README.md``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

# Directories never worth ingesting. Mirrors the conventions of the original
# end-to-end verification script; callers may extend via ``ignore=``.
DEFAULT_IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".codegraph",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".import_linter_cache",
        ".claude",
        "dist",
        "build",
        ".tox",
        ".eggs",
    }
)


def read_directory(
    root: Path,
    *,
    suffixes: Iterable[str] | None = (".py",),
    ignore: Iterable[str] | None = None,
) -> list[tuple[str, str]]:
    """Walk ``root`` and return ``[(relative_path, content), ...]``.

    ``suffixes`` filters by file extension (``None`` = all files; default
    ``(".py",)`` since Evidence today is a pycodegraph code graph). ``ignore``
    extends :data:`DEFAULT_IGNORE_DIRS`. Unreadable files are skipped.
    """
    accepted = set(suffixes) if suffixes is not None else None
    ignored = set(DEFAULT_IGNORE_DIRS)
    if ignore is not None:
        ignored |= set(ignore)

    files: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if accepted is not None and path.suffix not in accepted:
            continue
        if any(part in ignored for part in path.parts):
            continue
        try:
            content = path.read_text()
        except OSError:
            continue
        files.append((str(path.relative_to(root)), content))
    return files
