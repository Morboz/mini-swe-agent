# Vendored packages

This directory holds **source copies** of packages whose canonical home is the
Formsy repository. They are vendored (not installed as wheels) so that
`mini-swe-agent` can depend on them without the cross-repo release dance — see
**ADR-0004** in the Formsy repo (`docs/adr/0004-miniswe-agent-vendored-sdk-integration.md`).

| Package            | Canonical source (Formsy repo)        | Purpose                                  |
| ------------------ | ------------------------------------- | ---------------------------------------- |
| `formsy_contracts` | `contracts/src/formsy_contracts/`     | Pydantic wire models for the Evidence API |
| `formsy_sdk`       | `sdk/src/formsy_sdk/`                 | Sync/async HTTP client for the Evidence API |

## Re-copy procedure (on Formsy SDK upgrade)

1. Copy the `.py` files from the Formsy repo paths above into the matching
   directories here. **Do not** copy `pyproject.toml` — packaging is handled by
   `mini-swe-agent`'s own `pyproject.toml`.
2. Fix imports so the vendored copies resolve under `minisweagent._vendor`:
   - **Cross-package** (SDK → contracts): in
     `formsy_sdk/__init__.py` and `formsy_sdk/client.py`, rewrite
     `from formsy_contracts import ...` →
     `from minisweagent._vendor.formsy_contracts import ...`.
   - **Intra-package**: prefer relative imports (`from .models import ...`,
     `from .client import ...`, `from ._files import ...`) so future re-copies
     need fewer edits.
3. Bump the version note below if the upstream SDK changed its wire shape.

## Drift

There is **no automated drift detection** today. The Formsy server is the source
of truth for the wire format; if it ships a breaking Evidence API change, this
vendored copy must be re-copied in lockstep. A contract test pinning these models
against the server is a candidate follow-up.

_Last vendored from: Formsy `feat/evidence-api` branch (Evidence API +
formsy-sdk, ADR-0002/0003)._
