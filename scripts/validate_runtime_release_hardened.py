#!/usr/bin/env python3
"""Run runtime receipt validation with replayable evidence attestation support.

The underlying validator remains the canonical receipt schema and git-binding
implementation. This entrypoint extends the evidence-only attestation boundary
to include raw parity bundles under ``parity/evidence/``. The independent replay
and source-binding audit runs separately before this command in the Makefile.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_mlx.source_binding import ATTESTATION_PATH_PREFIXES  # noqa: E402

try:  # Direct script execution.
    import validate_runtime_release as _validator  # type: ignore[no-redef]
except ModuleNotFoundError:  # Imported as scripts.validate_runtime_release_hardened.
    from scripts import validate_runtime_release as _validator  # type: ignore[no-redef]


def main() -> None:
    _validator.ATTESTATION_PATH_PREFIXES = ATTESTATION_PATH_PREFIXES
    _validator.main()


if __name__ == "__main__":
    main()
