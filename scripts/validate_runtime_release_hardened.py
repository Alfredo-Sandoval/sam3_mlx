#!/usr/bin/env python3
"""Run the runtime receipt validator with replayable evidence attestation support.

The underlying validator remains the canonical schema and source-binding
implementation. This entrypoint extends the receipt-only attestation boundary to
include raw parity evidence under ``parity/evidence/``. The independent replay
auditor runs separately before this command in the release Makefile target.
"""

from __future__ import annotations

import validate_runtime_release as _validator


ATTESTATION_PATH_PREFIXES = (
    "parity/receipts/",
    "parity/manifests/",
    "parity/evidence/",
)


def main() -> None:
    _validator.ATTESTATION_PATH_PREFIXES = ATTESTATION_PATH_PREFIXES
    _validator.main()


if __name__ == "__main__":
    main()
