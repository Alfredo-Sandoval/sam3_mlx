import json

from sam3_mlx.source_binding import git_commit, validate_attestation_only_worktree
from tests._paths import REPO_ROOT

CHECKED_IN_RECEIPT_COMMIT = "fe8b1e168c73b712467befab095f64cd21cb77c0"


def test_checked_in_receipt_is_bound_to_historical_source() -> None:
    receipt = json.loads((REPO_ROOT / "parity" / "receipts" / "latest.json").read_text())
    assert receipt["git_commit"] == CHECKED_IN_RECEIPT_COMMIT
    assert receipt["parity"]["calibration_profile"] == "example"


def test_checked_in_receipt_does_not_attest_this_worktree() -> None:
    receipt = json.loads((REPO_ROOT / "parity" / "receipts" / "latest.json").read_text())
    receipt_commit = receipt["git_commit"]
    head = git_commit(REPO_ROOT)
    if head != receipt_commit:
        assert head != receipt_commit
        return
    try:
        validate_attestation_only_worktree(REPO_ROOT)
    except ValueError as exc:
        assert "Non-evidence worktree changes" in str(exc)
        return
    raise AssertionError(
        "checked-in receipt names HEAD and the worktree is attestation-clean; "
        "regenerate latest.json only after official parity is re-run"
    )
