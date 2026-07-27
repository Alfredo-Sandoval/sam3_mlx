import subprocess

import pytest

from sam3_mlx.source_binding import (
    ATTESTATION_PATH_PREFIXES,
    git_commit,
    validate_attestation_only_worktree,
)


def _run(repo, *args):
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "init")
    _run(repo, "config", "user.email", "tests@example.com")
    _run(repo, "config", "user.name", "Release Tests")
    (repo / "source.py").write_text("VALUE = 1\n")
    _run(repo, "add", "source.py")
    _run(repo, "commit", "-m", "initial")
    return repo


def test_source_binding_accepts_clean_and_evidence_only_worktrees(tmp_path):
    repo = _repo(tmp_path)
    commit = git_commit(repo)

    observed_commit, changed = validate_attestation_only_worktree(repo)
    assert observed_commit == commit
    assert changed == ()

    evidence = repo / "parity" / "evidence" / "example.npz"
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(b"evidence")

    observed_commit, changed = validate_attestation_only_worktree(repo)
    assert observed_commit == commit
    assert changed == ("parity/evidence/example.npz",)


def test_source_binding_rejects_dirty_source_even_with_evidence(tmp_path):
    repo = _repo(tmp_path)
    receipt = repo / "parity" / "receipts" / "latest.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}\n")
    (repo / "source.py").write_text("VALUE = 2\n")

    with pytest.raises(ValueError, match="Non-evidence worktree changes"):
        validate_attestation_only_worktree(repo)


def test_attestation_prefixes_include_raw_evidence():
    assert ATTESTATION_PATH_PREFIXES == (
        "parity/receipts/",
        "parity/manifests/",
        "parity/evidence/",
    )
