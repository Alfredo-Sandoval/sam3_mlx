import json
from pathlib import Path

import pytest

from sam3_mlx import convert


def test_default_mlx_checkpoint_download_is_revision_pinned(monkeypatch, tmp_path):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        model_dir = tmp_path / "snapshot"
        model_dir.mkdir()
        (model_dir / "model.safetensors").touch()
        return str(model_dir)

    monkeypatch.setattr(convert, "snapshot_download", fake_snapshot_download)

    result = convert.load_from_hub()

    assert result == Path(tmp_path / "snapshot" / "model.safetensors")
    assert captured["repo_id"] == convert.MLX_COMMUNITY_REPO
    assert captured["revision"] == convert.MLX_COMMUNITY_REVISION


def test_custom_checkpoint_repo_does_not_reuse_default_revision(monkeypatch, tmp_path):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        model_dir = tmp_path / "custom-snapshot"
        model_dir.mkdir()
        (model_dir / "model.safetensors").touch()
        return str(model_dir)

    monkeypatch.setattr(convert, "snapshot_download", fake_snapshot_download)

    convert.load_from_hub("owner/custom-sam3")

    assert "revision" not in captured


def test_official_checkpoint_download_is_revision_pinned(monkeypatch, tmp_path):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(convert, "snapshot_download", fake_snapshot_download)

    convert.download(convert.PYTORCH_REPO)

    assert captured["revision"] == convert.PYTORCH_REVISION


def test_sam31_checkpoint_file_download_is_revision_pinned(monkeypatch, tmp_path):
    from sam3_mlx.model_builder import download_ckpt_from_hf

    calls = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / kwargs["filename"])

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        fake_hf_hub_download,
    )

    result = download_ckpt_from_hf("sam3.1")

    assert result == str(tmp_path / "sam3.1_multiplex.pt")
    assert [call["revision"] for call in calls] == [
        convert.SAM31_REVISION,
        convert.SAM31_REVISION,
    ]


def test_conversion_cache_rejects_weights_without_source_provenance(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "model.safetensors.index.json").write_text("{}")

    with pytest.raises(ValueError, match="has no conversion provenance"):
        convert.download_and_convert(mlx_path=tmp_path)


def test_conversion_cache_rejects_different_source_revision(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    (tmp_path / "conversion.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "source_repo": convert.PYTORCH_REPO,
                "source_revision": "different-revision",
            }
        )
    )

    with pytest.raises(ValueError, match="does not match the requested source"):
        convert.download_and_convert(mlx_path=tmp_path)


def test_conversion_cache_reuses_matching_pinned_source(monkeypatch, tmp_path):
    weights_path = tmp_path / "model.safetensors"
    weights_path.touch()
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    (tmp_path / "conversion.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "source_repo": convert.PYTORCH_REPO,
                "source_revision": convert.PYTORCH_REVISION,
            }
        )
    )

    def fail_download(*args, **kwargs):
        raise AssertionError("matching conversion cache should not download")

    monkeypatch.setattr(convert, "download", fail_download)

    assert convert.download_and_convert(mlx_path=tmp_path) == weights_path


def test_conversion_requires_revision_for_custom_source(tmp_path):
    with pytest.raises(ValueError, match="revision is required for custom"):
        convert.download_and_convert(
            hf_repo="owner/custom-sam3",
            mlx_path=tmp_path,
        )
