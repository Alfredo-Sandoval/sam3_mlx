.PHONY: preprocess-parity-check artifact-check release-evidence-audit release-check runtime-release-check

preprocess-parity-check:
	SAM3_MLX_REQUIRE_TORCHVISION=1 uv run pytest -q tests/image/test_sam3_image_processor.py::test_transform_matches_torchvision_on_synthetic_aspect_ratios_when_available

artifact-check:
	uv run python scripts/validate_release.py

release-evidence-audit:
	uv run python scripts/audit_release_candidate.py --receipt parity/receipts/latest.json

release-check: artifact-check runtime-release-check

runtime-release-check: release-evidence-audit
	uv run python scripts/validate_runtime_release_hardened.py --receipt parity/receipts/latest.json
