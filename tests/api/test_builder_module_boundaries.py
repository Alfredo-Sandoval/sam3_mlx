import ast

from tests._paths import REPO_ROOT


CHECKPOINT_ENTRYPOINTS = {
    "_audit_sam3_image_checkpoint_load",
    "_load_checkpoint",
    "_load_multiplex_checkpoint",
    "_load_multiplex_tracker_checkpoint",
    "_load_tracker_checkpoint",
    "_normalize_inst_interactive_weights",
    "_normalize_sam3_image_weights",
    "_normalize_sam31_multiplex_tracker_weights",
    "_normalize_sam31_multiplex_weights",
    "_normalize_tracker_checkpoint_weights",
    "download_ckpt_from_hf",
}

MULTIPLEX_ENTRYPOINTS = {
    "_build_checkpoint_free_multiplex_predictor_model",
    "_build_multiplex_detector_for_predictor",
    "_create_multiplex_maskmem_backbone",
    "_create_multiplex_transformer",
    "_create_multiplex_tri_backbone",
    "build_sam3_multiplex_video_model",
    "build_sam3_multiplex_video_predictor",
}


def _module_definitions(path):
    tree = ast.parse(path.read_text())
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_model_builder_delegates_checkpoint_subsystem():
    builder_path = REPO_ROOT / "sam3_mlx" / "model_builder.py"
    checkpoint_path = REPO_ROOT / "sam3_mlx" / "checkpoint.py"

    assert checkpoint_path.is_file()
    builder_definitions = _module_definitions(builder_path)
    checkpoint_definitions = _module_definitions(checkpoint_path)

    assert CHECKPOINT_ENTRYPOINTS.isdisjoint(builder_definitions)
    assert CHECKPOINT_ENTRYPOINTS <= checkpoint_definitions
    assert len(builder_path.read_text().splitlines()) < 950


def test_model_builder_delegates_multiplex_assembly():
    builder_path = REPO_ROOT / "sam3_mlx" / "model_builder.py"
    multiplex_path = REPO_ROOT / "sam3_mlx" / "multiplex_builder.py"

    assert multiplex_path.is_file()
    builder_definitions = _module_definitions(builder_path)
    multiplex_definitions = _module_definitions(multiplex_path)

    assert MULTIPLEX_ENTRYPOINTS.isdisjoint(builder_definitions)
    assert MULTIPLEX_ENTRYPOINTS <= multiplex_definitions
    assert len(multiplex_path.read_text().splitlines()) < 500
