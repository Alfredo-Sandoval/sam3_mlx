"""Verify the built wheel from outside the source package path."""

from importlib.metadata import version
from importlib.resources import files
from importlib.util import find_spec
from inspect import signature

import sam3_mlx

assert sam3_mlx.__version__ == version("sam3-mlx")
assert "build_sam3_image_model" in sam3_mlx.__all__
assert "checkpoint_path" in signature(sam3_mlx.build_sam3_image_model).parameters
assert files("sam3_mlx").joinpath("assets/bpe_simple_vocab_16e6.txt.gz").is_file()
assert find_spec("sam3_mlx.agent") is None
assert find_spec("sam3_mlx.eval") is None
assert find_spec("sam3_mlx.train") is None
