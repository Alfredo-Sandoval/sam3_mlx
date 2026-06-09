import numpy as np
import mlx.core as mx
import pytest

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.maskformer_segmentation import PixelDecoder


torch = pytest.importorskip("torch")


def test_pixel_decoder_matches_torch_group_norm_semantics():
    torch_conv = torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
    torch_norm = torch.nn.GroupNorm(8, 8)
    decoder = PixelDecoder(hidden_dim=8, num_upsampling_stages=1)

    with torch.no_grad():
        torch_conv.weight.copy_(
            torch.linspace(-0.2, 0.2, steps=torch_conv.weight.numel()).reshape_as(
                torch_conv.weight
            )
        )
        torch_conv.bias.copy_(torch.linspace(-0.05, 0.05, steps=8))
        torch_norm.weight.copy_(torch.linspace(0.8, 1.2, steps=8))
        torch_norm.bias.copy_(torch.linspace(-0.1, 0.1, steps=8))

    decoder.conv_layers[0].weight = mx.array(
        torch_conv.weight.detach().numpy().transpose(0, 2, 3, 1)
    )
    decoder.conv_layers[0].bias = mx.array(torch_conv.bias.detach().numpy())
    decoder.norms[0].weight = mx.array(torch_norm.weight.detach().numpy())
    decoder.norms[0].bias = mx.array(torch_norm.bias.detach().numpy())

    high_res = np.linspace(-1.5, 1.5, num=1 * 8 * 4 * 4, dtype=np.float32).reshape(
        1, 8, 4, 4
    )
    low_res = np.linspace(0.75, -0.75, num=1 * 8 * 2 * 2, dtype=np.float32).reshape(
        1, 8, 2, 2
    )

    with torch.no_grad():
        expected = (
            high_res
            + torch.nn.functional.interpolate(
                torch.from_numpy(low_res),
                size=high_res.shape[-2:],
                mode="nearest",
            ).numpy()
        )
        expected = torch_conv(torch.from_numpy(expected))
        expected = torch.nn.functional.relu(torch_norm(expected)).numpy()

    actual = decoder([mx.array(high_res), mx.array(low_res)])

    np.testing.assert_allclose(to_numpy(actual), expected, rtol=1e-5, atol=1e-5)
