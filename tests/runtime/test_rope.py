import numpy as np
import mlx.core as mx

from sam3_mlx.sam.rope import apply_rotary_enc, apply_rotary_enc_real


def test_rotary_repeat_freqs_tiles_query_table_for_keys():
    xq = mx.zeros((1, 1, 2, 4), dtype=mx.float32)
    xk = mx.array(
        [
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ]
            ]
        ],
        dtype=mx.float32,
    )
    freqs_cis = mx.array(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ],
        dtype=mx.float32,
    )

    _, rotated = apply_rotary_enc(xq, xk, freqs_cis, repeat_freqs_k=True)

    expected = np.array(
        [
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [-6.0, 5.0, -8.0, 7.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [-14.0, 13.0, -16.0, 15.0],
                ]
            ]
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.array(rotated), expected, atol=0.0, rtol=0.0)


def test_real_rotary_repeat_freqs_tiles_query_table_for_keys():
    xq = mx.zeros((1, 1, 2, 4), dtype=mx.float32)
    xk = mx.array(
        [
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ]
            ]
        ],
        dtype=mx.float32,
    )
    freqs_real = mx.array([[1.0, 1.0], [0.0, 0.0]], dtype=mx.float32)
    freqs_imag = mx.array([[0.0, 0.0], [1.0, 1.0]], dtype=mx.float32)

    _, rotated = apply_rotary_enc_real(
        xq,
        xk,
        freqs_cis_real=freqs_real,
        freqs_cis_imag=freqs_imag,
        repeat_freqs_k=True,
    )

    expected = np.array(
        [
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [-6.0, 5.0, -8.0, 7.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [-14.0, 13.0, -16.0, 15.0],
                ]
            ]
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.array(rotated), expected, atol=0.0, rtol=0.0)
