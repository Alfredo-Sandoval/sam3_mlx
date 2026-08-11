from sam3_mlx.model.bounded_cache import BoundedLRUCache


def test_bounded_lru_cache_evicts_least_recently_used():
    cache = BoundedLRUCache[int, str](maxsize=2)
    cache[1] = "a"
    cache[2] = "b"
    assert cache.get(1) == "a"  # refresh 1
    cache[3] = "c"
    assert 2 not in cache
    assert cache.get(1) == "a"
    assert cache.get(3) == "c"
    assert len(cache) == 2


def test_position_encoding_cache_is_bounded():
    import mlx.core as mx

    from sam3_mlx.model.position_encoding import PositionEmbeddingSine
    from sam3_mlx.mlx_runtime import evaluate_boundary

    pos = PositionEmbeddingSine(num_pos_feats=8, cache_size=2)
    for height, width in ((4, 4), (8, 8), (16, 16)):
        out = pos(mx.zeros((1, 1, height, width), dtype=mx.float32))
        evaluate_boundary(out)
        assert out.shape == (1, 8, height, width)
    assert len(pos.cache) <= 2


def test_rope_resolution_cache_is_bounded():
    from sam3_mlx.model.vitdet import Attention
    from sam3_mlx.mlx_runtime import evaluate_boundary

    attention = Attention(
        dim=8,
        num_heads=2,
        input_size=(4, 4),
        use_rope=True,
        cls_token=False,
    )
    for size in range(1, 11):
        frequencies = attention._rope_freqs_for_tokens(  # pyright: ignore[reportPrivateUsage]
            size,
            spatial_size=(size, 1),
        )
        evaluate_boundary(frequencies)
    cache = attention._freqs_cis_cache  # pyright: ignore[reportPrivateUsage]
    assert len(cache) == cache.maxsize
    assert (1, 1) not in cache


def test_decoder_coordinate_cache_is_bounded():
    import mlx.core as mx
    from mlx import nn

    from sam3_mlx.model.decoder import TransformerDecoder
    from sam3_mlx.mlx_runtime import evaluate_boundary

    class _CrossAttention:
        num_heads = 1

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.cross_attn = _CrossAttention()
            self.cross_attn_image = _CrossAttention()

    decoder = TransformerDecoder(
        d_model=4,
        frozen=False,
        interaction_layer=None,
        layer=_Layer(),
        num_layers=1,
        num_queries=1,
        return_intermediate=True,
        box_refine=True,
        boxRPB="linear",
    )
    boxes = mx.zeros((1, 1, 4), dtype=mx.float32)
    for size in range(1, 11):
        matrix = decoder._get_rpb_matrix(  # pyright: ignore[reportPrivateUsage]
            boxes, (size, size)
        )
        evaluate_boundary(matrix)
    assert len(decoder.coord_cache) == decoder.coord_cache.maxsize
    assert (1, 1) not in decoder.coord_cache
