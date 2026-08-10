from __future__ import annotations

import math
from collections.abc import Sequence
from functools import partial
from typing import Protocol, cast

import mlx.core as mx
from mlx import nn

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.sam.common import MLPBlock
from sam3_mlx.sam.rope import apply_rotary_enc, apply_rotary_enc_real, compute_axial_cis


class _ArrayMethods(Protocol):
    def reshape(self, *shape: int) -> mx.array: ...

    def transpose(self, *axes: int) -> mx.array: ...


class _ArrayModule(Protocol):
    def __call__(self, x: mx.array) -> mx.array: ...


class _ActivationFactory(Protocol):
    def __call__(self) -> _ArrayModule: ...


class _ComputeAxialCis(Protocol):
    def __call__(
        self,
        end_x: int,
        end_y: int,
        scale_pos: float = 1.0,
        offset: int = 0,
        device: object | None = None,
    ) -> mx.array: ...


def _reshape(array: mx.array, *shape: int) -> mx.array:
    return cast(_ArrayMethods, array).reshape(*shape)


def _transpose(array: mx.array, *axes: int) -> mx.array:
    return cast(_ArrayMethods, array).transpose(*axes)


def _as_array_module(module: object) -> _ArrayModule:
    return cast(_ArrayModule, module)


def _raise_transformer_unsupported(feature: str, *, reason: str, detail: str) -> None:
    raise_unsupported(
        feature,
        reason=reason,
        detail=detail,
    )


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: _ActivationFactory = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = [
            TwoWayAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                activation=activation,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=(i == 0),
            )
            for i in range(depth)
        ]
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = _as_array_module(nn.LayerNorm(embedding_dim))

    def __call__(
        self,
        image_embedding: mx.array,
        image_pe: mx.array,
        point_embedding: mx.array,
    ) -> tuple[mx.array, mx.array]:
        batch_size, channels, height, width = image_embedding.shape
        image_embedding = _transpose(
            _reshape(image_embedding, batch_size, channels, height * width),
            0,
            2,
            1,
        )
        image_pe = _transpose(
            _reshape(image_pe, batch_size, channels, height * width),
            0,
            2,
            1,
        )

        queries = point_embedding
        keys = image_embedding
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = self.norm_final_attn(queries + attn_out)
        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: _ActivationFactory = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = _as_array_module(nn.LayerNorm(embedding_dim))
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = _as_array_module(nn.LayerNorm(embedding_dim))
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = _as_array_module(nn.LayerNorm(embedding_dim))
        self.norm4 = _as_array_module(nn.LayerNorm(embedding_dim))
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.skip_first_layer_pe = skip_first_layer_pe

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        query_pe: mx.array,
        key_pe: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            queries = queries + self.self_attn(q=q, k=q, v=queries)
        queries = self.norm1(queries)

        q = queries + query_pe
        k = keys + key_pe
        queries = self.norm2(queries + self.cross_attn_token_to_image(q=q, k=k, v=keys))

        queries = self.norm3(queries + self.mlp(queries))

        q = queries + query_pe
        k = keys + key_pe
        keys = self.norm4(keys + self.cross_attn_image_to_token(q=k, k=q, v=queries))
        return queries, keys


class Attention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int | None = None,
        use_fa3: bool = False,
    ) -> None:
        super().__init__()
        if use_fa3:
            _raise_transformer_unsupported(
                "sam3_mlx.sam.transformer.Attention(use_fa3=True)",
                reason="flash-attn-3",
                detail="FlashAttention 3 is non-MLX and not ported to MLX.",
            )
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        if self.internal_dim % num_heads != 0:
            raise AssertionError("num_heads must divide embedding_dim.")
        self.q_proj = _as_array_module(nn.Linear(embedding_dim, self.internal_dim))
        self.k_proj = _as_array_module(nn.Linear(self.kv_in_dim, self.internal_dim))
        self.v_proj = _as_array_module(nn.Linear(self.kv_in_dim, self.internal_dim))
        self.out_proj = _as_array_module(nn.Linear(self.internal_dim, embedding_dim))
        self.dropout_p = dropout

    def _separate_heads(self, x: mx.array, num_heads: int) -> mx.array:
        batch_size, num_tokens, channels = x.shape
        x = _reshape(x, batch_size, num_tokens, num_heads, channels // num_heads)
        return _transpose(x, 0, 2, 1, 3)

    def _recombine_heads(self, x: mx.array) -> mx.array:
        batch_size, num_heads, num_tokens, channels = x.shape
        return _reshape(
            _transpose(x, 0, 2, 1, 3),
            batch_size,
            num_tokens,
            num_heads * channels,
        )

    def __call__(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        if self.dropout_p and getattr(self, "training", False):
            _raise_transformer_unsupported(
                "sam3_mlx.sam.transformer.Attention(dropout_training=True)",
                reason="torch-autograd",
                detail="Training-time attention dropout is not implemented in this MLX port.",
            )
        scale = q.shape[-1] ** -0.5
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        return self.out_proj(self._recombine_heads(out))


class RoPEAttention(Attention):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int | None = None,
        use_fa3: bool = False,
        rope_theta: float = 10000.0,
        rope_k_repeat: bool = False,
        feat_sizes: Sequence[int] = (64, 64),
        use_rope_real: bool = False,
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            downsample_rate=downsample_rate,
            dropout=dropout,
            kv_in_dim=kv_in_dim,
            use_fa3=use_fa3,
        )
        if len(feat_sizes) != 2:
            raise ValueError("feat_sizes must contain exactly two spatial dimensions.")
        feat_x, feat_y = int(feat_sizes[0]), int(feat_sizes[1])
        self.use_rope_real = use_rope_real
        self.compute_cis: _ComputeAxialCis = cast(
            _ComputeAxialCis,
            partial(
                compute_axial_cis,
                dim=self.internal_dim // self.num_heads,
                theta=rope_theta,
            ),
        )
        self.freqs_cis = self.compute_cis(end_x=feat_x, end_y=feat_y)
        self.freqs_cis_real = self.freqs_cis[..., 0]
        self.freqs_cis_imag = self.freqs_cis[..., 1]
        self.rope_k_repeat = rope_k_repeat

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        num_k_exclude_rope: int = 0,
    ) -> mx.array:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        side = int(math.sqrt(q.shape[-2]))
        if side * side != q.shape[-2]:
            raise ValueError("RoPEAttention expects square spatial query tokens.")
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=side, end_y=side)
            self.freqs_cis_real = self.freqs_cis[..., 0]
            self.freqs_cis_imag = self.freqs_cis[..., 1]
        if q.shape[-2] != k.shape[-2] and not self.rope_k_repeat:
            raise AssertionError(
                "rope_k_repeat=True is required when q and k lengths differ."
            )

        num_k_rope = k.shape[-2] - num_k_exclude_rope
        if self.use_rope_real:
            q, rotated_k = apply_rotary_enc_real(
                q,
                k[:, :, :num_k_rope],
                freqs_cis_real=self.freqs_cis_real,
                freqs_cis_imag=self.freqs_cis_imag,
                repeat_freqs_k=self.rope_k_repeat,
            )
        else:
            q, rotated_k = apply_rotary_enc(
                q,
                k[:, :, :num_k_rope],
                self.freqs_cis,
                repeat_freqs_k=self.rope_k_repeat,
            )
        k = mx.concat([rotated_k, k[:, :, num_k_rope:]], axis=-2)

        if self.dropout_p and getattr(self, "training", False):
            _raise_transformer_unsupported(
                "sam3_mlx.sam.transformer.RoPEAttention(dropout_training=True)",
                reason="torch-autograd",
                detail="Training-time attention dropout is not implemented in this MLX port.",
            )
        scale = q.shape[-1] ** -0.5
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        return self.out_proj(self._recombine_heads(out))
