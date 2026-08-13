from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
import math
from typing import Literal, Protocol, TypeAlias, cast

import mlx.core as mx
from mlx import nn

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.resolutions import window_layout_is_exact
from sam3_mlx.model import data_misc as data_misc_module
from sam3_mlx.model.bounded_cache import BoundedLRUCache
from sam3_mlx.model.data_misc import NestedTensor
from sam3_mlx.model.model_misc import DropPath, LayerScale
from sam3_mlx.model.model_misc import Mlp as _Mlp

_DEFAULT_ROPE_CACHE_SIZE = 8

FeatureMap: TypeAlias = mx.array | NestedTensor
NormFactory: TypeAlias = Callable[..., nn.Module]
ActFactory: TypeAlias = Callable[..., nn.Module]


class _ArrayModule(Protocol):
    def __call__(self, x: mx.array) -> mx.array: ...


class _LinearModule(Protocol):
    weight: mx.array
    bias: mx.array | None

    def __call__(self, x: mx.array) -> mx.array: ...


class _LayerNormModule(Protocol):
    weight: mx.array
    bias: mx.array

    def __call__(self, x: mx.array) -> mx.array: ...


class _NestedFeature(Protocol):
    tensors: mx.array
    mask: mx.array | None


class _Pad(Protocol):
    def __call__(
        self,
        a: mx.array,
        pad_width: int | tuple[int] | tuple[int, int] | list[tuple[int, int]],
        mode: Literal["constant", "edge"] = "constant",
        constant_values: int | float | bool | mx.array = 0,
        *,
        stream: object | None = None,
    ) -> mx.array: ...


class _Einsum(Protocol):
    def __call__(
        self,
        subscripts: str,
        *operands: mx.array,
        stream: object | None = None,
    ) -> mx.array: ...


class _Linspace(Protocol):
    def __call__(
        self,
        start: float,
        stop: float,
        num: int | None = 50,
        dtype: mx.Dtype | None = mx.float32,
        stream: object | None = None,
    ) -> mx.array: ...


class _Interpolate(Protocol):
    def __call__(
        self,
        input: mx.array,
        size: tuple[int, int] | None = None,
        scale_factor: float | tuple[float, float] | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
        antialias: bool = False,
    ) -> mx.array: ...


class _UpsampleFactory(Protocol):
    def __call__(
        self,
        scale_factor: float | tuple[float, float] = 2.0,
        mode: str = "nearest",
        align_corners: bool | None = None,
    ) -> _ArrayModule: ...


class _ComputeAxialCis(Protocol):
    def __call__(
        self,
        *,
        end_x: int,
        end_y: int,
        scale_pos: float = 1.0,
        offset: int = 0,
    ) -> mx.array: ...


_pad = cast(_Pad, getattr(mx, "pad"))
_einsum = cast(_Einsum, getattr(mx, "einsum"))
_linspace = cast(_Linspace, getattr(mx, "linspace"))
_interpolate = cast(_Interpolate, getattr(data_misc_module, "interpolate"))
_upsample = cast(_UpsampleFactory, getattr(nn, "Upsample"))


def _as_array_module(module: object) -> _ArrayModule:
    return cast(_ArrayModule, module)


def _as_linear(module: object) -> _LinearModule:
    return cast(_LinearModule, module)


def _as_layer_norm(module: object) -> _LayerNormModule:
    return cast(_LayerNormModule, module)


def _as_nested(feature: object) -> _NestedFeature:
    return cast(_NestedFeature, feature)


class Mlp(_Mlp):
    def forward(self, x: mx.array) -> mx.array:
        return super().__call__(x)


def polar(a: mx.array, b: mx.array) -> mx.array:
    return (a * mx.exp(1j * b)).astype(mx.complex64)


def real(x: mx.array) -> mx.array:
    parts = mx.reshape(mx.view(x, mx.float32), (*x.shape, 2))
    return mx.reshape(parts, (*x.shape[:-1], -1))


def view_as_complex(x: mx.array) -> mx.array:
    assert x.shape[-1] % 2 == 0
    new_shape = list(x.shape[:-1]) + [-1, 2]
    parts = mx.reshape(x, new_shape)
    return (parts[..., 0] + 1j * parts[..., 1]).astype(mx.complex64)


def init_t_xy(
    end_x: int, end_y: int, scale: float = 1.0, offset: int = 0
) -> tuple[mx.array, mx.array]:
    t = mx.arange(end_x * end_y, dtype=mx.float32)
    t_x = (t % end_x).astype(mx.float32)
    t_y = mx.floor(mx.divide(t, end_x)).astype(mx.float32)
    return t_x * scale + offset, t_y * scale + offset


def compute_axial_cis(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
    scale_pos: float = 1.0,
    offset: int = 0,
) -> mx.array:
    freqs_x = 1.0 / (
        theta ** (mx.arange(0, dim, 4)[: (dim // 4)].astype(mx.float32) / dim)
    )
    freqs_y = 1.0 / (
        theta ** (mx.arange(0, dim, 4)[: (dim // 4)].astype(mx.float32) / dim)
    )

    t_x, t_y = init_t_xy(end_x, end_y, scale=scale_pos, offset=offset)
    freqs_x = mx.outer(t_x, freqs_x)
    freqs_y = mx.outer(t_y, freqs_y)
    freqs_cis_x = polar(mx.ones_like(freqs_x), freqs_x)
    freqs_cis_y = polar(mx.ones_like(freqs_y), freqs_y)
    return mx.concat([freqs_cis_x, freqs_cis_y], axis=-1)


def reshape_for_broadcast(freqs_cis: mx.array, x: mx.array) -> mx.array:
    ndim = x.ndim

    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return mx.reshape(freqs_cis, shape)


def apply_rotary_enc(
    xq: mx.array, xk: mx.array, freqs_cis: mx.array, repeat_freqs_k: bool = False
) -> tuple[mx.array, mx.array]:
    xq_ = view_as_complex(xq)
    xk_ = view_as_complex(xk) if xk.shape[-2] != 0 else None

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = mx.flatten(real(xq_ * freqs_cis), start_axis=3)
    if xk_ is None:
        return xq_out.astype(xq.dtype), xk

    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        reps = [1] * (freqs_cis.ndim - 2) + [r, 1]
        freqs_cis = mx.tile(freqs_cis, reps)
    xk_out = mx.flatten(real(xk_ * freqs_cis), start_axis=3)
    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)


def window_partition(x: mx.array, window_size: int) -> tuple[mx.array, tuple[int, int]]:
    batch, height, width, channels = (
        int(x.shape[0]),
        int(x.shape[1]),
        int(x.shape[2]),
        int(x.shape[3]),
    )

    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = _pad(
            x,
            [
                (0, 0),
                (0, pad_h),
                (0, pad_w),
                (0, 0),
            ],
        )
    hp, wp = height + pad_h, width + pad_w

    x = mx.reshape(
        x,
        (
            batch,
            hp // window_size,
            window_size,
            wp // window_size,
            window_size,
            channels,
        ),
    )
    windows = mx.reshape(
        mx.transpose(x, axes=(0, 1, 3, 2, 4, 5)),
        (-1, window_size, window_size, channels),
    )
    return windows, (hp, wp)


def window_unpartition(
    windows: mx.array,
    window_size: int,
    pad_hw: tuple[int, int],
    hw: tuple[int, int],
) -> mx.array:
    hp, wp = pad_hw
    height, width = hw
    batch = int(windows.shape[0]) // (hp * wp // window_size // window_size)
    x = mx.reshape(
        windows,
        (batch, hp // window_size, wp // window_size, window_size, window_size, -1),
    )
    x = mx.reshape(mx.transpose(x, axes=(0, 1, 3, 2, 4, 5)), (batch, hp, wp, -1))

    if hp > height or wp > width:
        x = x[:, :height, :width, :]

    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: mx.array) -> mx.array:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos = _resize_rel_pos(rel_pos, max_rel_dist)

    q_coords = mx.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = mx.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos[relative_coords.astype(mx.int64)]


def _resize_rel_pos(rel_pos: mx.array, target_len: int) -> mx.array:
    src_len = int(rel_pos.shape[0])
    if src_len == target_len:
        return rel_pos
    src_pos = (mx.arange(target_len, dtype=mx.float32) + 0.5) * (
        src_len / target_len
    ) - 0.5
    src_pos = mx.clip(src_pos, 0.0, float(src_len - 1))
    src_lo = mx.floor(src_pos).astype(mx.int64)
    src_hi = mx.minimum(src_lo + 1, src_len - 1)
    weight = (src_pos - src_lo.astype(mx.float32))[:, None]
    return rel_pos[src_lo] * (1.0 - weight) + rel_pos[src_hi] * weight


def concat_rel_pos(
    q: mx.array,
    k: mx.array,
    q_hw: tuple[int, int],
    k_hw: tuple[int, int],
    rel_pos_h: mx.array,
    rel_pos_w: mx.array,
    rescale: bool = False,
    relative_coords: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    q_h, q_w = q_hw
    k_h, k_w = k_hw
    assert (q_h == q_w) and (k_h == k_w), "only square inputs supported"

    if relative_coords is not None:
        rh = rel_pos_h[relative_coords]
        rw = rel_pos_w[relative_coords]
    else:
        rh = get_rel_pos(q_h, k_h, rel_pos_h)
        rw = get_rel_pos(q_w, k_w, rel_pos_w)

    batch, _, dim = (int(q.shape[0]), int(q.shape[1]), int(q.shape[2]))
    r_q = mx.reshape(q, (batch, q_h, q_w, dim))
    old_scale = dim**0.5
    new_scale = (dim + k_h + k_w) ** 0.5 if rescale else old_scale
    scale_ratio = new_scale / old_scale

    rel_h = _einsum("bhwc,hkc->bhwk", r_q, rh) * new_scale
    rel_w = _einsum("bhwc,wkc->bhwk", r_q, rw) * new_scale

    eye_h = mx.reshape(mx.eye(k_h, dtype=q.dtype), (1, k_h, 1, k_h))
    eye_w = mx.reshape(mx.eye(k_w, dtype=q.dtype), (1, 1, k_w, k_w))
    eye_h = mx.broadcast_to(eye_h, (batch, k_h, k_w, k_h))
    eye_w = mx.broadcast_to(eye_w, (batch, k_h, k_w, k_w))

    q = mx.concat([r_q * scale_ratio, rel_h, rel_w], axis=-1)
    q = mx.reshape(q, (batch, q_h * q_w, -1))
    k = mx.concat([mx.reshape(k, (batch, k_h, k_w, -1)), eye_h, eye_w], axis=-1)
    k = mx.reshape(k, (batch, k_h * k_w, -1))
    return q, k


def get_abs_pos(
    abs_pos: mx.array,
    has_cls_token: bool,
    hw: tuple[int, int],
    retain_cls_token: bool = False,
    tiling: bool = False,
) -> mx.array:
    if retain_cls_token:
        assert has_cls_token

    height, width = hw
    if has_cls_token:
        cls_pos = abs_pos[:, :1]
        spatial_pos = abs_pos[:, 1:]
    else:
        cls_pos = None
        spatial_pos = abs_pos

    xy_num = int(spatial_pos.shape[1])
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != height or size != width:
        new_abs_pos = mx.transpose(
            mx.reshape(spatial_pos, (1, size, size, -1)), axes=(0, 3, 1, 2)
        )
        if tiling:
            current_h, current_w = int(new_abs_pos.shape[2]), int(new_abs_pos.shape[3])
            rep_h = (height // current_h) + 1
            rep_w = (width // current_w) + 1

            new_abs_pos = mx.tile(new_abs_pos, (1, 1, rep_h, rep_w))
            new_abs_pos = new_abs_pos[:, :, :height, :width]
        else:
            current_h, current_w = int(new_abs_pos.shape[2]), int(new_abs_pos.shape[3])

            scale_h = height / current_h
            scale_w = width / current_w
            upsample_fn = _upsample(
                scale_factor=(scale_h, scale_w), mode="cubic", align_corners=False
            )
            new_abs_pos = mx.transpose(
                upsample_fn(mx.transpose(new_abs_pos, axes=(0, 2, 3, 1))),
                axes=(0, 3, 1, 2),
            )

        if not retain_cls_token:
            return mx.transpose(new_abs_pos, axes=(0, 2, 3, 1))

        assert cls_pos is not None
        return mx.concat(
            [
                cls_pos,
                mx.reshape(
                    mx.transpose(new_abs_pos, axes=(0, 2, 3, 1)),
                    (1, height * width, -1),
                ),
            ],
            axis=1,
        )

    if not retain_cls_token:
        return mx.reshape(spatial_pos, (1, height, width, -1))

    assert cls_pos is not None
    return mx.concat([cls_pos, spatial_pos], axis=1)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        kernel_size: tuple[int, int] = (16, 16),
        stride: tuple[int, int] = (16, 16),
        padding: tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.proj = _as_array_module(
            nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        )

    def forward(self, x: mx.array) -> mx.array:
        # B C H W -> B H W C
        x = mx.transpose(x, axes=(0, 2, 3, 1))
        x = self.proj(x)
        # B H W C
        return x

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: tuple[int, int] | None = None,
        cls_token: bool = True,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        rope_pt_size: tuple[int, int] | None = None,
        rope_interp: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.cls_token = cls_token

        self.qkv = _as_linear(nn.Linear(dim, dim * 3, bias=qkv_bias))
        self.proj = _as_linear(nn.Linear(dim, dim))

        self.use_rel_pos = use_rel_pos
        self.input_size = input_size

        self.use_rope = use_rope
        self.rope_theata = rope_theta
        self.rope_pt_size = rope_pt_size
        self.rope_interp = rope_interp

        self.rel_pos_h: mx.array | None = None
        self.rel_pos_w: mx.array | None = None
        self.relative_coords: mx.array | None = None
        self.freqs_cis: mx.array | None = None
        self.compute_cis: _ComputeAxialCis | None = None

        self._setup_rel_pos(rel_pos_zero_init)
        self._setup_rope_freqs()

    def _setup_rel_pos(self, rel_pos_zero_init: bool = True) -> None:
        if not self.use_rel_pos:
            self.rel_pos_h = None
            self.rel_pos_w = None
            self.relative_coords = None
            return

        assert self.input_size is not None
        assert self.cls_token is False, "not supported"
        self.rel_pos_h = mx.zeros((2 * self.input_size[0] - 1, self.head_dim))
        self.rel_pos_w = mx.zeros((2 * self.input_size[1] - 1, self.head_dim))

        if not rel_pos_zero_init:
            self.rel_pos_h = (
                mx.random.truncated_normal(
                    lower=-2, upper=2, shape=self.rel_pos_h.shape
                )
                * 0.02
            )
            self.rel_pos_w = (
                mx.random.truncated_normal(
                    lower=-2, upper=2, shape=self.rel_pos_w.shape
                )
                * 0.02
            )

        height, width = self.input_size
        q_coords = mx.arange(height)[:, None]
        k_coords = mx.arange(width)[None, :]
        relative_coords = (q_coords - k_coords) + (height - 1)
        self.relative_coords = relative_coords.astype(mx.int64)

    def _setup_rope_freqs(self) -> None:
        self._freqs_cis_cache: BoundedLRUCache[tuple[int, int], mx.array] = (
            BoundedLRUCache(maxsize=_DEFAULT_ROPE_CACHE_SIZE)
        )
        if not self.use_rope:
            self.freqs_cis = None
            self.compute_cis = None
            return

        assert self.input_size is not None

        if self.rope_pt_size is None:
            self.rope_pt_size = self.input_size

        self.compute_cis = cast(
            _ComputeAxialCis,
            partial(
                compute_axial_cis,
                dim=self.head_dim,
                theta=self.rope_theata,
            ),
        )

        scale_pos = 1.0
        if self.rope_interp:
            scale_pos = self.rope_pt_size[0] / self.input_size[0]
        freqs_cis = self.compute_cis(
            end_x=self.input_size[0],
            end_y=self.input_size[1],
            scale_pos=scale_pos,
        )

        if self.cls_token:
            t = mx.zeros(
                self.head_dim // 2,
                dtype=mx.float32,
            )
            cls_freqs_cis = polar(mx.ones_like(t), t)[None, :]
            freqs_cis = mx.concat([cls_freqs_cis, freqs_cis], axis=0)

        self.freqs_cis = freqs_cis

    def _build_rope_freqs(self, spatial_size: tuple[int, int]) -> mx.array:
        end_x, end_y = spatial_size
        scale_pos = 1.0
        if self.rope_interp:
            assert self.rope_pt_size is not None
            scale_pos = self.rope_pt_size[0] / end_x
        compute_cis = self.compute_cis
        assert compute_cis is not None
        freqs_cis = compute_cis(
            end_x=end_x,
            end_y=end_y,
            scale_pos=scale_pos,
        )
        if self.cls_token:
            t = mx.zeros(self.head_dim // 2, dtype=mx.float32)
            cls_freqs_cis = polar(mx.ones_like(t), t)[None, :]
            freqs_cis = mx.concat([cls_freqs_cis, freqs_cis], axis=0)
        return freqs_cis

    def _rope_freqs_for_tokens(
        self,
        token_count: int,
        spatial_size: tuple[int, int] | None,
    ) -> mx.array:
        assert self.freqs_cis is not None
        if self.freqs_cis.shape[0] == token_count:
            return self.freqs_cis
        if spatial_size is None:
            raise RuntimeError(
                "RoPE token count does not match precomputed frequencies and "
                "no spatial size was provided."
            )
        end_x, end_y = spatial_size
        expected_tokens = end_x * end_y + (1 if self.cls_token else 0)
        if expected_tokens != token_count:
            raise RuntimeError(
                "RoPE spatial size does not match token count: "
                f"{spatial_size} implies {expected_tokens} tokens, got {token_count}."
            )
        cache_key = (end_x, end_y)
        cached = self._freqs_cis_cache.get(cache_key)
        if cached is None:
            cached = self._build_rope_freqs(spatial_size)
            self._freqs_cis_cache[cache_key] = cached
        return cached

    def _apply_rope(
        self,
        q: mx.array,
        k: mx.array,
        spatial_size: tuple[int, int] | None = None,
    ) -> tuple[mx.array, mx.array]:
        if not self.use_rope:
            return q, k

        freqs_cis = self._rope_freqs_for_tokens(int(q.shape[-2]), spatial_size)
        return apply_rotary_enc(q, k, freqs_cis=freqs_cis)

    def forward(self, x: mx.array) -> mx.array:
        s = 1 if self.cls_token else 0
        if x.ndim == 4:
            batch = int(x.shape[0])
            height = int(x.shape[1])
            width = int(x.shape[2])
            assert s == 0
            length = height * width
            ndim = 4
        else:
            assert x.ndim == 3
            batch = int(x.shape[0])
            length = int(x.shape[1])
            ndim = 3
            side = int(math.sqrt(length - s))
            assert side * side == length - s
            height = width = side

        # qkv with shape (3, B, nHead, L, C)
        qkv = mx.reshape(self.qkv(x), (batch, length, 3, self.num_heads, -1))
        # q, k, v with shape (B, nHead, L, C)
        qkv = mx.transpose(qkv, axes=(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k = self._apply_rope(q, k, spatial_size=(height, width))
        if self.use_rel_pos:
            rel_pos_h = self.rel_pos_h
            rel_pos_w = self.rel_pos_w
            if rel_pos_h is None or rel_pos_w is None:
                raise RuntimeError(
                    "use_rel_pos is enabled but relative position tables are missing."
                )
            q, k = concat_rel_pos(
                mx.reshape(q, (-1, length, int(q.shape[-1]))),
                mx.reshape(k, (-1, length, int(k.shape[-1]))),
                (height, width),
                (height, width),
                rel_pos_h,
                rel_pos_w,
                rescale=True,
                relative_coords=self.relative_coords,
            )
            q = mx.reshape(q, (batch, self.num_heads, height * width, -1))
            k = mx.reshape(k, (batch, self.num_heads, height * width, -1))

        scale = float(q.shape[-1]) ** -0.5
        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

        if ndim == 4:
            x = mx.reshape(
                mx.transpose(
                    mx.reshape(x, (batch, self.num_heads, height, width, -1)),
                    axes=(0, 2, 3, 1, 4),
                ),
                (batch, height, width, -1),
            )
        else:
            x = mx.reshape(
                mx.transpose(
                    mx.reshape(x, (batch, length, self.num_heads, -1)),
                    axes=(0, 2, 1, 3),
                ),
                (batch, length, -1),
            )

        x = self.proj(x)
        return x

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        norm_layer: NormFactory = nn.LayerNorm,
        act_layer: ActFactory = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: tuple[int, int] | None = None,
        use_rope: bool = False,
        rope_pt_size: tuple[int, int] | None = None,
        rope_tiled: bool = False,
        rope_interp: bool = False,
        use_ve_rope: bool = False,
        cls_token: bool = False,
        dropout: float = 0.0,
        init_values: float | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = _as_array_module(norm_layer(dim))
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            use_rope=use_rope,
            rope_pt_size=rope_pt_size,
            rope_interp=rope_interp,
            cls_token=cls_token,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values)
            if init_values
            else _as_array_module(nn.Identity())
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else _as_array_module(nn.Identity())
        )

        self.norm2 = _as_array_module(norm_layer(dim))
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=(dropout, 0.0),
        )

        self.ls2 = (
            LayerScale(dim, init_values=init_values)
            if init_values
            else _as_array_module(nn.Identity())
        )
        self.dropout = (
            _as_array_module(nn.Identity())
            if dropout == 0
            else _as_array_module(nn.Dropout(dropout))
        )
        self.window_size = window_size

    def forward(self, x: mx.array, *, windowed: bool = False) -> mx.array:
        shortcut = x
        x = self.norm1(x)

        if self.window_size > 0 and not windowed:
            height = int(x.shape[1])
            width = int(x.shape[2])
            x, pad_hw = window_partition(x, self.window_size)
            x = self.ls1(self.attn(x))
            x = window_unpartition(x, self.window_size, pad_hw, (height, width))
        else:
            x = self.ls1(self.attn(x))

        x = shortcut + self.dropout(self.drop_path(x))
        x = x + self.dropout(self.drop_path(self.ls2(self.mlp(self.norm2(x)))))

        return x

    def __call__(self, x: mx.array, *, windowed: bool = False) -> mx.array:
        return self.forward(x, windowed=windowed)


class ViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        norm_layer: NormFactory | str = "LayerNorm",
        act_layer: ActFactory = nn.GELU,
        use_abs_pos: bool = True,
        tile_abs_pos: bool = True,
        rel_pos_blocks: Sequence[int] | bool = (2, 5, 8, 11),
        rel_pos_zero_init: bool = True,
        window_size: int = 14,
        global_att_blocks: Sequence[int] = (2, 5, 8, 11),
        use_rope: bool = False,
        rope_pt_size: int | None = None,
        use_interp_rope: bool = False,
        pretrain_img_size: int = 224,
        pretrain_use_cls_token: bool = True,
        retain_cls_token: bool = True,
        dropout: float = 0.0,
        return_interm_layers: bool = False,
        init_values: float | None = None,
        ln_pre: bool = False,
        ln_post: bool = False,
        bias_patch_embed: bool = True,
        compile_mode: str | bool | None = None,
        use_act_checkpoint: bool = True,
        persist_exact_windows: bool = True,
    ) -> None:
        super().__init__()
        if compile_mode not in (None, False):
            raise_unsupported(
                "sam3_mlx.model.vitdet.ViT(compile_mode)",
                reason="torch-compile",
                detail="torch.compile is not part of the sam3_mlx runtime.",
            )
        self.pretrain_use_cls_token = pretrain_use_cls_token

        window_block_indexes = [
            i for i in range(depth) if i not in set(global_att_blocks)
        ]
        self.full_attn_ids = list(global_att_blocks)
        self.rel_pos_blocks = [False] * depth
        if isinstance(rel_pos_blocks, bool):
            if rel_pos_blocks:
                self.rel_pos_blocks = [True] * depth
        else:
            for i in rel_pos_blocks:
                self.rel_pos_blocks[i] = True

        self.retain_cls_token = retain_cls_token
        if self.retain_cls_token:
            assert pretrain_use_cls_token
            assert len(window_block_indexes) == 0, (
                "windowing not supported with cls token"
            )
            assert sum(self.rel_pos_blocks) == 0, "rel pos not supported with cls token"

            scale = embed_dim**-0.5
            self.class_embedding = scale * mx.random.normal((1, 1, embed_dim))

        resolved_norm: NormFactory
        if isinstance(norm_layer, str):
            resolved_norm = cast(
                NormFactory, partial(getattr(nn, norm_layer), eps=1e-5)
            )
        else:
            resolved_norm = norm_layer

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=bias_patch_embed,
        )

        self.tile_abs_pos = tile_abs_pos
        self.use_abs_pos = use_abs_pos
        if self.tile_abs_pos:
            assert self.use_abs_pos

        if self.use_abs_pos:
            num_patches = (pretrain_img_size // patch_size) * (
                pretrain_img_size // patch_size
            )
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed: mx.array | None = mx.zeros((1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        dpr = [float(x.item()) for x in _linspace(0.0, drop_path_rate, depth)]

        self.blocks: list[Block] = []
        cur_stage = 1
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=resolved_norm,
                act_layer=act_layer,
                use_rel_pos=self.rel_pos_blocks[i],
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                use_rope=use_rope,
                rope_pt_size=(
                    (window_size, window_size)
                    if rope_pt_size is None
                    else (rope_pt_size, rope_pt_size)
                ),
                rope_interp=use_interp_rope,
                cls_token=self.retain_cls_token,
                dropout=dropout,
                init_values=init_values,
            )

            if i not in window_block_indexes:
                cur_stage += 1
            self.blocks.append(block)

        self.use_act_checkpoint = use_act_checkpoint
        self.persist_exact_windows = persist_exact_windows

        self.window_block_indexes = window_block_indexes
        self.return_interm_layers = return_interm_layers
        self.channel_list = (
            [embed_dim] * len(self.full_attn_ids)
            if return_interm_layers
            else [embed_dim]
        )

        if self.pos_embed is not None:
            self.pos_embed = (
                mx.random.truncated_normal(
                    lower=-2, upper=2, shape=self.pos_embed.shape
                )
                * 0.02
            )

        self.ln_pre = (
            _as_array_module(resolved_norm(embed_dim))
            if ln_pre
            else _as_array_module(nn.Identity())
        )
        self.ln_post = (
            _as_array_module(resolved_norm(embed_dim))
            if ln_post
            else _as_array_module(nn.Identity())
        )

        self._init_weights(self)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            lin = _as_linear(m)
            lin.weight = (
                mx.random.truncated_normal(lower=-2, upper=2, shape=lin.weight.shape)
                * 0.02
            )
            if lin.bias is not None:
                lin.bias = mx.zeros_like(lin.bias)
        elif isinstance(m, nn.LayerNorm):
            norm = _as_layer_norm(m)
            norm.weight = mx.ones_like(norm.weight)
            norm.bias = mx.zeros_like(norm.bias)

    def _prepare_tokens(
        self, x: mx.array | NestedTensor
    ) -> tuple[mx.array, int, bool, mx.array | None]:
        mask: mx.array | None = None
        if isinstance(x, NestedTensor):
            is_nested = True
            nested = _as_nested(x)
            image = nested.tensors
            mask = nested.mask
        else:
            is_nested = False
            image = x

        tokens = self.patch_embed(image)
        height = int(tokens.shape[1])
        width = int(tokens.shape[2])

        s = 0
        if self.retain_cls_token:
            tokens = mx.concat(
                [
                    self.class_embedding,
                    mx.reshape(
                        tokens, (int(tokens.shape[0]), -1, int(tokens.shape[-1]))
                    ),
                ],
                axis=1,
            )
            s = 1

        if self.pos_embed is not None:
            tokens = tokens + get_abs_pos(
                self.pos_embed,
                self.pretrain_use_cls_token,
                (height, width),
                self.retain_cls_token,
                tiling=self.tile_abs_pos,
            )

        return self.ln_pre(tokens), s, is_nested, mask

    def forward(self, x: mx.array | NestedTensor) -> list[FeatureMap]:
        tokens, s, is_nested, mask = self._prepare_tokens(x)
        return self._forward_blocks(tokens, s=s, is_nested=is_nested, mask=mask)

    def _spatial_feats(self, tokens: mx.array, s: int) -> mx.array:
        feats = tokens[:, s:]
        if feats.ndim == 4:
            return mx.transpose(feats, axes=(0, 3, 1, 2))
        assert feats.ndim == 3
        side = int(math.sqrt(int(feats.shape[1])))
        assert side * side == int(feats.shape[1])
        return mx.transpose(
            mx.reshape(
                feats,
                (int(feats.shape[0]), side, side, int(feats.shape[-1])),
            ),
            axes=(0, 3, 1, 2),
        )

    def _forward_blocks(
        self,
        tokens: mx.array,
        *,
        s: int,
        is_nested: bool,
        mask: mx.array | None,
    ) -> list[FeatureMap]:
        # Branch-local collectors keep NestedTensor vs array result typing precise.
        nested_outputs: list[FeatureMap] = []
        array_outputs: list[FeatureMap] = []
        masks: mx.array | None = None
        index = 0
        block_count = len(self.blocks)
        while index < block_count:
            block = self.blocks[index]
            persist = (
                self.persist_exact_windows
                and tokens.ndim == 4
                and block.window_size > 0
                and window_layout_is_exact(
                    int(tokens.shape[1]),
                    int(tokens.shape[2]),
                    block.window_size,
                )
            )
            if persist:
                window_size = block.window_size
                height = int(tokens.shape[1])
                width = int(tokens.shape[2])
                stop = index
                while (
                    stop < block_count
                    and self.blocks[stop].window_size == window_size
                ):
                    stop += 1
                windowed, pad_hw = window_partition(tokens, window_size)
                for group_index in range(index, stop):
                    windowed = self.blocks[group_index].forward(
                        windowed, windowed=True
                    )
                tokens = window_unpartition(
                    windowed, window_size, pad_hw, (height, width)
                )
                index = stop
                continue

            tokens = block(tokens)
            if (index == self.full_attn_ids[-1]) or (
                self.return_interm_layers and index in self.full_attn_ids
            ):
                if index == self.full_attn_ids[-1]:
                    tokens = self.ln_post(tokens)
                feats = self._spatial_feats(tokens, s)
                if is_nested:
                    if mask is not None and masks is None:
                        masks = _interpolate(
                            mask[None].astype(mx.float32),
                            size=(int(feats.shape[-2]), int(feats.shape[-1])),
                        ).astype(mx.bool_)[0]
                    nested_outputs.append(NestedTensor(feats, masks))
                else:
                    array_outputs.append(feats)
            index += 1
        return nested_outputs if is_nested else array_outputs

    def __call__(self, x: mx.array | NestedTensor) -> list[FeatureMap]:
        return self.forward(x)

    def get_layer_id(self, layer_name: str) -> int:
        num_layers = self.get_num_layers()
        if "rel_pos" in layer_name:
            return num_layers + 1
        if "ln_pre" in layer_name:
            return 0
        if "pos_embed" in layer_name or "cls_token" in layer_name:
            return 0
        if "patch_embed" in layer_name:
            return 0
        if "blocks" in layer_name:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        return num_layers + 1

    def get_num_layers(self) -> int:
        return len(self.blocks)
