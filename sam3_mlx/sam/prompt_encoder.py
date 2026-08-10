from __future__ import annotations

import math
from importlib import import_module
from typing import Protocol, TypeAlias, cast

import mlx.core as mx

from sam3_mlx.sam.common import Conv2dNCHW, LayerNorm2d


_ImageSize: TypeAlias = tuple[int, int]
_PointPrompts: TypeAlias = tuple[mx.array, mx.array]


class _ArrayMethods(Protocol):
    def reshape(self, *shape: int) -> mx.array: ...

    def transpose(self, *axes: int) -> mx.array: ...


class _ArrayModule(Protocol):
    def __call__(self, x: mx.array) -> mx.array: ...


class _EmbeddingModule(Protocol):
    @property
    def weight(self) -> mx.array: ...


class _ActivationFactory(Protocol):
    def __call__(self) -> _ArrayModule: ...


class _EmbeddingFactory(Protocol):
    def __call__(self, num_embeddings: int, dims: int) -> _EmbeddingModule: ...


def _reshape(array: mx.array, *shape: int) -> mx.array:
    return cast(_ArrayMethods, array).reshape(*shape)


def _transpose(array: mx.array, *axes: int) -> mx.array:
    return cast(_ArrayMethods, array).transpose(*axes)


def _label_mask(labels: mx.array, value: int) -> mx.array:
    return mx.expand_dims(cast(mx.array, labels == value), axis=-1)


_nn = import_module("mlx.nn")
_Module = cast(type[object], getattr(_nn, "Module"))
_gelu = cast(_ActivationFactory, getattr(_nn, "GELU"))
_embedding = cast(_EmbeddingFactory, getattr(_nn, "Embedding"))


class PromptEncoder(_Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: _ImageSize,
        input_image_size: _ImageSize,
        mask_in_chans: int,
        activation: _ActivationFactory = _gelu,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings = 4
        self.point_embeddings: list[_EmbeddingModule] = [
            _embedding(1, embed_dim) for _ in range(self.num_point_embeddings)
        ]
        self.not_a_point_embed = _embedding(1, embed_dim)

        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_downscaling: list[_ArrayModule] = [
            Conv2dNCHW(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            Conv2dNCHW(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            Conv2dNCHW(mask_in_chans, embed_dim, kernel_size=1),
        ]
        self.no_mask_embed = _embedding(1, embed_dim)

    def get_dense_pe(self) -> mx.array:
        return self.pe_layer(self.image_embedding_size)[None, ...]

    def _embed_points(self, points: mx.array, labels: mx.array, pad: bool) -> mx.array:
        points = mx.array(points, dtype=mx.float32) + 0.5
        labels = mx.array(labels)
        if pad:
            padding_point = mx.zeros((points.shape[0], 1, 2), dtype=points.dtype)
            padding_label = -mx.ones((labels.shape[0], 1), dtype=labels.dtype)
            points = mx.concat([points, padding_point], axis=1)
            labels = mx.concat([labels, padding_label], axis=1)

        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )
        point_embedding = mx.where(
            _label_mask(labels, -1),
            mx.zeros_like(point_embedding) + self.not_a_point_embed.weight,
            point_embedding,
        )
        for label_value, embedding in enumerate(self.point_embeddings):
            point_embedding = mx.where(
                _label_mask(labels, label_value),
                point_embedding + embedding.weight,
                point_embedding,
            )
        return point_embedding

    def _embed_boxes(self, boxes: mx.array) -> mx.array:
        boxes = mx.array(boxes, dtype=mx.float32) + 0.5
        coords = _reshape(boxes, -1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size
        )
        corner_offsets = mx.concat(
            [self.point_embeddings[2].weight, self.point_embeddings[3].weight],
            axis=0,
        )
        return corner_embedding + corner_offsets[None, :, :]

    def _embed_masks(self, masks: mx.array) -> mx.array:
        x = mx.array(masks, dtype=mx.float32)
        for layer in self.mask_downscaling:
            x = layer(x)
        return x

    def _get_batch_size(
        self,
        points: _PointPrompts | None,
        boxes: mx.array | None,
        masks: mx.array | None,
    ) -> int:
        if points is not None:
            return points[0].shape[0]
        if boxes is not None:
            return boxes.shape[0]
        if masks is not None:
            return masks.shape[0]
        return 1

    def __call__(
        self,
        points: _PointPrompts | None,
        boxes: mx.array | None,
        masks: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        batch_size = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = mx.zeros((batch_size, 0, self.embed_dim))
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = mx.concat([sparse_embeddings, point_embeddings], axis=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = mx.concat([sparse_embeddings, box_embeddings], axis=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = mx.broadcast_to(
                _reshape(self.no_mask_embed.weight, 1, -1, 1, 1),
                (
                    batch_size,
                    self.embed_dim,
                    self.image_embedding_size[0],
                    self.image_embedding_size[1],
                ),
            )
        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(_Module):
    def __init__(self, num_pos_feats: int = 64, scale: float | None = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.positional_encoding_gaussian_matrix = scale * mx.random.normal(
            (2, num_pos_feats)
        )

    def _pe_encoding(self, coords: mx.array) -> mx.array:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = (2 * math.pi) * coords
        return mx.concat([mx.sin(coords), mx.cos(coords)], axis=-1)

    def __call__(self, size: _ImageSize) -> mx.array:
        height, width = size
        y_embed = (mx.arange(height, dtype=mx.float32)[:, None] + 0.5) / height
        x_embed = (mx.arange(width, dtype=mx.float32)[None, :] + 0.5) / width
        y_embed = mx.broadcast_to(y_embed, (height, width))
        x_embed = mx.broadcast_to(x_embed, (height, width))
        pe = self._pe_encoding(mx.stack([x_embed, y_embed], axis=-1))
        return _transpose(pe, 2, 0, 1)

    def forward_with_coords(
        self,
        coords_input: mx.array,
        image_size: _ImageSize,
    ) -> mx.array:
        coords = mx.array(coords_input, dtype=mx.float32)
        scale = mx.array([image_size[1], image_size[0]], dtype=mx.float32)
        return self._pe_encoding(coords / scale)
