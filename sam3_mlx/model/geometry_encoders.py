from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, cast, overload

import mlx.core as mx
from mlx import nn


from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.model.grid_sample_mlx import grid_sample

from sam3_mlx.model import box_ops
from sam3_mlx.model.model_misc import CloneableModule, get_clones
from sam3_mlx.model.roi_align import roi_align


class _ArrayModule(Protocol):
    def __call__(self, x: mx.array) -> mx.array: ...


class _EmbeddingModule(_ArrayModule, Protocol):
    weight: mx.array


class _ArrayTransforms(Protocol):
    def reshape(self, *shape: int, stream: object | None = None) -> mx.array: ...

    def transpose(self, *axes: int, stream: object | None = None) -> mx.array: ...


class _BoxCxcywhToXyxy(Protocol):
    def __call__(self, boxes: mx.array) -> mx.array: ...


class _PositionEncoding(Protocol):
    def _encode_xy(self, x: mx.array, y: mx.array) -> tuple[mx.array, mx.array]: ...

    def encode_boxes(
        self, x: mx.array, y: mx.array, w: mx.array, h: mx.array
    ) -> mx.array: ...


class _PositionEncodingAccess(_PositionEncoding, Protocol):
    @staticmethod
    def encode_xy(
        position_encoding: _PositionEncoding, x: mx.array, y: mx.array
    ) -> tuple[mx.array, mx.array]:
        return position_encoding._encode_xy(x, y)


class _GeometryEncoderLayer(Protocol):
    def __call__(
        self,
        *,
        tgt: mx.array,
        memory: mx.array,
        tgt_key_padding_mask: mx.array,
        pos: mx.array,
    ) -> mx.array: ...


def _as_array_module(module: object) -> _ArrayModule:
    return cast(_ArrayModule, module)


def _as_embedding(module: object) -> _EmbeddingModule:
    return cast(_EmbeddingModule, module)


def _as_geometry_encoder_layer(module: object) -> _GeometryEncoderLayer:
    return cast(_GeometryEncoderLayer, module)


def _array_transforms(array: mx.array) -> _ArrayTransforms:
    return cast(_ArrayTransforms, array)


def _box_cxcywh_to_xyxy(boxes: mx.array) -> mx.array:
    convert = cast(_BoxCxcywhToXyxy, getattr(box_ops, "box_cxcywh_to_xyxy"))
    return convert(boxes)


def is_right_padded(mask: mx.array) -> bool | mx.array:
    """Given a padding mask (following pytorch convention, 1s for padded values),
    returns whether the padding is on the right or not."""
    # Handle empty masks (zero-sized dimension)
    if mask.size == 0 or (len(mask.shape) > 0 and mask.shape[-1] == 0):
        return True

    mask_int = mask.astype(mx.int64)
    sorted_mask = mx.sort(mask_int, axis=-1)
    return mx.all(cast(mx.array, mask_int == sorted_mask))


@overload
def concat_padded_sequences(
    seq1: mx.array,
    mask1: mx.array,
    seq2: mx.array,
    mask2: mx.array,
    return_index: Literal[False] = False,
) -> tuple[mx.array, mx.array]: ...


@overload
def concat_padded_sequences(
    seq1: mx.array,
    mask1: mx.array,
    seq2: mx.array,
    mask2: mx.array,
    return_index: Literal[True],
) -> tuple[mx.array, mx.array, mx.array]: ...


def concat_padded_sequences(
    seq1: mx.array,
    mask1: mx.array,
    seq2: mx.array,
    mask2: mx.array,
    return_index: bool = False,
) -> tuple[mx.array, mx.array] | tuple[mx.array, mx.array, mx.array]:
    """
    Concatenates two right-padded sequences, such that the resulting sequence
    is contiguous and also right-padded.

    Following pytorch's convention, tensors are sequence first, and the mask are
    batch first, with 1s for padded values.

    :param seq1: A tensor of shape (seq1_length, batch_size, hidden_size).
    :param mask1: A tensor of shape (batch_size, seq1_length).
    :param seq2: A tensor of shape (seq2_length, batch_size,  hidden_size).
    :param mask2: A tensor of shape (batch_size, seq2_length).
    :param return_index: If True, also returns the index of the ids of the element of seq2
        in the concatenated sequence. This can be used to retrieve the elements of seq2
    :return: A tuple (concatenated_sequence, concatenated_mask) if return_index is False,
        otherwise (concatenated_sequence, concatenated_mask, index).
    """
    seq1_length, batch_size, hidden_size = seq1.shape
    seq2_length, batch_size, hidden_size = seq2.shape

    assert (
        batch_size == seq1.shape[1] == seq2.shape[1] == mask1.shape[0] == mask2.shape[0]
    )
    assert hidden_size == seq1.shape[2] == seq2.shape[2]
    assert seq1_length == mask1.shape[1]
    assert seq2_length == mask2.shape[1]

    assert is_right_padded(mask1)
    assert is_right_padded(mask2)

    actual_seq1_lengths = (~mask1).sum(axis=-1)
    actual_seq2_lengths = (~mask2).sum(axis=-1)

    final_lengths = actual_seq1_lengths + actual_seq2_lengths
    max_length = seq1_length + seq2_length

    concatenated_mask = (
        mx.tile(mx.arange(max_length)[None], (batch_size, 1)) >= final_lengths[:, None]
    )
    # (max_len, batch_size, hidden_size)
    index = mx.tile(mx.arange(seq2_length)[:, None], (1, batch_size))
    index = index + actual_seq1_lengths[None]

    positions = mx.tile(mx.arange(max_length)[:, None], (1, batch_size))
    seq2_start = actual_seq1_lengths[None]
    seq2_end = seq2_start + seq2_length
    use_seq2 = (positions >= seq2_start) & (positions < seq2_end)

    sentinel_index = max_length
    seq1_source = mx.where(positions < seq1_length, positions, sentinel_index)
    seq2_source = seq1_length + positions - seq2_start
    source_index = mx.where(use_seq2, seq2_source, seq1_source).astype(mx.int64)
    source_index = mx.broadcast_to(
        source_index[:, :, None], (max_length, batch_size, hidden_size)
    )

    zero_pad = mx.zeros((1, batch_size, hidden_size), dtype=seq2.dtype)
    source_sequence = mx.concat([seq1.astype(seq2.dtype), seq2, zero_pad], axis=0)
    concatenated_sequence = mx.take_along_axis(source_sequence, source_index, axis=0)

    if return_index:
        return concatenated_sequence, concatenated_mask, index

    return concatenated_sequence, concatenated_mask


class Prompt:
    """Utility class to manipulate geometric prompts.

    We expect the sequences in pytorch convention, that is sequence first, batch second
    The dimensions are expected as follows:
    box_embeddings shape: N_boxes x B x C_box
    box_mask shape: B x N_boxes. Can be None if nothing is masked out
    point_embeddings shape: N_points x B x C_point
    point_mask shape: B x N_points. Can be None if nothing is masked out
    mask_embeddings shape: N_masks x B x 1 x H_mask x W_mask
    mask_mask shape: B x N_masks. Can be None if nothing is masked out

    We also store positive/negative labels. These tensors are also stored batch-first
    If they are None, we'll assume positive labels everywhere
    box_labels: long tensor of shape N_boxes x B
    point_labels: long tensor of shape N_points x B
    mask_labels: long tensor of shape N_masks x B
    """

    box_embeddings: mx.array | None
    box_mask: mx.array | None
    box_labels: mx.array | None
    point_embeddings: mx.array | None
    point_mask: mx.array | None
    point_labels: mx.array | None
    mask_embeddings: mx.array | None
    mask_mask: mx.array | None
    mask_labels: mx.array | None

    def __init__(
        self,
        box_embeddings: mx.array | None = None,
        box_mask: mx.array | None = None,
        point_embeddings: mx.array | None = None,
        point_mask: mx.array | None = None,
        box_labels: mx.array | None = None,
        point_labels: mx.array | None = None,
        mask_embeddings: mx.array | None = None,
        mask_mask: mx.array | None = None,  # Attention mask for mask prompt
        mask_labels: mx.array | None = None,
    ) -> None:
        # Check for null prompt
        if (
            box_embeddings is None
            and point_embeddings is None
            and mask_embeddings is None
        ):
            self.box_embeddings = None
            self.box_labels = None
            self.box_mask = None
            self.point_embeddings = None
            self.point_labels = None
            self.point_mask = None
            self.mask_embeddings = None
            self.mask_mask = None
            # Masks are assumed positive only for now.
            self.mask_labels = None
            return
        # Get sequence lengths and device
        box_seq_len, point_seq_len, mask_seq_len, bs = self._init_seq_len(
            box_embeddings, point_embeddings, mask_embeddings
        )

        # Initialize embeds, labels, attention masks.
        box_embeddings, box_labels, box_mask = self._init_box(
            box_embeddings, box_labels, box_mask, box_seq_len, bs
        )
        point_embeddings, point_labels, point_mask = self._init_point(
            point_embeddings, point_labels, point_mask, point_seq_len, bs
        )
        mask_embeddings, mask_labels, mask_mask = self._init_mask(
            mask_embeddings, mask_labels, mask_mask, mask_seq_len, bs
        )

        # Dimension checks
        assert box_embeddings is not None and list(box_embeddings.shape[:2]) == [
            box_seq_len,
            bs,
        ], (
            f"Wrong dimension for box embeddings. Expected [{box_seq_len}, {bs}, *] got {box_embeddings.shape}"
        )
        assert box_mask is not None and list(box_mask.shape) == [
            bs,
            box_seq_len,
        ], (
            f"Wrong dimension for box mask. Expected [{bs}, {box_seq_len}] got {box_mask.shape}"
        )
        assert point_embeddings is not None and list(point_embeddings.shape[:2]) == [
            point_seq_len,
            bs,
        ], (
            f"Wrong dimension for point embeddings. Expected [{point_seq_len}, {bs}, *] got {point_embeddings.shape}"
        )
        assert point_mask is not None and list(point_mask.shape) == [
            bs,
            point_seq_len,
        ], (
            f"Wrong dimension for point mask. Expected [{bs}, {point_seq_len}] got {point_mask.shape}"
        )
        assert box_labels is not None and list(box_labels.shape) == [
            box_seq_len,
            bs,
        ], (
            f"Wrong dimension for box labels. Expected [{box_seq_len}, {bs}] got {box_labels.shape}"
        )
        assert point_labels is not None and list(point_labels.shape) == [
            point_seq_len,
            bs,
        ], (
            f"Wrong dimension for point labels. Expected [{point_seq_len}, {bs}] got {point_labels.shape}"
        )
        assert (
            # Allowed to be None, we leave it to the encoder to check for validity before encoding.
            mask_embeddings is None
            or list(mask_embeddings.shape[:2])
            == [
                mask_seq_len,
                bs,
            ]
        ), (
            f"Wrong dimension for mask embeddings. Expected [{mask_seq_len}, {bs}, *] got {mask_embeddings.shape}"
        )
        assert mask_mask is None or list(mask_mask.shape) == [
            bs,
            mask_seq_len,
        ], (
            f"Wrong dimension for mask attn. mask. Expected [{bs}, {mask_seq_len}] got {mask_mask.shape}"
        )

        self.box_embeddings = box_embeddings
        self.point_embeddings = point_embeddings
        self.box_mask = box_mask
        self.point_mask = point_mask
        self.box_labels = box_labels
        self.point_labels = point_labels
        self.mask_embeddings = mask_embeddings
        self.mask_labels = mask_labels
        self.mask_mask = mask_mask

    def _init_seq_len_and_device(
        self,
        box_embeddings: mx.array | None,
        point_embeddings: mx.array | None,
        mask_embeddings: mx.array | None,
    ) -> tuple[int, int, int, int, object | None]:
        box_seq_len = point_seq_len = mask_seq_len = 0
        bs = None
        device = None
        if box_embeddings is not None:
            bs = box_embeddings.shape[1]
            box_seq_len = box_embeddings.shape[0]
            device = getattr(box_embeddings, "device", None)

        if point_embeddings is not None:
            point_seq_len = point_embeddings.shape[0]
            if bs is not None:
                assert bs == point_embeddings.shape[1], (
                    f"Batch size mismatch between box and point embeddings. Got {bs} and {point_embeddings.shape[1]}."
                )
            else:
                bs = point_embeddings.shape[1]
            point_device = getattr(point_embeddings, "device", None)
            if device is not None and point_device is not None:
                assert device == point_device, (
                    "Device mismatch between box and point embeddings"
                )
            elif device is None:
                device = point_device

        if mask_embeddings is not None:
            mask_seq_len = mask_embeddings.shape[0]
            if bs is not None:
                assert bs == mask_embeddings.shape[1], (
                    f"Batch size mismatch between box/point and mask embedding. Got {bs} and {mask_embeddings.shape[1]}"
                )
            else:
                bs = mask_embeddings.shape[1]
            mask_device = getattr(mask_embeddings, "device", None)
            if device is not None and mask_device is not None:
                assert device == mask_device, (
                    "Device mismatch between box/point and mask embeddings."
                )
            elif device is None:
                device = mask_device

        assert bs is not None
        return box_seq_len, point_seq_len, mask_seq_len, bs, device

    def _init_seq_len(
        self,
        box_embeddings: mx.array | None,
        point_embeddings: mx.array | None,
        mask_embeddings: mx.array | None,
    ) -> tuple[int, int, int, int]:
        box_seq_len, point_seq_len, mask_seq_len, bs, _ = self._init_seq_len_and_device(
            box_embeddings, point_embeddings, mask_embeddings
        )
        return box_seq_len, point_seq_len, mask_seq_len, bs

    def _init_box(
        self,
        box_embeddings: mx.array | None,
        box_labels: mx.array | None,
        box_mask: mx.array | None,
        box_seq_len: int,
        bs: int,
        device: object | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        del device
        if box_embeddings is None:
            box_embeddings = mx.zeros((box_seq_len, bs, 4))
        if box_labels is None:
            box_labels = mx.ones((box_seq_len, bs), dtype=mx.int64)
        if box_mask is None:
            box_mask = mx.zeros((bs, box_seq_len), dtype=mx.bool_)
        return box_embeddings, box_labels, box_mask

    def _init_point(
        self,
        point_embeddings: mx.array | None,
        point_labels: mx.array | None,
        point_mask: mx.array | None,
        point_seq_len: int,
        bs: int,
        device: object | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Initialize point embeddings and mask. C=2 for points (vs. 4 for boxes)."""
        del device
        if point_embeddings is None:
            point_embeddings = mx.zeros((point_seq_len, bs, 2))
        if point_labels is None:
            point_labels = mx.ones((point_seq_len, bs), dtype=mx.int64)
        if point_mask is None:
            point_mask = mx.zeros((bs, point_seq_len), dtype=mx.bool_)
        return point_embeddings, point_labels, point_mask

    def _init_mask(
        self,
        mask_embeddings: mx.array | None,
        mask_labels: mx.array | None,
        mask_mask: mx.array | None,
        mask_seq_len: int,
        bs: int,
        device: object | None = None,
    ) -> tuple[mx.array | None, mx.array, mx.array]:
        del device
        # NOTE: Mask embeddings can be of arbitrary resolution, so we don't initialize it here.
        # In case we append new mask, we check that its resolution matches exisiting ones (if any).
        # In case mask_embeddings is None, we should never encode it.
        if mask_labels is None:
            mask_labels = mx.ones((mask_seq_len, bs), dtype=mx.int64)
        if mask_mask is None:
            mask_mask = mx.zeros((bs, mask_seq_len), dtype=mx.bool_)
        return mask_embeddings, mask_labels, mask_mask

    def append_boxes(
        self, boxes: mx.array, labels: mx.array, mask: mx.array | None = None
    ) -> None:
        if self.box_embeddings is None:
            self.box_embeddings = boxes
            self.box_labels = labels
            self.box_mask = mask
            return

        bs = self.box_embeddings.shape[1]
        assert boxes.shape[1] == labels.shape[1] == bs
        assert list(boxes.shape[:2]) == list(labels.shape[:2])
        if mask is None:
            mask = mx.zeros((bs, boxes.shape[0]), dtype=mx.bool_)

        assert self.box_labels is not None
        assert self.box_mask is not None
        self.box_labels, _ = concat_padded_sequences(
            self.box_labels[..., None], self.box_mask, labels[..., None], mask
        )
        self.box_labels = self.box_labels.squeeze(-1)
        self.box_embeddings, self.box_mask = concat_padded_sequences(
            self.box_embeddings, self.box_mask, boxes, mask
        )

    def append_points(
        self, points: mx.array, labels: mx.array, mask: mx.array | None = None
    ) -> None:
        if self.point_embeddings is None:
            self.point_embeddings = points
            self.point_labels = labels
            self.point_mask = mask
            return

        bs = self.point_embeddings.shape[1]
        assert points.shape[1] == labels.shape[1] == bs
        assert list(points.shape[:2]) == list(labels.shape[:2])
        if mask is None:
            mask = mx.zeros((bs, points.shape[0]), dtype=mx.bool_)

        assert self.point_labels is not None
        assert self.point_mask is not None
        self.point_labels, _ = concat_padded_sequences(
            self.point_labels[..., None], self.point_mask, labels[..., None], mask
        )
        self.point_labels = self.point_labels.squeeze(-1)
        self.point_embeddings, self.point_mask = concat_padded_sequences(
            self.point_embeddings, self.point_mask, points, mask
        )

    def append_masks(
        self,
        masks: mx.array,
        labels: mx.array | None = None,
        attn_mask: mx.array | None = None,
    ) -> None:
        if labels is not None:
            assert list(masks.shape[:2]) == list(labels.shape[:2])
        if self.mask_embeddings is None:
            self.mask_embeddings = masks
            mask_seq_len, bs = masks.shape[:2]
            if labels is None:
                self.mask_labels = mx.ones((mask_seq_len, bs), dtype=mx.int64)
            else:
                self.mask_labels = labels
            if attn_mask is None:
                self.mask_mask = mx.zeros((bs, mask_seq_len), dtype=mx.bool_)
            else:
                self.mask_mask = attn_mask
        else:
            raise_unsupported(
                "sam3_mlx.model.geometry_encoders.Prompt.append_masks(multiple_masks=True)",
                reason="image-interactivity",
                detail="Only one mask per prompt is supported.",
                alternative="one mask prompt per Prompt",
            )

    def clone(self) -> Prompt:
        return Prompt(
            box_embeddings=(
                None if self.box_embeddings is None else mx.array(self.box_embeddings)
            ),
            box_mask=None if self.box_mask is None else mx.array(self.box_mask),
            point_embeddings=(
                None
                if self.point_embeddings is None
                else mx.array(self.point_embeddings)
            ),
            point_mask=None if self.point_mask is None else mx.array(self.point_mask),
            box_labels=None if self.box_labels is None else mx.array(self.box_labels),
            point_labels=(
                None if self.point_labels is None else mx.array(self.point_labels)
            ),
        )


class MaskEncoder(nn.Module):
    """
    Base class for mask encoders.
    """

    def __init__(
        self,
        mask_downsampler: nn.Module,
        position_encoding: nn.Module,
    ) -> None:
        super().__init__()
        self.mask_downsampler = _as_array_module(mask_downsampler)
        self.position_encoding = _as_array_module(position_encoding)

    def forward(
        self,
        masks: mx.array,
        pix_feat: mx.array | None = None,
        *args: object,
        **kwargs: object,
    ) -> tuple[mx.array, mx.array]:
        del pix_feat, args, kwargs
        masks = self.mask_downsampler(masks)
        masks_pos = self.position_encoding(masks).astype(masks.dtype)

        return masks, masks_pos

    def __call__(
        self,
        masks: mx.array,
        pix_feat: mx.array | None = None,
        *args: object,
        **kwargs: object,
    ) -> tuple[mx.array, mx.array]:
        return self.forward(masks, pix_feat, *args, **kwargs)


class FusedMaskEncoder(MaskEncoder):
    """Official SAM3 fused mask encoder ported to MLX tensor layout."""

    def __init__(
        self,
        mask_downsampler: nn.Module,
        position_encoding: nn.Module,
        fuser: nn.Module,
        in_dim: int = 256,
        out_dim: int = 256,
    ) -> None:
        super().__init__(mask_downsampler, position_encoding)
        self.fuser = _as_array_module(fuser)
        self.out_proj: nn.Module = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)

    @staticmethod
    def _conv_nchw(conv: nn.Module, x: mx.array) -> mx.array:
        if isinstance(conv, nn.Identity):
            return _as_array_module(conv)(x)
        output = _as_array_module(conv)(_array_transforms(x).transpose(0, 2, 3, 1))
        return _array_transforms(output).transpose(0, 3, 1, 2)

    def forward(
        self,
        masks: mx.array,
        pix_feat: mx.array | None = None,
        *args: object,
        **kwargs: object,
    ) -> tuple[mx.array, mx.array]:
        del args, kwargs
        if pix_feat is None:
            raise TypeError("pix_feat is required for FusedMaskEncoder")
        masks = self.mask_downsampler(masks)

        x = self._conv_nchw(self.pix_feat_proj, pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self._conv_nchw(self.out_proj, x)

        pos = self.position_encoding(x).astype(x.dtype)

        return x, pos

    def __call__(
        self,
        masks: mx.array,
        pix_feat: mx.array | None = None,
        *args: object,
        **kwargs: object,
    ) -> tuple[mx.array, mx.array]:
        return self.forward(masks, pix_feat, *args, **kwargs)


class SequenceGeometryEncoder(nn.Module):
    def __init__(
        self,
        encode_boxes_as_points: bool,
        points_direct_project: bool,
        points_pool: bool,
        points_pos_enc: bool,
        boxes_direct_project: bool,
        boxes_pool: bool,
        boxes_pos_enc: bool,
        d_model: int,
        pos_enc: _PositionEncoding | None,
        num_layers: int,
        layer: CloneableModule[nn.Module] | None,
        roi_size: int = 7,  # for boxes pool
        add_cls: bool = True,
        add_post_encode_proj: bool = True,
        mask_encoder: MaskEncoder | None = None,
        add_mask_label: bool = False,
        use_act_ckpt: bool = False,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.pos_enc = pos_enc
        self.encode_boxes_as_points = encode_boxes_as_points
        self.roi_size = roi_size
        # There usually are two labels: positive and negatives.
        # If we encode boxes as points, we have 3 types of points: regular, top left, bottom right
        # These 3 types can be positives or negatives, hence 2*3 = 6 labels
        num_labels = 6 if self.encode_boxes_as_points else 2
        self.label_embed = _as_embedding(nn.Embedding(num_labels, self.d_model))

        self.cls_embed: _EmbeddingModule | None = None
        if add_cls:
            self.cls_embed = _as_embedding(nn.Embedding(1, self.d_model))

        assert points_direct_project or points_pos_enc or points_pool, (
            "Error: need at least one way to encode points"
        )
        assert (
            encode_boxes_as_points
            or boxes_direct_project
            or boxes_pos_enc
            or boxes_pool
        ), "Error: need at least one way to encode boxes"

        self.points_direct_project: _ArrayModule | None = None
        if points_direct_project:
            self.points_direct_project = _as_array_module(nn.Linear(2, self.d_model))
        self.points_pool_project: _ArrayModule | None = None
        if points_pool:
            self.points_pool_project = _as_array_module(
                nn.Linear(self.d_model, self.d_model)
            )
        self.points_pos_enc_project: _ArrayModule | None = None
        if points_pos_enc:
            self.points_pos_enc_project = _as_array_module(
                nn.Linear(self.d_model, self.d_model)
            )

        self.boxes_direct_project: _ArrayModule | None = None
        self.boxes_pool_project: nn.Module | None = None
        self.boxes_pos_enc_project: _ArrayModule | None = None
        if not encode_boxes_as_points:
            if boxes_direct_project:
                self.boxes_direct_project = _as_array_module(nn.Linear(4, self.d_model))
            if boxes_pool:
                self.boxes_pool_project = nn.Conv2d(
                    self.d_model, self.d_model, self.roi_size
                )
            if boxes_pos_enc:
                self.boxes_pos_enc_project = _as_array_module(
                    nn.Linear(self.d_model + 2, self.d_model)
                )

        self.final_proj: _ArrayModule | None = None
        self.norm: _ArrayModule | None = None
        if add_post_encode_proj:
            self.final_proj = _as_array_module(nn.Linear(self.d_model, self.d_model))
            self.norm = _as_array_module(nn.LayerNorm(self.d_model))

        self.img_pre_norm = _as_array_module(nn.Identity())
        if self.points_pool_project is not None or self.boxes_pool_project is not None:
            self.img_pre_norm = _as_array_module(nn.LayerNorm(self.d_model))

        self.encode: list[nn.Module] | None = None
        self.encode_norm: _ArrayModule | None = None
        if num_layers > 0:
            assert add_cls, (
                "It's currently highly recommended to add a CLS when using a transformer"
            )
            assert layer is not None
            self.encode = get_clones(layer, num_layers)
            self.encode_norm = _as_array_module(nn.LayerNorm(self.d_model))

        self.mask_label_embed: _ArrayModule | None = None
        if mask_encoder is not None:
            assert isinstance(mask_encoder, MaskEncoder), (
                f"Expected mask_encoder of type MaskEncoder. Got {type(mask_encoder)}."
            )
            if add_mask_label:
                self.mask_label_embed = _as_array_module(nn.Embedding(2, self.d_model))
        self.add_mask_label = add_mask_label
        self.mask_encoder = mask_encoder
        self.use_act_ckpt = use_act_ckpt

    def _encode_points(
        self,
        points: mx.array,
        points_mask: mx.array,
        points_labels: mx.array,
        img_feats: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        points_embed: mx.array | None = None
        n_points, bs = points.shape[:2]

        if self.points_direct_project is not None:
            proj = self.points_direct_project(points)
            assert points_embed is None
            points_embed = proj

        if self.points_pool_project is not None:
            assert img_feats is not None
            # points are [Num_points, bs, 2], normalized in [0, 1]
            # the grid needs to be [Bs, H_out, W_out, 2] normalized in [-1,1]
            # Will take H_out = num_points, w_out = 1
            grid = _array_transforms(points).transpose(1, 0, 2)[:, :, None]
            grid = (grid * 2) - 1
            img_feats_nhwc = _array_transforms(img_feats).transpose(0, 2, 3, 1)
            sampled_nhwc = grid_sample(img_feats_nhwc, grid)
            sampled = _array_transforms(sampled_nhwc).transpose(0, 3, 1, 2)
            assert list(sampled.shape) == [bs, self.d_model, n_points, 1]
            sampled = _array_transforms(sampled.squeeze(-1)).transpose(2, 0, 1)
            proj = self.points_pool_project(sampled)
            if points_embed is None:
                points_embed = proj
            else:
                points_embed = points_embed + proj

        if self.points_pos_enc_project is not None:
            assert self.pos_enc is not None
            x, y = points[..., 0], points[..., 1]
            enc_x, enc_y = _PositionEncodingAccess.encode_xy(
                self.pos_enc, x.flatten(), y.flatten()
            )
            enc_x = _array_transforms(enc_x).reshape(n_points, bs, enc_x.shape[-1])
            enc_y = _array_transforms(enc_y).reshape(n_points, bs, enc_y.shape[-1])
            enc = mx.concat([enc_x, enc_y], axis=-1)

            proj = self.points_pos_enc_project(enc)
            if points_embed is None:
                points_embed = proj
            else:
                points_embed = points_embed + proj

        assert points_embed is not None
        type_embed = self.label_embed(points_labels.astype(mx.int64))
        return type_embed + points_embed, points_mask

    def _encode_boxes(
        self,
        boxes: mx.array,
        boxes_mask: mx.array,
        boxes_labels: mx.array,
        img_feats: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        boxes_embed: mx.array | None = None
        n_boxes, bs = boxes.shape[:2]

        if self.boxes_direct_project is not None:
            proj = self.boxes_direct_project(boxes)
            assert boxes_embed is None
            boxes_embed = proj

        if self.boxes_pool_project is not None:
            assert img_feats is not None
            H, W = img_feats.shape[-2:]

            # boxes are [Num_boxes, bs, 4], normalized in [0, 1]
            # We need to denormalize, and convert to [x, y, x, y]
            boxes_xyxy = _box_cxcywh_to_xyxy(boxes)
            scale = mx.array([W, H, W, H], dtype=boxes_xyxy.dtype)
            scale = _array_transforms(scale).reshape(1, 1, 4)
            boxes_xyxy = boxes_xyxy * scale
            # In Torch, boxes_inp is built using unbind(1) which returns a list of tensors for each batch element.
            # In MLX, we use a list comprehension. This correctly handles any batch size (including bs=0)
            # and avoids indexing into empty dimensions.
            boxes_inp = [boxes_xyxy[:, i, :].astype(mx.float32) for i in range(bs)]
            sampled = roi_align(img_feats, boxes_inp, self.roi_size, self.roi_size)
            assert list(sampled.shape) == [
                bs * n_boxes,
                self.d_model,
                self.roi_size,
                self.roi_size,
            ]
            sampled = _array_transforms(sampled).transpose(0, 2, 3, 1)
            proj = _as_array_module(self.boxes_pool_project)(sampled)
            proj = _array_transforms(
                _array_transforms(proj).reshape(bs, n_boxes, self.d_model)
            ).transpose(1, 0, 2)
            if boxes_embed is None:
                boxes_embed = proj
            else:
                boxes_embed = boxes_embed + proj

        if self.boxes_pos_enc_project is not None:
            assert self.pos_enc is not None
            cx, cy, w, h = (
                boxes[..., 0],
                boxes[..., 1],
                boxes[..., 2],
                boxes[..., 3],
            )
            enc = self.pos_enc.encode_boxes(
                cx.flatten(), cy.flatten(), w.flatten(), h.flatten()
            )

            enc = _array_transforms(enc).reshape(n_boxes, bs, enc.shape[-1])

            proj = self.boxes_pos_enc_project(enc)
            if boxes_embed is None:
                boxes_embed = proj
            else:
                boxes_embed = boxes_embed + proj

        assert boxes_embed is not None
        type_embed = self.label_embed(boxes_labels.astype(mx.int64))
        return type_embed + boxes_embed, boxes_mask

    def _encode_masks(
        self,
        masks: mx.array,
        attn_mask: mx.array,
        mask_labels: mx.array,
        img_feats: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        n_masks, bs = masks.shape[:2]
        assert n_masks == 1, (
            "We assume one mask per prompt for now. Code should still be functional if this assertion is removed."
        )
        assert list(attn_mask.shape) == [
            bs,
            n_masks,
        ], (
            f"Expected attn_mask to be of shape {bs}x{n_masks}. Got {list(attn_mask.shape)}."
        )
        assert self.mask_encoder is not None
        masks, pos = self.mask_encoder(
            masks=_array_transforms(masks)
            .reshape(-1, *masks.shape[2:])
            .astype(mx.float32),
            pix_feat=img_feats,
        )
        H, W = masks.shape[-2:]
        n_tokens_per_mask = H * W

        # NOTE: We add pos enc directly here because we don't track pos encoding
        # for the concatenated prompt (text plus other geometric prompts).
        masks = masks + pos
        masks = (
            _array_transforms(masks).reshape(n_masks, bs, *masks.shape[1:]).flatten(-2)
        )  # n_masks x bs x C x H*W
        masks = (
            _array_transforms(masks).transpose(0, 3, 1, 2).flatten(0, 1)
        )  # n_masks * H*W x bs x C
        attn_mask = mx.repeat(attn_mask, n_tokens_per_mask, axis=1)
        if self.add_mask_label:
            assert self.mask_label_embed is not None
            masks = masks + self.mask_label_embed(mask_labels.astype(mx.int64))
        return masks, attn_mask

    def forward(
        self,
        geo_prompt: Prompt,
        img_feats: Sequence[mx.array],
        img_sizes: Sequence[tuple[int, int]],
        img_pos_embeds: Sequence[mx.array] | None = None,
    ) -> tuple[mx.array, mx.array]:
        points = geo_prompt.point_embeddings
        points_mask = geo_prompt.point_mask
        points_labels = geo_prompt.point_labels
        boxes = geo_prompt.box_embeddings
        boxes_mask = geo_prompt.box_mask
        boxes_labels = geo_prompt.box_labels
        masks = geo_prompt.mask_embeddings
        masks_mask = geo_prompt.mask_mask
        masks_labels = geo_prompt.mask_labels
        assert points is not None
        assert points_mask is not None
        assert points_labels is not None
        assert boxes is not None
        assert boxes_mask is not None
        assert boxes_labels is not None
        seq_first_img_feats = img_feats[-1]  # [H*W, B, C]
        seq_first_img_pos_embeds = (
            img_pos_embeds[-1]
            if img_pos_embeds is not None
            else mx.zeros_like(seq_first_img_feats)
        )

        spatial_img_feats: mx.array | None = None
        if self.points_pool_project is not None or self.boxes_pool_project is not None:
            assert len(img_feats) == len(img_sizes)
            cur_img_feat = img_feats[-1]
            cur_img_feat = self.img_pre_norm(cur_img_feat)
            H, W = img_sizes[-1]
            assert cur_img_feat.shape[0] == H * W
            N, C = cur_img_feat.shape[-2:]
            # Put back in NxCxHxW
            cur_img_feat = _array_transforms(cur_img_feat).transpose(1, 2, 0)
            cur_img_feat = _array_transforms(cur_img_feat).reshape(N, C, H, W)
            spatial_img_feats = cur_img_feat

        if self.encode_boxes_as_points:
            assert boxes is not None
            assert geo_prompt.box_mask is not None
            assert geo_prompt.box_labels is not None
            assert boxes.shape[-1] == 4

            boxes_xyxy = _box_cxcywh_to_xyxy(boxes)
            top_left, bottom_right = mx.split(boxes_xyxy, 2, axis=-1)

            labels_tl = geo_prompt.box_labels + 2
            labels_br = geo_prompt.box_labels + 4

            # Append to the existing points
            points, _ = concat_padded_sequences(
                points, points_mask, top_left, boxes_mask
            )
            points_labels, points_mask = concat_padded_sequences(
                points_labels[..., None],
                points_mask,
                labels_tl[..., None],
                boxes_mask,
            )
            points_labels = points_labels.squeeze(-1)

            points, _ = concat_padded_sequences(
                points, points_mask, bottom_right, boxes_mask
            )
            points_labels, points_mask = concat_padded_sequences(
                points_labels[..., None],
                points_mask,
                labels_br[..., None],
                boxes_mask,
            )
            points_labels = points_labels.squeeze(-1)

        final_embeds, final_mask = self._encode_points(
            points=points,
            points_mask=points_mask,
            points_labels=points_labels,
            img_feats=spatial_img_feats,
        )

        if not self.encode_boxes_as_points:
            boxes_embeds, boxes_mask = self._encode_boxes(
                boxes=boxes,
                boxes_mask=boxes_mask,
                boxes_labels=boxes_labels,
                img_feats=spatial_img_feats,
            )

            final_embeds, final_mask = concat_padded_sequences(
                final_embeds, final_mask, boxes_embeds, boxes_mask
            )

        masks_embed: mx.array | None = None
        if masks is not None:
            assert self.mask_encoder is not None
            assert masks_mask is not None
            assert masks_labels is not None
            masks_embed, masks_mask = self._encode_masks(
                masks=masks,
                attn_mask=masks_mask,
                mask_labels=masks_labels,
                img_feats=spatial_img_feats,
            )
            if points.shape[0] == boxes.shape[0] == 0:
                return masks_embed, masks_mask
        bs = final_embeds.shape[1]
        assert final_mask.shape[0] == bs
        if self.cls_embed is not None:
            cls = mx.tile(
                _array_transforms(self.cls_embed.weight).reshape(1, 1, self.d_model),
                (1, bs, 1),
            )
            cls_mask = mx.zeros((bs, 1), dtype=final_mask.dtype)
            final_embeds, final_mask = concat_padded_sequences(
                final_embeds, final_mask, cls, cls_mask
            )

        if self.final_proj is not None:
            assert self.norm is not None
            final_embeds = self.norm(self.final_proj(final_embeds))

        if self.encode is not None:
            for lay in self.encode:
                final_embeds = _as_geometry_encoder_layer(lay)(
                    tgt=_array_transforms(final_embeds).transpose(1, 0, 2),
                    memory=_array_transforms(seq_first_img_feats).transpose(1, 0, 2),
                    tgt_key_padding_mask=final_mask,
                    pos=_array_transforms(seq_first_img_pos_embeds).transpose(1, 0, 2),
                )
                final_embeds = _array_transforms(final_embeds).transpose(1, 0, 2)
            assert self.encode_norm is not None
            final_embeds = self.encode_norm(final_embeds)
        # Finally, concat mask embeddings if any
        if masks is not None:
            assert masks_embed is not None
            assert masks_mask is not None
            final_embeds, final_mask = concat_padded_sequences(
                final_embeds, final_mask, masks_embed, masks_mask
            )
        return final_embeds, final_mask

    def __call__(
        self,
        geo_prompt: Prompt,
        img_feats: Sequence[mx.array],
        img_sizes: Sequence[tuple[int, int]],
        img_pos_embeds: Sequence[mx.array] | None = None,
    ) -> tuple[mx.array, mx.array]:
        return self.forward(
            geo_prompt=geo_prompt,
            img_feats=img_feats,
            img_sizes=img_sizes,
            img_pos_embeds=img_pos_embeds,
        )
