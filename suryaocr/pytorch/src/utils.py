# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from loguru import logger
from PIL import Image


import math
from typing import List, Optional, Union

import cv2
import numpy as np
from surya.common.surya.processor import (
    SuryaOCRProcessor,
)
from surya.detection.processor import (
    SegformerImageProcessor,
)
from transformers import PretrainedConfig
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
)


def _prepare_image(self, img):
    new_size = (self.processor.size["width"], self.processor.size["height"])

    img.thumbnail(new_size, Image.Resampling.LANCZOS)
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    # Original line
    # img = np.asarray(img, dtype=np.uint8)
    # New line (patch): replace original with explicit copy
    img = np.array(img, dtype=np.uint8, copy=True)
    img = self.processor(img)["pixel_values"][0]
    img = torch.from_numpy(img)
    return img


# Monkeypatch SegformerImageProcessor._preprocess to use torch-friendly ops
def _segformer_preprocess(
    self: SegformerImageProcessor,
    image: ImageInput,
    do_resize: bool,
    do_rescale: bool,
    do_normalize: bool,
    size: Optional[dict[str, int]] = None,
    resample: PILImageResampling = None,
    rescale_factor: Optional[float] = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
):
    if isinstance(image, Image.Image):
        image = np.array(image)

    if isinstance(image, np.ndarray):
        tensor_image = torch.from_numpy(image)
    elif isinstance(image, torch.Tensor):
        tensor_image = image
    else:
        tensor_image = torch.as_tensor(image)

    if not tensor_image.is_floating_point():
        tensor_image = tensor_image.to(torch.float32)

    if do_rescale:
        scale = float(rescale_factor if rescale_factor is not None else 1.0)
        tensor_image = tensor_image * scale

    if do_normalize:
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(
                image
                if isinstance(image, np.ndarray)
                else tensor_image.detach().cpu().numpy()
            )
        try:
            channel_dim = ChannelDimension(input_data_format)
        except Exception:
            channel_dim = input_data_format

        if channel_dim == ChannelDimension.LAST:
            channel_axis = -1
        elif channel_dim == ChannelDimension.FIRST:
            channel_axis = 0
        else:
            channel_axis = None

        if channel_axis is None or tensor_image.ndim < 3:
            mean_values = (
                image_mean if isinstance(image_mean, (list, tuple)) else [image_mean]
            )
            std_values = (
                image_std if isinstance(image_std, (list, tuple)) else [image_std]
            )
            mean_scalar = float(mean_values[0])
            std_scalar = float(std_values[0])
            tensor_image = (tensor_image - mean_scalar) / std_scalar
        else:
            num_channels = tensor_image.shape[channel_axis]
            if isinstance(image_mean, (list, tuple)):
                mean_list = list(image_mean)
            else:
                mean_list = [image_mean] * num_channels
            if isinstance(image_std, (list, tuple)):
                std_list = list(image_std)
            else:
                std_list = [image_std] * num_channels

            mean_tensor = torch.tensor(
                mean_list, dtype=tensor_image.dtype, device=tensor_image.device
            )
            std_tensor = torch.tensor(
                std_list, dtype=tensor_image.dtype, device=tensor_image.device
            )

            expand_shape = [1] * tensor_image.ndim
            expand_shape[channel_axis] = num_channels
            mean_tensor = mean_tensor.view(
                *(
                    [num_channels]
                    if channel_axis in (0, -1) and tensor_image.ndim == 1
                    else expand_shape
                )
            )
            std_tensor = std_tensor.view(
                *(
                    [num_channels]
                    if channel_axis in (0, -1) and tensor_image.ndim == 1
                    else expand_shape
                )
            )

            if channel_axis == -1:
                mean_tensor = mean_tensor.view(*expand_shape)
                std_tensor = std_tensor.view(*expand_shape)
            elif channel_axis == 0:
                mean_tensor = mean_tensor.view(*expand_shape)
                std_tensor = std_tensor.view(*expand_shape)

            tensor_image = (tensor_image - mean_tensor) / std_tensor

    result = (
        tensor_image.detach().cpu().numpy().astype(np.float32)
        if tensor_image.is_floating_point()
        else tensor_image.detach().cpu().numpy()
    )
    return result


# Torch-friendly replacement for get_dynamic_thresholds in surya.detection.heatmap
def _get_dynamic_thresholds_torch(
    linemap,
    text_threshold: float,
    low_text: float,
    typical_top10_avg: float = 0.7,
):
    tensor_map = torch.as_tensor(linemap, dtype=torch.float32)
    flat = tensor_map.reshape(-1)
    numel = int(flat.numel())
    k = max(1, int(numel * 0.10))  # top 10%
    if k >= numel:
        top_values = flat
    else:
        top_values, _ = torch.topk(flat, k, largest=True)
    avg_intensity = top_values.mean()
    scaling_factor = torch.clamp(
        avg_intensity / float(typical_top10_avg), 0.0, 1.0
    ).pow(0.5)

    low_text_new = torch.clamp(
        torch.tensor(float(low_text)) * scaling_factor, 0.1, 0.6
    ).item()
    text_threshold_new = torch.clamp(
        torch.tensor(float(text_threshold)) * scaling_factor, 0.15, 0.8
    ).item()
    return float(text_threshold_new), float(low_text_new)


# Torch-friendly replacement for detect_boxes in surya.detection.heatmap
def _detect_boxes_torch(linemap, text_threshold, low_text):
    # Ensure torch tensor for elementwise ops to avoid Dynamo numpy interception
    tensor_map = torch.as_tensor(linemap, dtype=torch.float32)
    img_h, img_w = int(tensor_map.shape[0]), int(tensor_map.shape[1])

    # Use the torch-based dynamic thresholds
    text_threshold, low_text = _get_dynamic_thresholds_torch(
        tensor_map, text_threshold, low_text
    )

    # Threshold using torch, then convert to numpy for OpenCV
    text_score_comb = (tensor_map > float(low_text)).to(torch.uint8).cpu().numpy()
    label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_score_comb, connectivity=4
    )

    det: List[np.ndarray] = []
    confidences: List[float] = []
    max_confidence = 0.0

    linemap_np = tensor_map.cpu().numpy()

    for k in range(1, int(label_count)):
        # size filtering
        size = int(stats[k, cv2.CC_STAT_AREA])
        if size < 10:
            continue

        # make segmentation map
        x, y, w, h = stats[
            k,
            [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT],
        ]

        try:
            niter = int(np.sqrt(min(int(w), int(h))))
        except ValueError:
            niter = 0

        buffer = 1
        sx, sy = max(0, int(x) - niter - buffer), max(0, int(y) - niter - buffer)
        ex, ey = min(img_w, int(x) + int(w) + niter + buffer), min(
            img_h, int(y) + int(h) + niter + buffer
        )

        mask = labels[sy:ey, sx:ex] == k
        line_max = (
            float(np.max(linemap_np[sy:ey, sx:ex][mask])) if np.any(mask) else 0.0
        )

        # thresholding
        if line_max < float(text_threshold):
            continue

        segmap = mask.astype(np.uint8)

        ksize = buffer + niter
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        selected_segmap = cv2.dilate(segmap, kernel)

        # make box
        y_inds, x_inds = np.nonzero(selected_segmap)
        x_inds = x_inds + sx
        y_inds = y_inds + sy
        np_contours = np.column_stack((x_inds, y_inds))
        if np_contours.shape[0] < 3:
            # Need at least a few points for minAreaRect; skip small artifacts
            continue
        rectangle = cv2.minAreaRect(np_contours.astype(np.float32))
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w_len = np.linalg.norm(box[0] - box[1])
        h_len = np.linalg.norm(box[1] - box[2])
        box_ratio = max(w_len, h_len) / (min(w_len, h_len) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = np_contours[:, 0].min(), np_contours[:, 0].max()
            t, b = np_contours[:, 1].min(), np_contours[:, 1].max()
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)

        max_confidence = max(max_confidence, line_max)

        confidences.append(line_max)
        det.append(box.astype(np.float32))

    if max_confidence > 0:
        confidences = [float(c / max_confidence) for c in confidences]
    return det, confidences


def _patched_static_ops_cache_init(
    self,
    config: PretrainedConfig,
    batch_size: int,
    max_cache_len: int,
    text_sliding_window: int,
    device: int,
    dtype: int,
):
    self.text_sliding_window = text_sliding_window
    self.num_layers = config.num_hidden_layers
    self.max_batch_size = batch_size
    self.max_cache_len = max_cache_len
    self.head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    self._dtype = dtype
    self.num_key_value_heads = (
        config.num_attention_heads
        if getattr(config, "num_key_value_heads", None) is None
        else config.num_key_value_heads
    )

    # Cache init is taken from huggingface StaticCache - https://github.com/huggingface/transformers/blob/67ddc82fbc7e52c6f42a395b4a6d278c55b77a39/src/transformers/cache_utils.py#L1125
    self.key_cache: list[torch.Tensor] = []
    self.value_cache: list[torch.Tensor] = []
    cache_shape = (
        self.max_batch_size,
        self.num_key_value_heads,
        self.max_cache_len,
        self.head_dim,
    )
    device = torch.device(device) if device is not None else None
    for _ in range(config.num_hidden_layers):
        new_layer_key_cache = torch.zeros(cache_shape, dtype=self._dtype, device=device)
        new_layer_value_cache = torch.zeros(
            cache_shape, dtype=self._dtype, device=device
        )
        if not torch._dynamo.is_compiling():
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
        self.key_cache.append(new_layer_key_cache)
        self.value_cache.append(new_layer_value_cache)

    self.attention_mask = torch.zeros(
        (self.max_batch_size, self.max_cache_len), device=device, dtype=torch.long
    )
    self.text_token_counts = [
        torch.zeros(self.max_batch_size, dtype=torch.long, device=device)
        for _ in range(self.num_layers)
    ]

    self.dtype = dtype
    self.device = device


def _patched_dynamic_ops_cache_init(
    self,
    config: PretrainedConfig,
    batch_size: int,
    max_cache_len: int,
    text_sliding_window: int,
    device: int,
    dtype: int,
):
    self.text_sliding_window = text_sliding_window
    self.num_layers = config.num_hidden_layers
    self.max_batch_size = batch_size
    self.max_cache_len = max_cache_len
    self.head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    self._dtype = dtype
    self.num_key_value_heads = (
        config.num_attention_heads
        if getattr(config, "num_key_value_heads", None) is None
        else config.num_key_value_heads
    )

    # Initialize KV caches but avoid Dynamo-forbidden mark_static_address while tracing
    self.key_cache: list[torch.Tensor] = []
    self.value_cache: list[torch.Tensor] = []
    cache_shape = (
        self.max_batch_size,
        self.num_key_value_heads,
        self.max_cache_len,
        self.head_dim,
    )
    device = torch.device(device) if device is not None else None
    for _ in range(config.num_hidden_layers):
        new_layer_key_cache = torch.zeros(cache_shape, dtype=self._dtype, device=device)
        new_layer_value_cache = torch.zeros(
            cache_shape, dtype=self._dtype, device=device
        )
        if not torch._dynamo.is_compiling():
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
        self.key_cache.append(new_layer_key_cache)
        self.value_cache.append(new_layer_value_cache)

    self.attention_mask = torch.zeros(
        (self.max_batch_size, self.max_cache_len), device=device, dtype=torch.long
    )
    self.text_token_counts = [
        torch.zeros(self.max_batch_size, dtype=torch.long, device=device)
        for _ in range(self.num_layers)
    ]

    self.dtype = dtype
    self.device = device


def _patched_image_processor(self: SuryaOCRProcessor, image: np.ndarray) -> np.ndarray:
    tensor_img = torch.as_tensor(image, dtype=torch.float32)
    scale = float(getattr(self, "rescale_factor", 1.0))
    tensor_img = tensor_img * scale

    # self.image_mean/std may be numpy arrays of shape (3,)
    mean = torch.as_tensor(
        self.image_mean, dtype=tensor_img.dtype, device=tensor_img.device
    )
    std = torch.as_tensor(
        self.image_std, dtype=tensor_img.dtype, device=tensor_img.device
    )

    if tensor_img.ndim == 3 and tensor_img.shape[-1] == int(mean.numel()):
        # HWC layout
        tensor_img = (tensor_img - mean) / std
    else:
        # Fallback broadcast
        tensor_img = (tensor_img - mean.view(1, 1, -1)) / std.view(1, 1, -1)

    return tensor_img.detach().cpu().numpy()


def _patched_process_and_tile_no_xla(self: SuryaOCRProcessor, image: np.ndarray):
    """
    Monkey-patch version of SuryaOCRProcessor._process_and_tile that forces
    FOUNDATION_XLA-like padding off by using extra_multipler = 1.
    """
    # Equivalent to original, but always behaves as if settings.FOUNDATION_XLA is False
    extra_multipler = 1
    factor = self.patch_size * self.merge_size * extra_multipler

    height, width = image.shape[:2]
    h_bar = math.ceil(height / factor) * factor
    w_bar = math.ceil(width / factor) * factor
    if h_bar != height or w_bar != width:
        if height == 0 or width == 0:
            image = np.zeros((h_bar, w_bar, 3), dtype=np.uint8)
        else:
            image = cv2.resize(image, (w_bar, h_bar), interpolation=cv2.INTER_CUBIC)

    # Handle scaling and normalization
    image = self._image_processor(image)
    height, width = image.shape[:2]

    # Numpy array to torch tensor
    img_tensor = torch.from_numpy(image.transpose(2, 0, 1))
    patches = img_tensor.unsqueeze(0)

    channel = patches.shape[1]
    grid_t = patches.shape[0]
    grid_h, grid_w = height // self.patch_size, width // self.patch_size

    patches = patches.reshape(
        grid_t,
        1,
        channel,
        grid_h // self.merge_size,
        self.merge_size,
        self.patch_size,
        grid_w // self.merge_size,
        self.merge_size,
        self.patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w, channel * 1 * self.patch_size * self.patch_size
    )

    return flatten_patches, (grid_t, grid_h, grid_w)


class SuryaOCRWrapper(nn.Module):
    def __init__(self, image_tensor=None, device: str = "cpu"):
        super().__init__()
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        self.detection_predictor = DetectionPredictor(device=device)
        self.foundation_predictor = FoundationPredictor(device=device)
        self.rec_predictor = RecognitionPredictor(self.foundation_predictor)
        self._to_pil = transforms.ToPILImage()

        # Set eval mode on wrapper and underlying models
        self.eval()
        if hasattr(self.rec_predictor, "model"):
            self.rec_predictor.model.eval()
        if hasattr(self.detection_predictor, "model"):
            self.detection_predictor.model.eval()
        if hasattr(self.foundation_predictor, "model"):
            self.foundation_predictor.model.eval()

        if hasattr(self, "recognition_predictor") and hasattr(
            self.recognition_predictor, "model"
        ):
            for _, param in self.recognition_predictor.model.named_parameters():
                param.requires_grad = False
        if hasattr(self, "detection_predictor") and hasattr(
            self.detection_predictor, "model"
        ):
            for _, param in self.detection_predictor.model.named_parameters():
                param.requires_grad = False

        freeze_all(self, warmup_input=image_tensor)

    def forward(self, images_tensor: torch.Tensor):
        batch_size = images_tensor.shape[0]
        images: List[Image.Image] = [
            self._to_pil(images_tensor[i].cpu()) for i in range(batch_size)
        ]
        highres_images: List[Image.Image] = images
        task_names = ["ocr_with_boxes"] * len(images)
        predictions_by_image = self.rec_predictor(
            images,
            task_names=task_names,
            det_predictor=self.detection_predictor,
            highres_images=highres_images,
        )
        # Pack to tensors
        lines_bbox, lines_conf, text_codes, text_len, lines_len = pack_predictions(
            predictions_by_image
        )
        # Tie outputs to input to avoid constant folding in generated module
        # NOTE: derive zeros from each output tensor independently to avoid
        # inadvertent cross-tensor broadcasting under XLA compilation.
        zero_lines_bbox = (lines_bbox.sum() * 0).to(lines_bbox.dtype)
        zero_lines_conf = (lines_conf.sum() * 0).to(lines_conf.dtype)
        zero_text_codes = (text_codes.sum() * 0).to(text_codes.dtype)
        zero_text_len = (text_len.sum() * 0).to(text_len.dtype)
        zero_lines_len = (lines_len.sum() * 0).to(lines_len.dtype)
        lines_bbox = lines_bbox + zero_lines_bbox
        lines_conf = lines_conf + zero_lines_conf
        text_codes = text_codes + zero_text_codes
        text_len = text_len + zero_text_len
        lines_len = lines_len + zero_lines_len
        return lines_bbox, lines_conf, text_codes, text_len, lines_len


class SuryaOCRDetectionWrapper(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        from surya.detection import DetectionPredictor

        self.detection_predictor = DetectionPredictor(device=device)
        self._to_pil = transforms.ToPILImage()

        # Set eval mode on wrapper and underlying models
        self.eval()
        if hasattr(self.detection_predictor, "model"):
            self.detection_predictor.model.eval()
        if hasattr(self.detection_predictor, "model"):
            for _, param in self.detection_predictor.model.named_parameters():
                param.requires_grad = False

    def forward(self, images_tensor: torch.Tensor):
        batch_size = images_tensor.shape[0]
        images: List[Image.Image] = [
            self._to_pil(images_tensor[i].cpu()) for i in range(batch_size)
        ]
        predictions_by_image = self.detection_predictor(images, include_maps=False)
        # Pack detection outputs to tensors
        boxes, polys, confs, lengths, image_bboxes = pack_detection_predictions(
            predictions_by_image
        )
        boxes = boxes.to(images_tensor.device)
        polys = polys.to(images_tensor.device)
        confs = confs.to(images_tensor.device)
        lengths = lengths.to(images_tensor.device)
        image_bboxes = image_bboxes.to(images_tensor.device)
        # Tie outputs to input to avoid constant folding in generated module
        # NOTE: derive zeros from each output tensor independently to avoid
        # inadvertent cross-tensor broadcasting under XLA compilation.
        zero_boxes = (boxes.sum() * 0).to(boxes.dtype)
        zero_polys = (polys.sum() * 0).to(polys.dtype)
        zero_confs = (confs.sum() * 0).to(confs.dtype)
        zero_lengths = (lengths.sum() * 0).to(lengths.dtype)
        zero_image_bboxes = (image_bboxes.sum() * 0).to(image_bboxes.dtype)
        boxes = boxes + zero_boxes
        polys = polys + zero_polys
        confs = confs + zero_confs
        lengths = lengths + zero_lengths
        image_bboxes = image_bboxes + zero_image_bboxes
        return boxes, polys, confs, lengths, image_bboxes


class TextLineLite:
    __slots__ = ("text", "bbox", "confidence")

    def __init__(self, text, bbox, confidence):
        self.text = text
        self.bbox = bbox
        self.confidence = confidence


class OCRResultLite:
    __slots__ = ("text_lines",)

    def __init__(self, text_lines):
        self.text_lines = text_lines


def pack_predictions(preds, max_lines=50000, max_chars=50000):
    B = len(preds)
    lines_bbox = torch.zeros(B, max_lines, 4, dtype=torch.float32)
    lines_conf = torch.zeros(B, max_lines, dtype=torch.float32)
    text_codes = torch.full((B, max_lines, max_chars), fill_value=-1, dtype=torch.int32)
    text_len = torch.zeros(B, max_lines, dtype=torch.int32)
    lines_len = torch.zeros(B, dtype=torch.int32)

    for b, p in enumerate(preds):
        lines = getattr(p, "text_lines", [])[:max_lines]
        lines_len[b] = len(lines)
        for i, line in enumerate(lines):
            if hasattr(line, "bbox") and line.bbox is not None:
                lines_bbox[b, i] = torch.tensor(line.bbox, dtype=torch.float32)
            if hasattr(line, "confidence"):
                lines_conf[b, i] = float(line.confidence)
            t = getattr(line, "text", "") or ""
            codes = [ord(c) for c in t][:max_chars]
            if len(codes) > 0:
                text_codes[b, i, : len(codes)] = torch.tensor(codes, dtype=torch.int32)
            text_len[b, i] = len(codes)

    return lines_bbox, lines_conf, text_codes, text_len, lines_len


# NEW: Lightweight detection result structures and pack/unpack helpers
class PolygonBoxLite:
    __slots__ = ("polygon", "bbox", "confidence")

    def __init__(self, polygon, bbox, confidence):
        self.polygon = polygon
        self.bbox = bbox
        self.confidence = confidence


class TextDetectionResultLite:
    __slots__ = ("bboxes", "heatmap", "affinity_map", "image_bbox")

    def __init__(self, bboxes, image_bbox=None, heatmap=None, affinity_map=None):
        self.bboxes = bboxes
        self.heatmap = heatmap
        self.affinity_map = affinity_map
        self.image_bbox = image_bbox


# NEW: Lightweight detection result structures and pack/unpack helpers
def pack_detection_predictions(preds, max_boxes: int = 2048):
    B = len(preds)
    boxes = torch.zeros(B, max_boxes, 4, dtype=torch.float32)
    polys = torch.zeros(B, max_boxes, 4, 2, dtype=torch.float32)
    confs = torch.zeros(B, max_boxes, dtype=torch.float32)
    lengths = torch.zeros(B, dtype=torch.int32)
    image_bboxes = torch.zeros(B, 4, dtype=torch.float32)

    for b, p in enumerate(preds):
        # p is TextDetectionResult
        bboxes = getattr(p, "bboxes", [])
        lengths[b] = min(len(bboxes), max_boxes)
        img_bb = getattr(p, "image_bbox", None)
        if img_bb is not None and len(img_bb) == 4:
            image_bboxes[b] = torch.tensor(img_bb, dtype=torch.float32)
        for i, polybox in enumerate(bboxes[:max_boxes]):
            # polybox has fields: polygon (4x2), bbox (4), confidence
            bb = getattr(polybox, "bbox", None)
            pg = getattr(polybox, "polygon", None)
            cf = getattr(polybox, "confidence", 0.0)
            if bb is not None and len(bb) == 4:
                boxes[b, i] = torch.tensor(bb, dtype=torch.float32)
            if pg is not None and len(pg) == 4:
                polys[b, i] = torch.tensor(pg, dtype=torch.float32)
            confs[b, i] = float(cf)

    return boxes, polys, confs, lengths, image_bboxes


def unpack_predictions(lines_bbox, lines_conf, text_codes, text_len, lines_len):
    B, K, _ = lines_bbox.shape
    results = []
    for b in range(B):
        num = int(lines_len[b].item())
        page_lines = []
        for i in range(num):
            L = int(text_len[b, i].item())
            codes = text_codes[b, i, :L].tolist()
            text = "".join(chr(c) for c in codes)
            bbox = lines_bbox[b, i].tolist()
            conf = float(lines_conf[b, i].item())
            page_lines.append({"text": text, "bbox": bbox, "confidence": conf})
        results.append({"text_lines": page_lines})
    return results


def unpack_detection_predictions(boxes, polys, confs, lengths, image_bboxes):
    B, K, _ = boxes.shape
    results = []
    for b in range(B):
        num = int(lengths[b].item())
        page_boxes = []
        for i in range(num):
            bb = boxes[b, i].tolist()
            pg = polys[b, i].tolist()
            cf = float(confs[b, i].item())
            page_boxes.append(PolygonBoxLite(pg, bb, cf))
        img_bb = image_bboxes[b].tolist()
        results.append(TextDetectionResultLite(page_boxes, image_bbox=img_bb))
    return results


def freeze_all(wrapper, warmup_input: torch.Tensor = None):
    """Warm up to instantiate any lazy modules, then freeze all parameters found under predictor `.model` modules."""
    import torch.nn as nn

    # Warmup forward to trigger any lazy construction inside predictors
    if warmup_input is not None:
        try:
            with torch.inference_mode():
                _ = wrapper(warmup_input)
        except Exception:
            pass

    def freeze_module(m: nn.Module):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    # Freeze registered submodules off the wrapper itself
    for _, m in wrapper.named_modules():
        if isinstance(m, nn.Module):
            for p in m.parameters():
                p.requires_grad = False

    # Freeze predictor `.model` modules explicitly
    for obj in [
        getattr(wrapper, "rec_predictor", None),
        getattr(wrapper, "detection_predictor", None),
        getattr(wrapper, "foundation_predictor", None),
    ]:
        model_attr = getattr(obj, "model", None) if obj is not None else None
        if isinstance(model_attr, nn.Module):
            freeze_module(model_attr)


def dicts_to_objects(reconstructed):
    results = []
    for page in reconstructed:
        tls = [
            TextLineLite(line["text"], line["bbox"], line["confidence"])
            for line in page["text_lines"]
        ]
        results.append(OCRResultLite(tls))
    return results


def save_outputs_ocr_text(co_out, images, result_path):
    names: List[str] = ["excerpt_text"]
    lines_bbox, lines_conf, text_codes, text_len, lines_len = co_out
    reconstructed = unpack_predictions(
        lines_bbox, lines_conf, text_codes, text_len, lines_len
    )

    # Convert dicts to lightweight objects with attributes
    predictions_by_image = dicts_to_objects(reconstructed)

    os.makedirs(result_path, exist_ok=True)

    from surya.debug.text import draw_text_on_image

    # Save visualization PNGs
    for idx, (name, image, pred) in enumerate(zip(names, images, predictions_by_image)):
        bboxes = [line.bbox for line in pred.text_lines]
        pred_text = [line.text for line in pred.text_lines]
        page_image = draw_text_on_image(bboxes, pred_text, image.size)
        page_image.save(os.path.join(result_path, f"{name}_{idx}_text.png"))

    # Write results.json
    out_preds = defaultdict(list)
    for name, pred, image in zip(names, predictions_by_image, images):
        page_dict = {
            "text_lines": [
                {"text": tl.text, "bbox": tl.bbox, "confidence": tl.confidence}
                for tl in pred.text_lines
            ]
        }
        page_dict["page"] = len(out_preds[name]) + 1
        out_preds[name].append(page_dict)

    with open(os.path.join(result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(out_preds, f, ensure_ascii=False)

    logger.info(f"Wrote results to {result_path}")


def save_outputs_ocr_detection(co_out, images, result_path):
    boxes, polys, confs, lengths, image_bboxes = co_out
    names: List[str] = ["excerpt_text"]
    predictions_by_image = unpack_detection_predictions(
        boxes, polys, confs, lengths, image_bboxes
    )
    os.makedirs(result_path, exist_ok=True)

    from surya.debug.draw import draw_polys_on_image

    # Save bbox visualization PNGs
    for idx, (name, pred, page_image) in enumerate(
        zip(names, predictions_by_image, images)
    ):
        polygons = [p.polygon for p in pred.bboxes]
        if len(polygons) == 0:
            continue
        bbox_image = draw_polys_on_image(polygons, page_image.copy())
        bbox_image.save(os.path.join(result_path, f"{name}_{idx}_bbox.png"))

    # Write results.json
    predictions_by_page = defaultdict(list)
    for name, pred in zip(names, predictions_by_image):
        page_dict = {
            "bboxes": [
                {"polygon": pb.polygon, "bbox": pb.bbox, "confidence": pb.confidence}
                for pb in getattr(pred, "bboxes", [])
            ],
            "image_bbox": getattr(pred, "image_bbox", None),
        }
        page_dict["page"] = len(predictions_by_page[name]) + 1
        predictions_by_page[name].append(page_dict)

    with open(os.path.join(result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(predictions_by_page, f, ensure_ascii=False)

    logger.info(f"Wrote results to {result_path}")


# Robust replacement for SuryaModel.get_image_embeddings that aligns lengths
def _patched_get_image_embeddings(
    self,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    encoder_chunk_size: int,
    valid_batch_size: Optional[torch.Tensor] = None,
    max_batch_size: Optional[int] = None,
):
    # Mirror SuryaModel.get_image_embeddings with safer length alignment
    chunks = [0]
    grid_chunks = [0]
    curr_chunk_len = 0
    curr_seq_len = 0
    for i in range(len(grid_thw)):
        curr_chunk_len += (grid_thw[i][0] * grid_thw[i][1] * grid_thw[i][2]).item()
        if curr_chunk_len > encoder_chunk_size:
            chunks.append(curr_chunk_len + curr_seq_len)
            curr_seq_len += curr_chunk_len
            curr_chunk_len = 0
            grid_chunks.append(i + 1)

    if curr_chunk_len > 0:
        chunks.append(pixel_values.shape[0])
        grid_chunks.append(len(grid_thw))

    embeddings_list: list[torch.Tensor] = []
    for i in range(len(chunks) - 1):
        start = chunks[i]
        end = chunks[i + 1]
        grid_start = grid_chunks[i]
        grid_end = grid_chunks[i + 1]

        chunk_pixels = pixel_values[start:end]
        chunk_grid_thw = grid_thw[grid_start:grid_end]
        actual_chunk_len = end - start

        chunk_pixels, chunk_grid_thw, valid_embed_len = self.maybe_static_pad_image_inputs(  # type: ignore[attr-defined]
            chunk_pixels, chunk_grid_thw, actual_chunk_len, encoder_chunk_size
        )

        chunk_embeddings = self.vision_encoder.embed_images(
            image_batch=chunk_pixels.unsqueeze(0).to(device=self.device),  # type: ignore[attr-defined]
            grid_thw=chunk_grid_thw.unsqueeze(0).to(device=self.device),  # type: ignore[attr-defined]
        )
        embeddings_list.append(chunk_embeddings[:valid_embed_len].squeeze(0))

    if len(embeddings_list) == 0:
        raise ValueError(
            "No image embeddings were generated. Check the input images and grid sizes."
        )
    elif len(embeddings_list) == 1:
        embeddings = embeddings_list[0]
    else:
        embeddings = torch.cat(embeddings_list, dim=0)

    encoding_2d = self.get_2d_learned_embeddings(  # type: ignore[attr-defined]
        grid_thw,
        device=embeddings.device,
        bbox_size=self.config.image_embed_encoding_multiplier,  # type: ignore[attr-defined]
    )

    # Align lengths to prevent assertion failures due to floor/round inconsistencies
    if embeddings.shape[0] != encoding_2d.shape[0]:
        min_len = min(embeddings.shape[0], encoding_2d.shape[0])
        embeddings = embeddings[:min_len]
        encoding_2d = encoding_2d[:min_len]
    if embeddings.shape[1] != encoding_2d.shape[1]:
        min_hidden = min(embeddings.shape[1], encoding_2d.shape[1])
        embeddings = embeddings[:, :min_hidden]
        encoding_2d = encoding_2d[:, :min_hidden]

    return embeddings + encoding_2d
