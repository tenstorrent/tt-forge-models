# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
YOLOX model loader implementation
"""

import torch
import os
from typing import Optional

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from ...tools.utils import get_file
from .src.utils import _forward_patch, _decode_outputs


class ModelVariant(StrEnum):
    """Available YOLOX model variants."""

    YOLOX_NANO = "Nano"
    YOLOX_TINY = "Tiny"
    YOLOX_S = "S"
    YOLOX_M = "M"
    YOLOX_L = "L"
    YOLOX_DARKNET = "Darknet"
    YOLOX_X = "X"


class ModelLoader(ForgeModel):
    """YOLOX model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.YOLOX_NANO: ModelConfig(
            pretrained_model_name="yolox_nano",
        ),
        ModelVariant.YOLOX_TINY: ModelConfig(
            pretrained_model_name="yolox_tiny",
        ),
        ModelVariant.YOLOX_S: ModelConfig(
            pretrained_model_name="yolox_s",
        ),
        ModelVariant.YOLOX_M: ModelConfig(
            pretrained_model_name="yolox_m",
        ),
        ModelVariant.YOLOX_L: ModelConfig(
            pretrained_model_name="yolox_l",
        ),
        ModelVariant.YOLOX_DARKNET: ModelConfig(
            pretrained_model_name="yolox_darknet",
        ),
        ModelVariant.YOLOX_X: ModelConfig(
            pretrained_model_name="yolox_x",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.YOLOX_NANO

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="YOLOX",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the YOLOX model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The YOLOX model instance.
        """
        from yolox.exp import get_exp  # Defer heavy import
        from yolox.models.yolo_head import YOLOXHead

        YOLOXHead.forward = _forward_patch
        YOLOXHead.decode_outputs = _decode_outputs

        # Get the model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        weight_url = f"https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{model_name}.pth"

        # Use the utility to download/cache the model weights
        weight_path = get_file(weight_url)

        # Handle special case for darknet variant
        if model_name == "yolox_darknet":
            exp_name = "yolov3"
        else:
            exp_name = model_name.replace("_", "-")

        # Load model
        exp = get_exp(exp_name=exp_name)
        model = exp.get_model()
        ckpt = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def input_preprocess(self, dtype_override=None, batch_size=1, image=None):
        """Preprocess an input image and return a model-ready tensor.

        Args:
            dtype_override: Optional torch.dtype override (default: float32).
            batch_size: Batch size (default: 1).
            image: PIL Image or None. If None, loads a default image from HuggingFace.

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        import numpy as np
        from yolox.data.data_augment import preproc as preprocess

        model_name = self._variant_config.pretrained_model_name
        input_shape = (
            (416, 416) if model_name in ["yolox_nano", "yolox_tiny"] else (640, 640)
        )

        if image is None:
            from datasets import load_dataset

            ds = load_dataset("mpnikhil/kitchen-classifier", split="train").with_format(
                "np"
            )
            img = ds[200]["image"]
        else:
            img = np.array(image)

        img_tensor, ratio = preprocess(img, input_shape)
        self.ratio = ratio
        self.input_shape = input_shape

        batch_tensor = (
            torch.from_numpy(img_tensor)
            .unsqueeze(0)
            .repeat_interleave(batch_size, dim=0)
        )

        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor

    def output_postprocess(self, output, top_k=None):
        """Post-process raw model output into structured detection results.

        Returns a dict with keys "labels", "probabilities", and "indices" (sorted by
        descending confidence), compatible with ForgeRunner's _postprocess_model_output.

        Args:
            output: Model output tensor [batch, n_anchors, 85] or list/tuple thereof.
            top_k: Maximum number of detections to return. None returns all.

        Returns:
            dict: {"labels": [...], "probabilities": [...], "indices": [...]}
        """
        from .src.utils import get_detections

        if isinstance(output, (list, tuple)):
            out_np = output[0].cpu().detach().float().numpy()
        else:
            out_np = output.cpu().detach().float().numpy()

        detections = get_detections(out_np, self.ratio, self.input_shape)
        if top_k is not None:
            detections = detections[:top_k]

        return {
            "labels": [d["class_name"] for d in detections],
            "probabilities": [f"{d['score'] * 100:.4f}%" for d in detections],
            "indices": [d["cls_ind"] for d in detections],
        }

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the YOLOX model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: ``{"x": image_tensor, "targets": targets_tensor}`` where ``targets`` has
            shape ``[B, max_labels, 5]`` with each row encoding ``[class_id, cx, cy, w, h]``.
            ``targets`` is ignored by the model in eval mode and required in train mode.
        """
        return self.input_preprocess(
            dtype_override=dtype_override, batch_size=batch_size
        )

    def unpack_forward_output(self, forward_output):
        """Extract the loss-relevant tensor from YOLOX's training-mode output.

        Forward output structure: bare ``dict`` with keys ``total_loss``,
        ``iou_loss``, ``l1_loss``, ``conf_loss``, ``cls_loss``, ``num_fg``.
        ``total_loss`` is a scalar tensor with ``requires_grad=True``; the four
        ``*_loss`` components are summed into it; ``num_fg`` is a Python float
        count (not a tensor).

        Selection: ``total_loss``. It is the single scalar that drives the
        backward pass — gradients flow through all four loss components and back
        to every model parameter. The component losses are redundant with it,
        and ``num_fg`` is non-differentiable.

        Why not the registry: the forward output is a bare ``dict`` (no class
        name to key on), so ``_register_attr`` cannot dispatch on it.
        """
        return forward_output["total_loss"]

    def post_processing(self, co_out):
        """Post-process the model outputs.

        Args:
            co_out: Compiled model outputs

        Returns:
            None: Prints the detection results
        """
        from .src.utils import get_detections

        for i in range(len(co_out)):
            co_out[i] = co_out[i].detach().float().numpy()

        for det in get_detections(co_out[0], self.ratio, self.input_shape):
            x_min, y_min, x_max, y_max = det["bbox"]
            print(
                f"Class: {det['class_name']}, Confidence: {det['score']}, "
                f"Coordinates: ({x_min}, {y_min}, {x_max}, {y_max})"
            )
