# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RetinaNet model loader implementation
"""

import os
import shutil
import zipfile
from collections import OrderedDict
from typing import Optional

import requests
import torch
from torch import Tensor
from PIL import Image
from torchvision import transforms, models
from dataclasses import dataclass
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from ...base import ForgeModel
from .src.model import Model
from datasets import load_dataset


def patched_retinanet_forward(
    self,
    images: list[Tensor],
    targets: Optional[list[dict[str, Tensor]]] = None,
) -> tuple[dict[str, Tensor], list[Tensor]]:
    """Run transform + backbone + head + anchor generation, skip postprocess.

    RetinaNet's postprocess_detections uses masked_select and batched_nms
    which cause L1 memory overflow on TT device.
    See: https://github.com/tenstorrent/tt-xla/issues/3389
    """
    if self.training:
        if targets is None:
            torch._assert(False, "targets should not be none when in training mode")
        else:
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                    )
                else:
                    torch._assert(
                        False,
                        f"Expected target boxes to be of type Tensor, got {type(boxes)}.",
                    )

    original_image_sizes: list[tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        torch._assert(
            len(val) == 2,
            f"expecting the last two dimensions of the Tensor to be H and W "
            f"instead got {img.shape[-2:]}",
        )
        original_image_sizes.append((val[0], val[1]))

    images, targets = self.transform(images, targets)

    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: list[float] = boxes[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}.",
                )

    features = self.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    features = list(features.values())

    head_outputs = self.head(features)

    anchors = self.anchor_generator(images, features)

    return (head_outputs, anchors)


@dataclass
class RetinaNetConfig(ModelConfig):
    """Configuration specific to RetinaNet models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available RetinaNet model variants."""

    # NVIDIA variants (custom model implementation)
    RETINANET_RN18FPN = "ResNet18_Backbone_with_FPN"
    RETINANET_RN34FPN = "ResNet34_Backbone_with_FPN"
    RETINANET_RN50FPN = "ResNet50_Backbone_with_FPN"
    RETINANET_RN101FPN = "ResNet101_Backbone_with_FPN"
    RETINANET_RN152FPN = "ResNet152_Backbone_with_FPN"

    # Torchvision variants
    RETINANET_RESNET50_FPN_V2 = "ResNet50_Backbone_with_FPN_V2"


class ModelLoader(ForgeModel):
    """RetinaNet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # Github variants
        ModelVariant.RETINANET_RN18FPN: RetinaNetConfig(
            pretrained_model_name="retinanet_rn18fpn",
            source=ModelSource.CUSTOM,
        ),
        ModelVariant.RETINANET_RN34FPN: RetinaNetConfig(
            pretrained_model_name="retinanet_rn34fpn",
            source=ModelSource.CUSTOM,
        ),
        ModelVariant.RETINANET_RN50FPN: RetinaNetConfig(
            pretrained_model_name="retinanet_rn50fpn",
            source=ModelSource.CUSTOM,
        ),
        ModelVariant.RETINANET_RN101FPN: RetinaNetConfig(
            pretrained_model_name="retinanet_rn101fpn",
            source=ModelSource.CUSTOM,
        ),
        ModelVariant.RETINANET_RN152FPN: RetinaNetConfig(
            pretrained_model_name="retinanet_rn152fpn",
            source=ModelSource.CUSTOM,
        ),
        # Torchvision variants
        ModelVariant.RETINANET_RESNET50_FPN_V2: RetinaNetConfig(
            pretrained_model_name="retinanet_resnet50_fpn_v2",
            source=ModelSource.TORCHVISION,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.RETINANET_RN50FPN

    # Weight mappings for torchvision variants
    _TORCHVISION_WEIGHTS = {
        ModelVariant.RETINANET_RESNET50_FPN_V2: "RetinaNet_ResNet50_FPN_V2_Weights",
    }

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

        # Get source from variant config
        source = cls._VARIANTS[variant].source

        return ModelInfo(
            model="RetinaNet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_OBJECT_DET,
            source=source,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self._cleanup_files = []  # Track files to cleanup
        self.image_sizes = None

    def _download_nvidia_model(self, variant_name):
        """Download and extract NVIDIA RetinaNet model."""
        url = f"https://github.com/NVIDIA/retinanet-examples/releases/download/19.04/{variant_name}.zip"
        local_zip_path = f"{variant_name}.zip"

        # Download the model
        response = requests.get(url)
        with open(local_zip_path, "wb") as f:
            f.write(response.content)

        # Extract the zip file
        extracted_path = variant_name
        with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
            zip_ref.extractall(".")

        # Find the .pth file
        checkpoint_path = ""
        for root, _, files in os.walk(extracted_path):
            for file in files:
                if file.endswith(".pth"):
                    checkpoint_path = os.path.join(root, file)
                    break

        # Track files for cleanup
        self._cleanup_files.extend([local_zip_path, extracted_path])

        return checkpoint_path

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load RetinaNet with forward patched to return raw head outputs and anchors.

        Post-processing (postprocess_detections, NMS) is decoupled and available via
        postprocess_detections() to avoid L1 memory overflow on TT device.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
                           NOTE: This parameter is currently ignored for retinanet_resnet50_fpn_v2(model always uses float32).
                           TODO (@ppadjinTT): remove this when torchvision starts supporting torchvision.ops.nms for bfloat16

        Returns:
            torch.nn.Module: The RetinaNet model instance with patched forward.
        """
        # Get the pretrained model name and source from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.TORCHVISION:
            from torchvision.models.detection.retinanet import RetinaNet

            RetinaNet.forward = patched_retinanet_forward

            weight_name = self._TORCHVISION_WEIGHTS[self._variant]
            weights = getattr(models.detection, weight_name).DEFAULT
            self.model = getattr(models.detection, model_name)(weights=weights)
        elif source == ModelSource.CUSTOM:
            checkpoint_path = self._download_nvidia_model(model_name)
            self.model = Model.load(checkpoint_path)
        self.model.eval()

        if dtype_override is not None:
            if model_name == "retinanet_resnet50_fpn_v2":
                print(
                    "NOTE: dtype_override ignored - batched_nms lacks BFloat16 support"
                )
            else:
                self.model = self.model.to(dtype_override)

        return self.model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the RetinaNet model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
                           NOTE: This parameter is currently ignored for retinanet_resnet50_fpn_v2(model always uses float32).
                           TODO (@ppadjinTT): remove this when torchvision starts supporting torchvision.ops.nms for bfloat16
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for RetinaNet.
        """
        # Load image from HuggingFace datasets
        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"]
        if hasattr(image, "convert"):
            image = image.convert("RGB")

        # Get the pretrained model name and source from the instance's variant config
        source = self._variant_config.source
        model_name = self._variant_config.pretrained_model_name

        if source == ModelSource.TORCHVISION:
            weight_name = self._TORCHVISION_WEIGHTS[self._variant]
            weights = getattr(models.detection, weight_name).DEFAULT
            preprocess = weights.transforms()
            img_t = preprocess(image)
            batch_t = torch.unsqueeze(img_t, 0).contiguous()
        elif source == ModelSource.CUSTOM:
            new_size = (640, 480)
            pil_img = image.resize(new_size, resample=Image.BICUBIC)
            preprocess = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            img = preprocess(pil_img)
            batch_t = img.unsqueeze(0)

        # Replicate tensors for batch size
        batch_t = batch_t.repeat_interleave(batch_size, dim=0)

        # Store image sizes for postprocessing
        self.image_sizes = [(batch_t.shape[2], batch_t.shape[3])]

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            if model_name == "retinanet_resnet50_fpn_v2":
                print(
                    "NOTE: dtype_override ignored - batched_nms lacks BFloat16 support"
                )
            else:
                batch_t = batch_t.to(dtype_override)

        return batch_t

    def postprocess_detections(self, outputs):
        """Run post-processing (NMS, score filtering) on CPU.

        Args:
            outputs: Tuple of (head_outputs, anchors) from the patched forward.

        Returns:
            list[dict]: Detection results with boxes, scores, labels.
        """
        head_outputs, anchors = outputs
        detections = self.model.postprocess_detections(
            head_outputs, anchors, self.image_sizes
        )
        detections = self.model.transform.postprocess(
            detections, self.image_sizes, self.image_sizes
        )
        return detections

    def cleanup(self):
        """Clean up downloaded files."""
        for file_path in self._cleanup_files:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Warning: Could not clean up {file_path}: {e}")
        self._cleanup_files.clear()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
