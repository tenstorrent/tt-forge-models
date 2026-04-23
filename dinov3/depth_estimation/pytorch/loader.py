# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DINOv3 ViT CHMv2 DPT-head model loader implementation for canopy height
depth estimation (PyTorch).
"""
import json
import os

import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from datasets import load_dataset
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

# Public ONNX community mirror (no auth required) for config/processor
_ONNX_COMMUNITY_REPO = "onnx-community/dinov3-vitl16-chmv2-dpt-head-ONNX"
# Public DINOv3 ViT-L/16 backbone mirror (no auth required)
_PUBLIC_BACKBONE_REPO = "xycheni/facebook-dinov3-vitl16-pretrain-lvd1689m"
_PUBLIC_BACKBONE_PTH = "dinov3-vitl16-pretrain-lvd1689m.pth"


class ModelVariant(StrEnum):
    """Available DINOv3 ViT CHMv2 DPT-head model variants."""

    VITL16_CHMV2_DPT_HEAD = "ViTL16_CHMv2_DPT_Head"


class ModelLoader(ForgeModel):
    """DINOv3 ViT CHMv2 DPT-head model loader for depth estimation."""

    _VARIANTS = {
        ModelVariant.VITL16_CHMV2_DPT_HEAD: ModelConfig(
            pretrained_model_name="facebook/dinov3-vitl16-chmv2-dpt-head",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VITL16_CHMV2_DPT_HEAD

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DINOv3 ViT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from huggingface_hub.errors import GatedRepoError

        pretrained_model_name = self._variant_config.pretrained_model_name
        token = os.environ.get("HF_TOKEN")

        if token:
            try:
                self.processor = AutoImageProcessor.from_pretrained(
                    pretrained_model_name, token=token
                )
                return self.processor
            except (GatedRepoError, OSError):
                pass

        # Fall back to public ONNX community mirror for the processor config
        self.processor = AutoImageProcessor.from_pretrained(_ONNX_COMMUNITY_REPO)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the model.

        When HF_TOKEN is set (and grants access to facebook/dinov3-vitl16-chmv2-dpt-head),
        the fully pretrained model is loaded. Otherwise a model with the same
        architecture is constructed from publicly available sources:
          - Config/processor: onnx-community/dinov3-vitl16-chmv2-dpt-head-ONNX
          - Backbone weights: xycheni/facebook-dinov3-vitl16-pretrain-lvd1689m
          - DPT head: randomly initialised (sufficient for compile-only testing)
        """
        from huggingface_hub.errors import GatedRepoError

        pretrained_model_name = self._variant_config.pretrained_model_name
        token = kwargs.pop("token", None) or os.environ.get("HF_TOKEN")

        if token:
            model_kwargs = {"token": token}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs
            try:
                model = AutoModelForDepthEstimation.from_pretrained(
                    pretrained_model_name, **model_kwargs
                )
            except (GatedRepoError, OSError):
                model = self._load_model_public(dtype_override=dtype_override)
        else:
            model = self._load_model_public(dtype_override=dtype_override)

        model.eval()
        return model

    def _load_model_public(self, *, dtype_override=None):
        """Build CHMv2ForDepthEstimation from publicly available sources."""
        from transformers import CHMv2Config, CHMv2ForDepthEstimation
        from huggingface_hub import hf_hub_download

        # Load architecture config from public ONNX community mirror
        config_path = hf_hub_download(_ONNX_COMMUNITY_REPO, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)

        skip_keys = {
            "architectures",
            "transformers_version",
            "transformers.js_config",
            "dtype",
        }
        config = CHMv2Config(
            **{k: v for k, v in config_dict.items() if k not in skip_keys}
        )

        # Instantiate model (DPT head has random weights; backbone loaded below)
        model = CHMv2ForDepthEstimation(config)

        # Load public backbone weights and inject into model
        pth_path = hf_hub_download(_PUBLIC_BACKBONE_REPO, _PUBLIC_BACKBONE_PTH)
        backbone_sd = torch.load(pth_path, map_location="cpu", weights_only=True)
        # Keys in the .pth: "embeddings.cls_token", "layer.0.attention.k_proj.weight", …
        # The CHMv2 model stores the backbone at backbone.model.<key>
        prefixed_sd = {f"backbone.model.{k}": v for k, v in backbone_sd.items()}
        missing, unexpected = model.load_state_dict(prefixed_sd, strict=False)
        _ = missing, unexpected  # non-backbone keys intentionally missing

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        inputs = self.processor(images=image, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype.is_floating_point:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
