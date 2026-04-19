# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DDColor model loader implementation for image colorization tasks.
"""

import json

import torch
from huggingface_hub import hf_hub_download
from torchvision import transforms
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from datasets import load_dataset


class ModelVariant(StrEnum):
    """Available DDColor model variants."""

    DDCOLOR_MODELSCOPE = "modelscope"


class ModelLoader(ForgeModel):
    """DDColor model loader implementation for image colorization tasks."""

    _VARIANTS = {
        ModelVariant.DDCOLOR_MODELSCOPE: ModelConfig(
            pretrained_model_name="piddnad/ddcolor_modelscope",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DDCOLOR_MODELSCOPE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DDColor",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from .src import DDColor

        repo_id = self._variant_config.pretrained_model_name

        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path) as f:
            config = json.load(f)

        model = DDColor(
            encoder_name=config.get("encoder_name", "convnext-l"),
            decoder_name=config.get("decoder_name", "MultiScaleColorDecoder"),
            input_size=config.get("input_size", [512, 512]),
            num_output_channels=config.get("num_output_channels", 2),
            last_norm=config.get("last_norm", "Spectral"),
            do_normalize=config.get("do_normalize", False),
            num_queries=config.get("num_queries", 100),
            num_scales=config.get("num_scales", 3),
            dec_layers=config.get("dec_layers", 9),
        )

        weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].convert("RGB")

        transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )

        input_tensor = transform(image).unsqueeze(0)

        if batch_size > 1:
            input_tensor = input_tensor.repeat(batch_size, 1, 1, 1)

        if dtype_override is not None:
            input_tensor = input_tensor.to(dtype_override)

        return input_tensor
