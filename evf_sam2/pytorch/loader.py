# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EVF-SAM2 (Early Vision-Language Fusion SAM2) loader implementation
for text-prompted image segmentation.

Requires the EVF-SAM repository to be cloned at /tmp/evf_sam_repo.
"""

import os
import sys

import torch
from typing import Optional
from PIL import Image
from loguru import logger
from transformers import AutoTokenizer
from datasets import load_dataset

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

EVF_SAM_REPO_PATH = "/tmp/evf_sam_repo"


class EvfSam2InferenceWrapper(torch.nn.Module):
    """Wraps EvfSam2Model to expose the inference path as forward."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, images_evf, input_ids):
        return self.model.inference(
            images,
            images_evf,
            input_ids,
            resize_list=[(1024, 1024)],
            original_size_list=[(1024, 1024)],
        )


_evf_repo_ready = False
_evf_saved_modules = {}


def _setup_evf_repo():
    """Clone the EVF-SAM repo and create missing __init__.py files."""
    global _evf_repo_ready
    if _evf_repo_ready:
        return

    if not os.path.isdir(EVF_SAM_REPO_PATH):
        import subprocess

        subprocess.check_call(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "https://github.com/hustvl/EVF-SAM.git",
                EVF_SAM_REPO_PATH,
            ]
        )

    for subdir in [
        "model",
        "model/unilm",
        "model/unilm/beit3",
        "model/segment_anything_2",
    ]:
        init_file = os.path.join(EVF_SAM_REPO_PATH, subdir, "__init__.py")
        if not os.path.exists(init_file):
            open(init_file, "w").close()

    if EVF_SAM_REPO_PATH not in sys.path:
        sys.path.insert(0, EVF_SAM_REPO_PATH)

    _evf_repo_ready = True


def _swap_model_modules():
    """Temporarily evict conflicting 'model' modules from sys.modules so
    that the EVF-SAM ``model`` package can be imported / used."""
    saved = {}
    for key in list(sys.modules):
        if key == "model" or key.startswith("model."):
            saved[key] = sys.modules.pop(key)
    _evf_saved_modules.update(saved)


def _restore_model_modules():
    """Move EVF-SAM ``model.*`` entries aside and restore the originals."""
    for key in list(sys.modules):
        if key == "model" or key.startswith("model."):
            sys.modules["_evf_" + key] = sys.modules.pop(key)
    sys.modules.update(_evf_saved_modules)
    _evf_saved_modules.clear()


class ModelVariant(StrEnum):
    """Available EVF-SAM2 model variants."""

    MULTITASK = "Multitask"


class ModelLoader(ForgeModel):
    """EVF-SAM2 model loader implementation for text-prompted image segmentation."""

    _VARIANTS = {
        ModelVariant.MULTITASK: ModelConfig(
            pretrained_model_name="YxZhang/evf-sam2-multitask",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MULTITASK

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="EVF-SAM2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        _setup_evf_repo()
        _swap_model_modules()
        try:
            from model.evf_sam2 import EvfSam2Model

            model_kwargs = {}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs

            model = EvfSam2Model.from_pretrained(
                pretrained_model_name, low_cpu_mem_usage=True, **model_kwargs
            )
            model.eval()
        finally:
            _restore_model_modules()

        return EvfSam2InferenceWrapper(model)

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        try:
            dataset = load_dataset("huggingface/cats-image")["test"]
            raw_image = dataset[0]["image"].convert("RGB")
        except Exception as e:
            logger.warning(
                f"Failed to load image from dataset. Using random fallback tensor. Reason: {e}"
            )
            raw_image = Image.fromarray(
                (torch.rand(3, 224, 224) * 255).byte().permute(1, 2, 0).numpy()
            )

        input_ids = self.tokenizer("a cat", return_tensors="pt", padding=True)[
            "input_ids"
        ]

        image_beit = raw_image.resize((224, 224))
        image_beit = torch.tensor(
            list(image_beit.getdata()), dtype=torch.float32
        ).reshape(1, 224, 224, 3)
        image_beit = image_beit.permute(0, 3, 1, 2) / 127.5 - 1.0

        image_sam = raw_image.resize((1024, 1024))
        image_sam = torch.tensor(
            list(image_sam.getdata()), dtype=torch.float32
        ).reshape(1, 1024, 1024, 3)
        image_sam = image_sam.permute(0, 3, 1, 2) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        image_sam = (image_sam - mean) / std

        if dtype_override is not None:
            image_beit = image_beit.to(dtype_override)
            image_sam = image_sam.to(dtype_override)

        if batch_size > 1:
            input_ids = input_ids.repeat_interleave(batch_size, dim=0)
            image_beit = image_beit.repeat_interleave(batch_size, dim=0)
            image_sam = image_sam.repeat_interleave(batch_size, dim=0)

        return {
            "images": image_sam,
            "images_evf": image_beit,
            "input_ids": input_ids,
        }
