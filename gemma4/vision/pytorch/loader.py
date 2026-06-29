# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma4 vision-tower loader implementation.

google/gemma-4-26B-A4B-it is a Gemma4ForConditionalGeneration (any-to-any:
text + vision + audio). This loader brings up the *vision tower* component in
isolation (``Gemma4VisionModel``), the image encoder that turns image patches
into soft tokens for the text decoder. The text decoder is brought up
separately (see ``gemma4/pytorch``), mirroring how a diffusion pipeline brings
up each component on its own.

The Gemma4 vision tower is a NaFlex-style (variable-resolution) ViT: patches
are pre-unfolded to ``3 * patch_size**2`` vectors and embedded with a plain
``nn.Linear`` (no Conv2d), with per-patch (x, y) ``pixel_position_ids`` and a
data-dependent boolean-mask pooler. Inputs are produced by the real
``Gemma4Processor`` so shapes and position ids match the checkpoint exactly.
"""

import glob
import os
from typing import Optional

import torch
from transformers import AutoConfig, AutoProcessor
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ....tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available Gemma4 vision-tower variants."""

    GEMMA_4_26B_A4B_IT = "26B-A4B-it"


class ModelLoader(ForgeModel):
    """Gemma4 vision-tower (image encoder) loader."""

    _VARIANTS = {
        ModelVariant.GEMMA_4_26B_A4B_IT: LLMModelConfig(
            pretrained_model_name="google/gemma-4-26B-A4B-it",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_4_26B_A4B_IT

    # Vision-tower weights live under this prefix in the unified checkpoint.
    _VISION_PREFIX = "model.vision_tower."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Gemma 4 vision tower",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)
        return self.processor

    def _load_vision_state_dict(self, pretrained_model_name):
        """Read only the ``model.vision_tower.*`` tensors from the sharded
        safetensors checkpoint and strip the prefix so they line up with a
        standalone ``Gemma4VisionModel`` state dict — avoids materializing the
        full 26B unified model just to exercise the ~0.4B vision tower."""
        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        snap = snapshot_download(
            pretrained_model_name, allow_patterns=["*.safetensors", "*.json"]
        )
        shards = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
        state_dict = {}
        for shard in shards:
            with safe_open(shard, framework="pt") as f:
                for key in f.keys():
                    if key.startswith(self._VISION_PREFIX):
                        new_key = key[len(self._VISION_PREFIX) :]
                        state_dict[new_key] = f.get_tensor(key)
        return state_dict

    def load_model(self, dtype_override=None):
        """Build a standalone Gemma4 vision tower and load its weights.

        Args:
            dtype_override: Optional torch dtype for the weights (the checkpoint
                ships in bfloat16).

        Returns:
            torch.nn.Module: the ``Gemma4VisionModel`` image encoder.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        config = AutoConfig.from_pretrained(pretrained_model_name)
        self.config = config
        vision_config = config.vision_config

        model = Gemma4VisionModel(vision_config)
        state_dict = self._load_vision_state_dict(pretrained_model_name)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # Buffers like std_bias/std_scale may be registered but absent from the
        # checkpoint; surface anything unexpected for debugging but don't fail.
        if unexpected:
            print(f"[gemma4 vision] unexpected keys: {unexpected[:5]} ...")
        model = model.eval()
        if dtype_override is not None:
            model = model.to(dtype_override)
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Produce vision-tower inputs via the real Gemma4Processor.

        Returns a dict ``{"pixel_values", "pixel_position_ids"}`` — the two
        positional inputs of ``Gemma4VisionModel.forward`` — extracted from the
        processor output for a single sample image.
        """
        if self.processor is None:
            self._load_processor()

        image_file = get_file(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
        )
        from PIL import Image

        image = Image.open(image_file).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        proc = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Gemma4Processor emits patch-unfolded pixel_values [B, num_patches,
        # 3*patch_size**2] and per-patch (x, y) coords under "image_position_ids"
        # — the latter is the model's "pixel_position_ids" forward argument.
        pixel_values = proc["pixel_values"]
        pixel_position_ids = proc["image_position_ids"]

        if batch_size > 1:
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)
            pixel_position_ids = pixel_position_ids.repeat_interleave(
                batch_size, dim=0
            )

        if dtype_override is not None:
            pixel_values = cast_input_to_type(pixel_values, dtype_override)

        return {
            "pixel_values": pixel_values,
            "pixel_position_ids": pixel_position_ids,
        }
