# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit-InScene LoRA model loader implementation.

Loads the Qwen-Image-Edit diffusion transformer and applies the
peteromallet/Qwen-Image-Edit-InScene LoRA adapter for generating new
shots within the same scene while maintaining scene structure.

Available variants:
- QWEN_IMAGE_EDIT_INSCENE: InScene LoRA (InScene-0.7.safetensors)
- QWEN_IMAGE_EDIT_INSCENE_ANNOTATE: InScene Annotate LoRA
  (InScene-Annotate-0.4.safetensors) for region-focused editing via
  green rectangle annotations.
"""

import re
from typing import Any, Optional

import torch
from diffusers import QwenImageEditPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safetensors_load_file

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

BASE_MODEL = "Qwen/Qwen-Image-Edit"
LORA_REPO = "peteromallet/Qwen-Image-Edit-InScene"

LORA_WEIGHT_INSCENE = "InScene-0.7.safetensors"
LORA_WEIGHT_INSCENE_ANNOTATE = "InScene-Annotate-0.4.safetensors"

# Maps old blocks.X.{attn,ffn} naming to diffusers transformer_blocks naming.
_OLD_MODULE_TO_NEW = {
    "self_attn.q": "attn.to_q",
    "self_attn.k": "attn.to_k",
    "self_attn.v": "attn.to_v",
    "self_attn.o": "attn.to_out.0",
    "cross_attn.q": "attn.add_q_proj",
    "cross_attn.k": "attn.add_k_proj",
    "cross_attn.v": "attn.add_v_proj",
    "cross_attn.o": "attn.to_add_out",
    "ffn.0": "img_mlp.net.0.proj",
    "ffn.2": "img_mlp.net.2",
}

_OLD_KEY_RE = re.compile(
    r"^(diffusion_model)\.blocks\.(\d+)\.(.+?)\.(lora_[AB]\.weight)$"
)


def _remap_lora_state_dict(state_dict: dict) -> dict:
    """Remap LoRA keys from old blocks.X format to diffusers transformer_blocks format."""
    remapped = {}
    for key, value in state_dict.items():
        m = _OLD_KEY_RE.match(key)
        if m:
            prefix, block_num, module_part, suffix = m.groups()
            if module_part in _OLD_MODULE_TO_NEW:
                new_module = _OLD_MODULE_TO_NEW[module_part]
                key = f"{prefix}.transformer_blocks.{block_num}.{new_module}.{suffix}"
        remapped[key] = value
    return remapped


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit-InScene model variants."""

    QWEN_IMAGE_EDIT_INSCENE = "Edit_InScene"
    QWEN_IMAGE_EDIT_INSCENE_ANNOTATE = "Edit_InScene_Annotate"


_LORA_FILES = {
    ModelVariant.QWEN_IMAGE_EDIT_INSCENE: LORA_WEIGHT_INSCENE,
    ModelVariant.QWEN_IMAGE_EDIT_INSCENE_ANNOTATE: LORA_WEIGHT_INSCENE_ANNOTATE,
}


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit-InScene LoRA model loader for same-scene shot generation."""

    _VARIANTS = {
        ModelVariant.QWEN_IMAGE_EDIT_INSCENE: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.QWEN_IMAGE_EDIT_INSCENE_ANNOTATE: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QWEN_IMAGE_EDIT_INSCENE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_EDIT_INSCENE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load the Qwen-Image-Edit transformer with InScene LoRA weights applied.

        Loads the full pipeline, applies the diffusers-format LoRA, fuses the
        weights into the transformer, and returns the transformer component.

        Returns:
            QwenImageTransformer2DModel with LoRA weights fused.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        pipe = QwenImageEditPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
        )

        lora_file = _LORA_FILES[self._variant]
        if self._variant == ModelVariant.QWEN_IMAGE_EDIT_INSCENE_ANNOTATE:
            # InScene-Annotate-0.4 uses old blocks.X key format; remap to
            # the current diffusers transformer_blocks naming before loading.
            lora_path = hf_hub_download(LORA_REPO, filename=lora_file)
            state_dict = safetensors_load_file(lora_path)
            state_dict = _remap_lora_state_dict(state_dict)
            pipe.load_lora_weights(state_dict)
        else:
            pipe.load_lora_weights(LORA_REPO, weight_name=lora_file)

        pipe.fuse_lora()

        self._transformer = pipe.transformer
        self._transformer.eval()
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the diffusion transformer.

        Returns a dict matching QwenImageTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        batch_size = kwargs.get("batch_size", 1)

        # From model config: in_channels=64 (img_in linear input dimension)
        img_dim = 64
        # joint_attention_dim from config = 3584
        text_dim = 3584
        txt_seq_len = 32

        # img_seq_len must equal frame * height * width for positional encoding
        frame, height, width = 1, 8, 8
        img_seq_len = frame * height * width

        hidden_states = torch.randn(batch_size, img_seq_len, img_dim, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, text_dim, dtype=dtype
        )
        encoder_hidden_states_mask = torch.ones(batch_size, txt_seq_len, dtype=dtype)
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)
        img_shapes = [(frame, height, width)] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }
