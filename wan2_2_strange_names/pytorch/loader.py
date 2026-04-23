# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kokosha01/Wan2.2_StrangeNames model loader implementation.

LoRA adapters trained on Wan 2.2 I2V A14B (hidden_dim=5120, 40 layers).
Each safetensors file is a separate variant applied on top of the base
WAN 2.2 transformer. LoRA keys use the original non-diffusers naming
convention and are remapped to the diffusers WanTransformer3DModel paths
before applying the delta weights directly.

Repository: https://huggingface.co/Kokosha01/Wan2.2_StrangeNames
"""

import re
from typing import Any, Optional

import safetensors.torch
import torch
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from huggingface_hub import hf_hub_download

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

BASE_MODEL = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
LORA_REPO = "Kokosha01/Wan2.2_StrangeNames"

# Wan 2.2 I2V A14B config constants (from transformer/config.json)
_IN_CHANNELS = 36
_TEXT_DIM = 4096
_NUM_LAYERS = 40


class ModelVariant(StrEnum):
    """Available Kokosha01/Wan2.2_StrangeNames LoRA variants."""

    GLASS_ROOT_D2 = "GlassRoot_D2"
    NOVA_MIND_X1 = "NovaMind_X1"
    FROST_BYTE_K7 = "FrostByte_K7"
    IRON_SIGHT_V7 = "IronSight_V7"
    SOLAR_FLINT_L2 = "SolarFlint_L2"
    VELVET_RUSH_Q4 = "VelvetRush_Q4"
    PHANTOM_WEAVE_R5 = "PhantomWeave_R5"
    ECHO_VAULT_T9 = "EchoVault_T9"


_LORA_FILES = {
    ModelVariant.GLASS_ROOT_D2: "GlassRoot_D2.safetensors",
    ModelVariant.NOVA_MIND_X1: "NovaMind_X1.safetensors",
    ModelVariant.FROST_BYTE_K7: "FrostByte_K7.safetensors",
    ModelVariant.IRON_SIGHT_V7: "IronSight_V7.safetensors",
    ModelVariant.SOLAR_FLINT_L2: "SolarFlint_L2.safetensors",
    ModelVariant.VELVET_RUSH_Q4: "VelvetRush_Q4.safetensors",
    ModelVariant.PHANTOM_WEAVE_R5: "PhantomWeave_R5.safetensors",
    ModelVariant.ECHO_VAULT_T9: "EchoVault_T9.safetensors",
}


def _remap_lora_path(path: str) -> Optional[str]:
    """Convert a LoRA base path (no suffix) to a WAN transformer parameter path.

    Returns None for paths that have no corresponding parameter in the model
    (e.g. image-conditional projections absent from this variant).
    """
    path = path.removeprefix("diffusion_model.")

    # Skip image-conditional cross-attention projections (not in base model)
    if ".cross_attn.k_img" in path or ".cross_attn.v_img" in path:
        return None

    # self_attn.{q,k,v,o} -> attn1.{to_q,to_k,to_v,to_out.0}
    path = re.sub(r"\.self_attn\.q(?=\.|$)", ".attn1.to_q", path)
    path = re.sub(r"\.self_attn\.k(?=\.|$)", ".attn1.to_k", path)
    path = re.sub(r"\.self_attn\.v(?=\.|$)", ".attn1.to_v", path)
    path = re.sub(r"\.self_attn\.o(?=\.|$)", ".attn1.to_out.0", path)

    # cross_attn.{q,k,v,o} -> attn2.{to_q,to_k,to_v,to_out.0}
    path = re.sub(r"\.cross_attn\.q(?=\.|$)", ".attn2.to_q", path)
    path = re.sub(r"\.cross_attn\.k(?=\.|$)", ".attn2.to_k", path)
    path = re.sub(r"\.cross_attn\.v(?=\.|$)", ".attn2.to_v", path)
    path = re.sub(r"\.cross_attn\.o(?=\.|$)", ".attn2.to_out.0", path)

    # ffn.{0,2} -> ffn.net.{0.proj,2}
    path = re.sub(r"\.ffn\.0(?=\.|$)", ".ffn.net.0.proj", path)
    path = re.sub(r"\.ffn\.2(?=\.|$)", ".ffn.net.2", path)

    return path


def _apply_lora(transformer: WanTransformer3DModel, state_dict: dict) -> None:
    """Apply LoRA deltas directly to transformer parameter weights.

    Uses scale=1.0 (alpha=rank convention: no scaling needed).
    Keys that cannot be remapped to existing parameters are silently skipped.
    """
    model_params = dict(transformer.named_parameters())
    down_suffix = ".lora_down.weight"

    for k, down_w in state_dict.items():
        if not k.endswith(down_suffix):
            continue
        base = k[: -len(down_suffix)]
        up_w = state_dict.get(base + ".lora_up.weight")
        if up_w is None:
            continue

        remapped = _remap_lora_path(base)
        if remapped is None:
            continue

        param_key = remapped + ".weight"
        if param_key not in model_params:
            continue

        delta = (up_w @ down_w).to(model_params[param_key].dtype)
        with torch.no_grad():
            model_params[param_key].data += delta


class ModelLoader(ForgeModel):
    """Kokosha01/Wan2.2_StrangeNames LoRA model loader."""

    _VARIANTS = {
        variant: ModelConfig(pretrained_model_name=LORA_REPO)
        for variant in ModelVariant
    }

    DEFAULT_VARIANT = ModelVariant.GLASS_ROOT_D2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer: Optional[WanTransformer3DModel] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Wan2.2_StrangeNames",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(
        self, dtype: torch.dtype = torch.bfloat16
    ) -> WanTransformer3DModel:
        transformer = WanTransformer3DModel.from_pretrained(
            BASE_MODEL,
            subfolder="transformer",
            torch_dtype=dtype,
        )

        lora_file = _LORA_FILES[self._variant]
        lora_path = hf_hub_download(repo_id=LORA_REPO, filename=lora_file)
        state_dict = safetensors.torch.load_file(lora_path)
        _apply_lora(transformer, state_dict)

        self._transformer = transformer
        return transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load the WAN 2.2 I2V transformer with LoRA weights fused in-place."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._transformer is None:
            self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype_override)
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Return synthetic inputs matching the WAN 3D transformer forward signature."""
        dtype = kwargs.get("dtype_override", torch.bfloat16)

        # Patch size is [1, 2, 2] so spatial dims must be divisible by 2.
        hidden_states = torch.randn(1, _IN_CHANNELS, 1, 8, 8, dtype=dtype)
        timestep = torch.tensor([500], dtype=torch.long)
        encoder_hidden_states = torch.randn(1, 16, _TEXT_DIM, dtype=dtype)

        return [hidden_states, timestep, encoder_hidden_states]
