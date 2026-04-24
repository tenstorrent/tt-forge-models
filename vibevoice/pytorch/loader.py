# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VibeVoice model loader implementation for text-to-speech tasks.
"""
import torch
import torch.nn as nn
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


class VibeVoiceQwen2Wrapper(nn.Module):
    """Wrapper around the VibeVoice Qwen2 LLM backbone.

    Exposes a clean forward pass that takes pre-computed input embeddings
    and produces hidden states from the Qwen2 decoder.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds):
        outputs = self.model(inputs_embeds=inputs_embeds, use_cache=False)
        return outputs.last_hidden_state


class ModelVariant(StrEnum):
    """Available VibeVoice model variants."""

    VIBEVOICE_7B_BNB_4BIT = "7B-bnb-4bit"
    LARGE_Q8 = "Large-Q8"


class ModelLoader(ForgeModel):
    """VibeVoice model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.VIBEVOICE_7B_BNB_4BIT: ModelConfig(
            pretrained_model_name="marksverdhai/vibevoice-7b-bnb-4bit",
        ),
        ModelVariant.LARGE_Q8: ModelConfig(
            pretrained_model_name="FabioSarracino/VibeVoice-Large-Q8",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIBEVOICE_7B_BNB_4BIT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="VibeVoice",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from huggingface_hub import hf_hub_download
        from transformers import Qwen2Config, Qwen2Model
        import json

        # The HF model uses BnB 8-bit quantization which requires CUDA/Intel+IPEX.
        # For CPU compile-only testing we load just the Qwen2 backbone with random
        # weights using the architecture described in the model's decoder_config.
        config_path = hf_hub_download(
            self._variant_config.pretrained_model_name, "config.json"
        )
        with open(config_path) as f:
            raw_config = json.load(f)

        decoder_cfg = raw_config["decoder_config"]
        qwen2_config = Qwen2Config(
            hidden_size=decoder_cfg["hidden_size"],
            intermediate_size=decoder_cfg["intermediate_size"],
            num_hidden_layers=decoder_cfg["num_hidden_layers"],
            num_attention_heads=decoder_cfg["num_attention_heads"],
            num_key_value_heads=decoder_cfg["num_key_value_heads"],
            rms_norm_eps=decoder_cfg["rms_norm_eps"],
            rope_theta=decoder_cfg["rope_theta"],
            vocab_size=decoder_cfg["vocab_size"],
            max_position_embeddings=decoder_cfg["max_position_embeddings"],
        )

        dtype = dtype_override or torch.bfloat16
        backbone = Qwen2Model(qwen2_config).to(dtype)
        model = VibeVoiceQwen2Wrapper(backbone)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.bfloat16
        # Qwen2 backbone hidden_size=3584, use a short sequence of embeddings
        inputs_embeds = torch.randn(1, 32, 3584, dtype=dtype)
        return (inputs_embeds,)
