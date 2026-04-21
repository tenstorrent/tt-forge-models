# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SongGeneration v2 Large model loader (lglg666/SongGeneration-v2-large).

SongGeneration (LeVo) is Tencent AI Lab's lyric-to-song generation framework.
The v2-large checkpoint packages a LLaMA-based audio language model plus a
Flow1dVAE audio tokenizer and Qwen2 text conditioners. The Hugging Face repo
ships only a raw ``config.yaml`` and a single ``model.pt`` torch checkpoint
(no ``config.json``), so this loader parses the YAML to reconstruct the LLaMA
backbone via ``transformers.LlamaForCausalLM``.
"""
from typing import Optional

import torch
import yaml
from huggingface_hub import hf_hub_download
from transformers import LlamaConfig, LlamaForCausalLM

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


class ModelVariant(StrEnum):
    """Available SongGeneration v2 variants."""

    V2_LARGE = "v2_large"


class ModelLoader(ForgeModel):
    """SongGeneration v2 Large loader for the LLaMA-based audio LM backbone."""

    _VARIANTS = {
        ModelVariant.V2_LARGE: ModelConfig(
            pretrained_model_name="lglg666/SongGeneration-v2-large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2_LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._llama_config: Optional[LlamaConfig] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="SongGeneration_v2_large",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_llama_config(self) -> LlamaConfig:
        """Parse config.yaml and build an equivalent LlamaConfig."""
        repo_id = self._variant_config.pretrained_model_name
        config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
        with open(config_path, "r") as fp:
            cfg = yaml.safe_load(fp)

        lm = cfg["lm"]
        self._llama_config = LlamaConfig(
            hidden_size=lm["dim"],
            intermediate_size=lm["intermediate_size"],
            num_hidden_layers=lm["num_layers"],
            num_attention_heads=lm["num_heads"],
            num_key_value_heads=lm["num_heads"],
            max_position_embeddings=lm["max_position_embeddings"],
            rope_theta=lm["rope_theta"],
            vocab_size=lm["code_size"],
            attention_bias=lm.get("bias_attn", False),
            mlp_bias=lm.get("bias_ff", False),
            use_cache=False,
        )
        return self._llama_config

    def load_model(self, *, dtype_override=None, **kwargs):
        """Build the LLaMA backbone matching the SongGeneration v2 LM config."""
        llama_config = self._load_llama_config()
        model = LlamaForCausalLM(llama_config)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size: int = 1):
        """Generate sample audio-token ids matching the LM vocabulary."""
        if self._llama_config is None:
            self._load_llama_config()

        seq_len = 128
        input_ids = torch.randint(
            0, self._llama_config.vocab_size, (batch_size, seq_len)
        )
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
