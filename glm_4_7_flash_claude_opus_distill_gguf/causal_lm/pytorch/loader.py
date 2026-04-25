# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM 4.7 Flash Claude Opus Distill GGUF model loader implementation for causal language modeling.

Note: The deepseek2/deepseek_v2 GGUF architecture is not fully supported by transformers.
We load the tokenizer from the HF-native base model and patch get_gguf_hf_weights_map
to map deepseek_v2 -> deepseek2 so gguf-py can resolve the weight name mapping.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

# The HF-native base model for tokenizer loading (deepseek2 GGUF tokenizer arch not supported).
HF_TOKENIZER_MODEL = "unsloth/GLM-4.7-Flash"

_orig_get_gguf_hf_weights_map = _gguf_utils.get_gguf_hf_weights_map


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    if model_type is None:
        model_type = hf_model.config.model_type
    # deepseek_v2 HF config model_type maps to deepseek2 in gguf-py MODEL_ARCH_NAMES
    if model_type == "deepseek_v2":
        model_type = "deepseek2"
    return _orig_get_gguf_hf_weights_map(
        hf_model, processor, model_type, num_layers, qual_name
    )


_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


class ModelVariant(StrEnum):
    """Available GLM 4.7 Flash Claude Opus Distill GGUF model variants for causal language modeling."""

    GLM_4_7_FLASH_CLAUDE_OPUS_DISTILL_V2_HERETIC_I1_GGUF = (
        "4_7_Flash_Claude_Opus_4_5_High_Reasoning_Distill_v2_heretic_i1_GGUF"
    )
    AIWORKSOFBT_GLM_4_7_FLASH_CLAUDE_OPUS_DISTILL_GGUF = (
        "aiworksofbt_4_7_Flash_Claude_Opus_4_5_High_Reasoning_Distill_GGUF"
    )


class ModelLoader(ForgeModel):
    """GLM 4.7 Flash Claude Opus Distill GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GLM_4_7_FLASH_CLAUDE_OPUS_DISTILL_V2_HERETIC_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-v2-heretic-i1-GGUF",
            max_length=128,
        ),
        ModelVariant.AIWORKSOFBT_GLM_4_7_FLASH_CLAUDE_OPUS_DISTILL_GGUF: LLMModelConfig(
            pretrained_model_name="aiworksofbt/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_4_7_FLASH_CLAUDE_OPUS_DISTILL_V2_HERETIC_I1_GGUF

    _GGUF_FILES = {
        ModelVariant.GLM_4_7_FLASH_CLAUDE_OPUS_DISTILL_V2_HERETIC_I1_GGUF: "GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-v2-heretic.i1-Q4_K_M.gguf",
        ModelVariant.AIWORKSOFBT_GLM_4_7_FLASH_CLAUDE_OPUS_DISTILL_GGUF: "glm-4.7-flash-claude-4.5-opus.q4_k_m.gguf",
    }

    sample_text = "Give me a short introduction to large language model."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="GLM 4.7 Flash Claude Opus Distill GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self._GGUF_FILES[self._variant]

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self._GGUF_FILES[self._variant]
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts = [text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._GGUF_FILES[self._variant],
        )
        return self.config
