# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 DFlash model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
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


class ModelVariant(StrEnum):
    """Available Qwen 3 DFlash model variants for causal language modeling."""

    QWEN_3_4B_DFLASH_B16 = "4B_DFlash_b16"
    QWEN_3_5_35B_A3B_DFLASH = "3_5_35B_A3B_DFlash"


class ModelLoader(ForgeModel):
    """Qwen 3 DFlash model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_4B_DFLASH_B16: LLMModelConfig(
            pretrained_model_name="z-lab/Qwen3-4B-DFlash-b16",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_35B_A3B_DFLASH: LLMModelConfig(
            pretrained_model_name="z-lab/Qwen3.5-35B-A3B-DFlash",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_4B_DFLASH_B16

    sample_text = "Give me a short introduction to large language model."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self._is_dflash_draft = False
        self._hidden_size = None
        self._num_target_layers = None
        self._model_dtype = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen 3 DFlash",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
                trust_remote_code=True,
            )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs |= kwargs

        # Check if this is a DFlash draft model (uses AutoModel, not AutoModelForCausalLM)
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        architectures = getattr(config, "architectures", [])
        self._is_dflash_draft = "DFlashDraftModel" in architectures

        if self._is_dflash_draft:
            model = AutoModel.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()
            self._hidden_size = config.hidden_size
            dflash_cfg = getattr(config, "dflash_config", {})
            target_layer_ids = dflash_cfg.get("target_layer_ids", [])
            self._num_target_layers = len(target_layer_ids)
            self._model_dtype = next(model.parameters()).dtype
        else:
            self._ensure_tokenizer()
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        max_length = self._variant_config.max_length

        if self._is_dflash_draft:
            position_ids = torch.arange(max_length).unsqueeze(0).expand(batch_size, -1)
            noise_embedding = torch.zeros(
                batch_size, max_length, self._hidden_size, dtype=self._model_dtype
            )
            target_hidden = torch.zeros(
                batch_size,
                max_length,
                self._num_target_layers * self._hidden_size,
                dtype=self._model_dtype,
            )
            return {
                "position_ids": position_ids,
                "noise_embedding": noise_embedding,
                "target_hidden": target_hidden,
            }

        self._ensure_tokenizer()

        inputs = self.tokenizer(
            [self.sample_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
