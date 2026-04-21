# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LaViDa-LLaDA v1.0 Instruct model loader implementation for causal language modeling.
"""
import os
import shutil
import torch
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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
    """Available LaViDa-LLaDA model variants for causal language modeling."""

    LAVIDA_LLADA_V1_0_INSTRUCT = "lavida_llada_v1_0_instruct"


class ModelLoader(ForgeModel):
    """LaViDa-LLaDA v1.0 Instruct model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.LAVIDA_LLADA_V1_0_INSTRUCT: LLMModelConfig(
            pretrained_model_name="jacklishufan/lavida-llada-v1.0-instruct",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LAVIDA_LLADA_V1_0_INSTRUCT

    sample_text = "What is the capital of France?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LaViDa-LLaDA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._get_local_model_dir(),
            trust_remote_code=True,
            **tokenizer_kwargs,
        )

        return self.tokenizer

    def _get_local_model_dir(self):
        """Download model snapshot and inject missing modeling_llada.py from GSAI-ML/LLaDA-8B-Instruct.

        In transformers 5.x, use_cache is treated as a GenerationConfig param and is
        not stored on PretrainedConfig, so we patch the modeling file to use getattr.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        model_dir = snapshot_download(pretrained_model_name)
        modeling_dst = os.path.join(model_dir, "modeling_llada.py")
        if not os.path.exists(modeling_dst):
            modeling_src = hf_hub_download(
                "GSAI-ML/LLaDA-8B-Instruct", "modeling_llada.py"
            )
            with open(modeling_src) as f:
                content = f.read()
            content = content.replace(
                "self.config.use_cache",
                "getattr(self.config, 'use_cache', False)",
            )
            with open(modeling_dst, "w") as f:
                f.write(content)
        return model_dir

    def load_model(self, *, dtype_override=None, **kwargs):
        model_dir = self._get_local_model_dir()

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            config.n_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, **model_kwargs
        )
        model.eval()
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {"role": "user", "content": self.sample_text},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts = [text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._get_local_model_dir(),
            trust_remote_code=True,
        )

        return self.config
