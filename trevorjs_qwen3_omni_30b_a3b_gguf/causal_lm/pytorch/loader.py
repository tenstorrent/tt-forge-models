# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TrevorJS Qwen3-Omni-30B-A3B GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _patch_gguf_utils_for_qwen3omnimoe():
    """Patch transformers GGUF utils to support qwen3omnimoe architecture as qwen3_moe."""
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.integrations.ggml import GGUF_CONFIG_MAPPING
    from transformers.modeling_gguf_pytorch_utils import (
        TENSOR_PROCESSORS,
        Qwen2MoeTensorProcessor,
    )

    arch = "qwen3omnimoe"
    target = "qwen3_moe"

    if arch not in GGUF_CONFIG_MAPPING:
        # GGUF_CONFIG_MAPPING is the same object as GGUF_TO_TRANSFORMERS_MAPPING["config"]
        GGUF_CONFIG_MAPPING[arch] = GGUF_CONFIG_MAPPING[target].copy()
        TENSOR_PROCESSORS[arch] = TENSOR_PROCESSORS.get(
            "qwen3moe", Qwen2MoeTensorProcessor
        )
        gguf_utils.GGUF_SUPPORTED_ARCHITECTURES = list(
            gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"].keys()
        )


_patch_gguf_utils_for_qwen3omnimoe()

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
    """Available TrevorJS Qwen3-Omni-30B-A3B GGUF model variants for causal language modeling."""

    TREVORJS_QWEN3_OMNI_30B_A3B_GGUF = "30B_A3B_GGUF"


class ModelLoader(ForgeModel):
    """TrevorJS Qwen3-Omni-30B-A3B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.TREVORJS_QWEN3_OMNI_30B_A3B_GGUF: LLMModelConfig(
            pretrained_model_name="TrevorJS/Qwen3-Omni-30B-A3B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TREVORJS_QWEN3_OMNI_30B_A3B_GGUF

    GGUF_FILE = "thinker-q4_k_m.gguf"

    sample_text = "Give me a short introduction to large language models."

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
            model="TrevorJS Qwen3-Omni-30B-A3B GGUF",
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
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
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
        model_kwargs["gguf_file"] = self.GGUF_FILE

        config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        # qwen3omnimoe is the GGUF architecture name for Qwen3-Omni MoE; remap to
        # qwen3_moe so that AutoModelForCausalLM resolves to Qwen3MoeForCausalLM.
        if getattr(config, "model_type", None) == "qwen3omnimoe":
            config.model_type = "qwen3_moe"
        if self.num_layers is not None:
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

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
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
