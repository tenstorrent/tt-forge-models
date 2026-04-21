# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nekomata 14B PFN QFin GGUF model loader implementation for causal language modeling.
"""
import torch
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


def _patch_transformers_qwen_gguf():
    """Register 'qwen' GGUF architecture as an alias for 'qwen2'."""
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    if "qwen" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen")

    for section in GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen2" in GGUF_TO_TRANSFORMERS_MAPPING[section]:
            GGUF_TO_TRANSFORMERS_MAPPING[section][
                "qwen"
            ] = GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen2"]

    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    qwen2_converter = GGUF_TO_FAST_CONVERTERS.get("qwen2")
    if qwen2_converter:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen", qwen2_converter)

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "qwen":
            result["config"]["model_type"] = "qwen2"
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint
    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_transformers_qwen_gguf()


class ModelVariant(StrEnum):
    """Available Nekomata 14B PFN QFin GGUF model variants for causal language modeling."""

    NEKOMATA_14B_PFN_QFIN_Q4_K_M_GGUF = "Nekomata_14B_PFN_QFin_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Nekomata 14B PFN QFin GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEKOMATA_14B_PFN_QFIN_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="RichardErkhov/pfnet_-_nekomata-14b-pfn-qfin-gguf",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEKOMATA_14B_PFN_QFIN_Q4_K_M_GGUF

    GGUF_FILE = "nekomata-14b-pfn-qfin.Q4_K_M.gguf"

    sample_text = "What is your favorite city?"

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
            model="Nekomata 14B PFN QFin GGUF",
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

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
