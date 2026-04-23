# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MediX R1 30B GGUF model loader implementation for image to text.

The GGUF contains the Qwen3-VL-MoE text backbone; vision encoder weights are
absent. This loader targets the text backbone using AutoModelForCausalLM and
uses text-only inputs.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
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


def _patch_transformers_qwen3vlmoe_gguf():
    """Monkey-patch transformers to add qwen3vlmoe GGUF architecture support.

    The GGUF contains the Qwen3-VL-MoE text backbone; vision encoder weights
    are absent. We register qwen3vlmoe and remap model_type to qwen3_moe so
    that AutoModelForCausalLM loads the quantized text backbone correctly.
    """
    from transformers.integrations.ggml import GGUF_CONFIG_MAPPING
    from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen3vlmoe" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_CONFIG_MAPPING["qwen3vlmoe"] = GGUF_CONFIG_MAPPING["qwen3_moe"]
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vlmoe")

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen3vlmoe":
            config["model_type"] = "qwen3_moe"
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_transformers_qwen3vlmoe_gguf()


class ModelVariant(StrEnum):
    """Available MediX R1 30B GGUF model variants for image to text."""

    MEDIX_R1_30B_Q4_K_M = "30b_q4_k_m"


class ModelLoader(ForgeModel):
    """MediX R1 30B GGUF model loader implementation for image to text tasks.

    Loads the quantized Qwen3-VL-MoE text backbone from the GGUF file. The
    GGUF does not include vision encoder weights, so only the text backbone
    is tested.
    """

    _VARIANTS = {
        ModelVariant.MEDIX_R1_30B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="MBZUAI/MediX-R1-30B-GGUF",
            max_length=128,
        ),
    }

    _GGUF_FILES = {
        ModelVariant.MEDIX_R1_30B_Q4_K_M: "MediX-R1-30B-Q4_K_M.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.MEDIX_R1_30B_Q4_K_M

    TOKENIZER_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MediX R1 30B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["gguf_file"] = gguf_file
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            [self.sample_text] * batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._variant_config.max_length,
        )
        return inputs
