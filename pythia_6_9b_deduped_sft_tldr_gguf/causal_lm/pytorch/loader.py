# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RichardErkhov Pythia 6.9B Deduped SFT TLDR GGUF model loader implementation
for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def _patch_transformers_gptneox_gguf():
    """Monkey-patch transformers to add gptneox GGUF architecture support.

    Transformers does not include gptneox in its GGUF config/tokenizer
    mappings.  This patch registers the architecture so that GPT-NeoX /
    Pythia GGUF checkpoints can be loaded.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "gptneox" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("gptneox")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["gptneox"] = {
        "context_length": "max_position_embeddings",
        "embedding_length": "hidden_size",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "attention.head_count": "num_attention_heads",
        "attention.layer_norm_epsilon": "layer_norm_eps",
        "rope.dimension_count": None,
        "vocab_size": "vocab_size",
    }

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFGPTConverter,
    )

    if "gptneox" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["gptneox"] = GGUFGPTConverter

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "gptneox":
            config["model_type"] = "gpt_neox"
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, *args, **kwargs
    ):
        effective_type = (
            hf_model.config.model_type if model_type is None else model_type
        )
        if effective_type == "gpt_neox":
            model_type = "gptneox"
        return orig_get_map(hf_model, processor, model_type, *args, **kwargs)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


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
    """Available Pythia 6.9B Deduped SFT TLDR GGUF model variants for causal language modeling."""

    PYTHIA_6_9B_DEDUPED_SFT_TLDR_Q4_K_M_GGUF = "6_9B_DEDUPED_SFT_TLDR_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """RichardErkhov Pythia 6.9B Deduped SFT TLDR GGUF model loader implementation
    for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.PYTHIA_6_9B_DEDUPED_SFT_TLDR_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="RichardErkhov/HuggingFaceH4_-_EleutherAI_pythia-6.9b-deduped__sft__tldr-gguf",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PYTHIA_6_9B_DEDUPED_SFT_TLDR_Q4_K_M_GGUF

    GGUF_FILE = "EleutherAI_pythia-6.9b-deduped__sft__tldr.Q4_K_M.gguf"

    sample_text = (
        "SUBREDDIT: r/relationships\n"
        "TITLE: My best friend is moving across the country\n"
        "POST: My best friend of ten years just told me she's moving to the "
        "other side of the country for a new job. I am happy for her but also "
        "heartbroken. We've been through everything together and now I don't "
        "know how to cope with the distance.\n"
        "TL;DR:"
    )

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
            model="Pythia 6.9B Deduped SFT TLDR GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _patch_transformers_gptneox_gguf()
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
        _patch_transformers_gptneox_gguf()
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
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        _patch_transformers_gptneox_gguf()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
