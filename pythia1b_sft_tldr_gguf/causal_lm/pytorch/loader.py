# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RichardErkhov Pythia 1B SFT TLDR GGUF model loader implementation
for causal language modeling.
"""
import os
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


def _patch_transformers_gptneox_gguf():
    """Monkey-patch transformers to add gptneox GGUF architecture support.

    The Pythia model uses the 'gptneox' architecture identifier in its GGUF
    metadata. Transformers 5.x has GPTNeoXForCausalLM but lacks GGUF loading
    support for the gptneox architecture. We bridge the gap by registering
    the config mapping and remapping model_type to gpt_neox.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "gptneox" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # 1. Register gptneox as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("gptneox")

    # 2. Add config mapping for gptneox -> GPTNeoXConfig fields
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["gptneox"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "embedding_length": "hidden_size",
        "feed_forward_length": "intermediate_size",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_epsilon": "layer_norm_eps",
        "use_parallel_residual": "use_parallel_residual",
        "rope.freq_base": "rotary_emb_base",
        "rope.dimension_count": None,  # Computed to rotary_pct below
        "vocab_size": "vocab_size",
    }

    # 3. Register gpt_neox tokenizer converter (BPE, same family as GPT-2)
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFGPTConverter

    if "gpt_neox" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["gpt_neox"] = GGUFGPTConverter

    # 4. Patch load_gguf_checkpoint to remap model_type and compute rotary_pct
    _orig_load_gguf_checkpoint = gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(
        gguf_path, return_tensors=False, model_to_load=None
    ):
        result = _orig_load_gguf_checkpoint(
            gguf_path, return_tensors=return_tensors, model_to_load=model_to_load
        )
        config = result.get("config", {})
        if config.get("model_type") == "gptneox":
            config["model_type"] = "gpt_neox"
            config.setdefault("use_parallel_residual", True)
            # Compute rotary_pct from rope dimension count (Pythia default: 0.25)
            hidden_size = config.get("hidden_size", 2048)
            num_heads = config.get("num_attention_heads", 8)
            head_dim = hidden_size // num_heads if num_heads else 256
            try:
                from gguf import GGUFReader
                from transformers.modeling_gguf_pytorch_utils import _gguf_parse_value

                reader = GGUFReader(str(gguf_path))
                for key, field in reader.fields.items():
                    if "rope.dimension_count" in key:
                        rope_dim = _gguf_parse_value(
                            field.parts[field.data[0]], field.types
                        )
                        config["rotary_pct"] = rope_dim / head_dim
                        break
            except Exception:
                config.setdefault("rotary_pct", 0.25)
            config.setdefault("rotary_pct", 0.25)
        return result

    gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


_patch_transformers_gptneox_gguf()


class ModelVariant(StrEnum):
    """Available Pythia 1B SFT TLDR GGUF model variants for causal language modeling."""

    PYTHIA_1B_SFT_TLDR_Q4_K_M_GGUF = "1B_SFT_TLDR_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """RichardErkhov Pythia 1B SFT TLDR GGUF model loader implementation
    for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.PYTHIA_1B_SFT_TLDR_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="RichardErkhov/mnoukhov_-_pythia1b-sft-tldr-gguf",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PYTHIA_1B_SFT_TLDR_Q4_K_M_GGUF

    GGUF_FILE = "pythia1b-sft-tldr.Q4_K_M.gguf"

    # Non-GGUF model used as config/tokenizer source when TT_RANDOM_WEIGHTS is set
    # to avoid downloading the large GGUF file just for the model structure.
    BASE_MODEL_NAME = "EleutherAI/pythia-1b-deduped"

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
            model="Pythia 1B SFT TLDR GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        if os.environ.get("TT_RANDOM_WEIGHTS"):
            # Load tokenizer from base model to avoid downloading the large GGUF file.
            self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL_NAME)
        else:
            tokenizer_kwargs = {"gguf_file": self.GGUF_FILE}
            if dtype_override is not None:
                tokenizer_kwargs["torch_dtype"] = dtype_override
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

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            # Provide explicit config so the random_weights hook uses it directly
            # without trying to download the large GGUF file.
            config = AutoConfig.from_pretrained(self.BASE_MODEL_NAME)
            if self.num_layers is not None:
                config.num_hidden_layers = self.num_layers
            model_kwargs = {"config": config}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs
        else:
            model_kwargs = {"gguf_file": self.GGUF_FILE}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs
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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
