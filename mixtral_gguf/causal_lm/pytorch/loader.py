# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mixtral GGUF model loader implementation for causal language modeling.
"""

import glob
import os
import re
from typing import Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, MixtralConfig

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _patch_transformers_mixtral_gguf():
    """Monkey-patch transformers to handle Mixtral GGUF loading.

    TheBloke Mixtral GGUF files use the old llama.cpp convention:
    - general.architecture = "llama"  (not "mixtral")
    - Per-expert tensors stored individually: blk.{bid}.ffn_{gate,down,up}.{eid}.weight

    Transformers 5.x has two problems with this:
    1. get_gguf_hf_weights_map raises NotImplementedError for model_type="mixtral"
    2. No processor handles the old per-expert tensor format
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUFTensor,
        LlamaTensorProcessor,
        TENSOR_PROCESSORS,
    )

    if "mixtral" in TENSOR_PROCESSORS:
        return  # already patched

    GGUF_EXPERT_PATTERN = re.compile(
        r"blk\.(?P<bid>\d+)\.ffn_(?P<w>gate|down|up)\.(?P<eid>\d+)\.weight$"
    )
    HF_W_MAP = {"gate": "w1", "up": "w3", "down": "w2"}

    class MixtralTensorProcessor(LlamaTensorProcessor):
        """Handles Mixtral GGUF files storing per-expert tensors in old llama.cpp format."""

        def process(self, weights, name: str, **kwargs):
            if m := re.fullmatch(GGUF_EXPERT_PATTERN, name):
                parsed_parameters = kwargs.get("parsed_parameters")
                if parsed_parameters is None:
                    return GGUFTensor(weights, name, {})
                bid, eid = int(m["bid"]), int(m["eid"])
                hf_w = HF_W_MAP[m["w"]]
                hf_key = (
                    f"model.layers.{bid}.block_sparse_moe"
                    f".experts.{eid}.{hf_w}.weight"
                )
                parsed_parameters["tensors"][hf_key] = torch.from_numpy(
                    np.copy(weights)
                )
                return GGUFTensor(weights, None, {})
            return super().process(weights, name, **kwargs)

    TENSOR_PROCESSORS["mixtral"] = MixtralTensorProcessor

    # Remap model_type="mixtral" → "llama" in the weight-map lookup so
    # non-expert tensors use the Llama weight mapping.
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type == "mixtral":
            model_type = "llama"
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map

    # When loading a MixtralForCausalLM, temporarily replace the llama
    # TENSOR_PROCESSOR with MixtralTensorProcessor so per-expert tensors
    # are correctly mapped to MixtralForCausalLM's expert sub-modules.
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        model_to_load = kwargs.get("model_to_load") or (
            args[2] if len(args) >= 3 else None
        )
        is_mixtral = (
            model_to_load is not None
            and hasattr(model_to_load, "config")
            and getattr(model_to_load.config, "model_type", "") == "mixtral"
        )
        if is_mixtral:
            orig_llama = TENSOR_PROCESSORS.get("llama")
            TENSOR_PROCESSORS["llama"] = MixtralTensorProcessor
            try:
                return orig_load(*args, **kwargs)
            finally:
                if orig_llama is not None:
                    TENSOR_PROCESSORS["llama"] = orig_llama
                elif "llama" in TENSOR_PROCESSORS:
                    del TENSOR_PROCESSORS["llama"]
        return orig_load(*args, **kwargs)

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_transformers_mixtral_gguf()


class ModelVariant(StrEnum):
    """Available Mixtral GGUF model variants for causal language modeling."""

    MIXTRAL_8X7B_INSTRUCT_V01_GGUF = "8x7B_Instruct_v0.1_GGUF"
    MIXTRAL_8X22B_INSTRUCT_V01_GGUF = "8x22B_Instruct_v0.1_GGUF"


class ModelLoader(ForgeModel):
    """Mixtral GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MIXTRAL_8X7B_INSTRUCT_V01_GGUF: LLMModelConfig(
            pretrained_model_name="TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
            max_length=128,
        ),
        ModelVariant.MIXTRAL_8X22B_INSTRUCT_V01_GGUF: LLMModelConfig(
            pretrained_model_name="MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MIXTRAL_8X7B_INSTRUCT_V01_GGUF

    _GGUF_FILES = {
        ModelVariant.MIXTRAL_8X7B_INSTRUCT_V01_GGUF: "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
        ModelVariant.MIXTRAL_8X22B_INSTRUCT_V01_GGUF: "Mixtral-8x22B-Instruct-v0.1.Q2_K-00001-of-00003.gguf",
    }

    sample_text = "What is the capital of France?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Mixtral GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def _build_mixtral_config_from_gguf(self):
        # transformers 5.x loads Mixtral GGUFs whose general.architecture is
        # "llama" as LlamaForCausalLM, silently dropping expert weights and
        # producing garbage output.  Build an explicit MixtralConfig from the
        # GGUF metadata so from_pretrained instantiates MixtralForCausalLM.
        try:
            from gguf import GGUFReader
            from transformers.modeling_gguf_pytorch_utils import _gguf_parse_value

            safe_id = self._variant_config.pretrained_model_name.replace("/", "--")
            pattern = os.path.expanduser(
                f"~/.cache/huggingface/hub/models--{safe_id}/snapshots/**/{self.gguf_file}"
            )
            matches = glob.glob(pattern, recursive=True)
            if not matches:
                return None
            reader = GGUFReader(matches[0])

            def _read(field_name):
                f = reader.fields.get(field_name)
                if f is None:
                    return None
                return _gguf_parse_value(f.parts[f.data[0]], f.types)

            expert_count = _read("llama.expert_count")
            if not expert_count or int(expert_count) == 0:
                return None

            tokens_field = reader.fields.get("tokenizer.ggml.tokens")
            vocab_size = (
                len(tokens_field.data)
                if tokens_field is not None
                else int(_read("llama.vocab_size") or 32000)
            )

            config = MixtralConfig(
                hidden_size=int(_read("llama.embedding_length") or 4096),
                intermediate_size=int(_read("llama.feed_forward_length") or 14336),
                num_hidden_layers=int(_read("llama.block_count") or 32),
                num_attention_heads=int(_read("llama.attention.head_count") or 32),
                num_key_value_heads=int(_read("llama.attention.head_count_kv") or 8),
                num_local_experts=int(expert_count),
                num_experts_per_tok=int(_read("llama.expert_used_count") or 2),
                rms_norm_eps=float(
                    _read("llama.attention.layer_norm_rms_epsilon") or 1e-5
                ),
                rope_theta=float(_read("llama.rope.freq_base") or 1000000.0),
                vocab_size=vocab_size,
            )
            if self.num_layers is not None:
                config.num_hidden_layers = self.num_layers
            return config
        except Exception:
            return None

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.gguf_file

        mixtral_config = self._build_mixtral_config_from_gguf()
        if mixtral_config is not None:
            model_kwargs["config"] = mixtral_config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # MixtralExperts.forward uses a Python for-loop with a dynamic trip
        # count derived from nonzero(expert_hit).  XLA cannot trace this, so
        # switch to batched_mm which uses only static tensor operations.
        model.config._experts_implementation = "batched_mm"

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
