# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Phi-1.5 GGUF model loader implementation for causal language modeling.
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


def _patch_transformers_phi2_gguf():
    """Monkey-patch transformers to add phi2 GGUF architecture support.

    Transformers does not natively support loading phi2 GGUF checkpoints.
    This patch registers the phi2 architecture (used by Phi-1.5 and Phi-2)
    and adds a custom tensor processor that splits the fused QKV tensor
    stored in phi2 GGUF files into separate q_proj, k_proj, v_proj tensors
    expected by PhiForCausalLM.
    """
    import numpy as np
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        TensorProcessor,
        GGUFTensor,
    )

    if "phi2" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("phi2")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["phi2"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_epsilon": "layer_norm_eps",
        "rope.dimension_count": None,
        "vocab_size": "vocab_size",
    }

    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFGPTConverter

    if "phi2" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["phi2"] = GGUFGPTConverter

    class Phi2TensorProcessor(TensorProcessor):
        """Handles phi2 GGUF tensors.

        phi2 GGUF stores attention Q, K, V as a single fused tensor
        (blk.N.attn_qkv) while PhiForCausalLM expects separate q_proj,
        k_proj, v_proj.  This processor splits the fused tensor using the
        tensor_key_mapping to derive the correct HF parameter names.
        """

        def process(self, weights, name, **kwargs):
            tensor_key_mapping = kwargs.get("tensor_key_mapping", {})
            parsed_parameters = kwargs.get("parsed_parameters", {})

            if "tensors" not in parsed_parameters:
                parsed_parameters["tensors"] = {}

            # Handle output.weight -> lm_head.weight and output.bias -> lm_head.bias
            if name == "output.weight":
                parsed_parameters["tensors"]["lm_head.weight"] = torch.from_numpy(
                    np.copy(weights)
                )
                return GGUFTensor(weights, None, {})
            if name == "output.bias":
                parsed_parameters["tensors"]["lm_head.bias"] = torch.from_numpy(
                    np.copy(weights)
                )
                return GGUFTensor(weights, None, {})

            # Split fused QKV into separate Q, K, V tensors
            if "attn_qkv" in name:
                parts = name.split(".")
                block_num = parts[1]
                suffix = "." + parts[-1]  # ".weight" or ".bias"

                q, k, v = np.array_split(weights, 3, axis=0)

                for split_w, component in [(q, "attn_q"), (k, "attn_k"), (v, "attn_v")]:
                    gguf_key = f"blk.{block_num}.{component}{suffix}"
                    if gguf_key in tensor_key_mapping:
                        hf_name = tensor_key_mapping[gguf_key]
                        parsed_parameters["tensors"][hf_name] = torch.from_numpy(
                            np.copy(split_w)
                        )

                return GGUFTensor(weights, None, {})

            return GGUFTensor(weights, name, {})

    TENSOR_PROCESSORS["phi2"] = Phi2TensorProcessor

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "phi2":
            config["model_type"] = "phi"
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_transformers_phi2_gguf()


class ModelVariant(StrEnum):
    """Available Phi-1.5 GGUF model variants for causal language modeling."""

    PHI_1_5_Q4_K_M = "Phi_1_5_Q4_K_M"


class ModelLoader(ForgeModel):
    """Phi-1.5 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.PHI_1_5_Q4_K_M: LLMModelConfig(
            pretrained_model_name="TKDKid1000/phi-1_5-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PHI_1_5_Q4_K_M

    GGUF_FILE = "phi-1_5-Q4_K_M.gguf"

    sample_text = "Africa is an emerging economy because"

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
            model="Phi-1.5 GGUF",
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
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

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
            padding="max_length",
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
