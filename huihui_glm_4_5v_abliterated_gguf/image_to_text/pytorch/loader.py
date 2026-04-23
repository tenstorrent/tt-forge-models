# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui GLM-4.5V Abliterated GGUF model loader implementation for image to text.
"""

import torch
from transformers import AutoTokenizer, Glm4MoeForCausalLM
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


def _patch_transformers_glm4moe_gguf():
    """Monkey-patch transformers to add glm4moe GGUF architecture support.

    The GLM-4.5V GGUF file uses the 'glm4moe' architecture identifier in its
    GGUF metadata, which represents the text-only MoE language model.
    Transformers 5.x has Glm4MoeForCausalLM but lacks GGUF loading support
    for the glm4moe architecture. We bridge the gap by registering the
    config/tokenizer/tensor-processor mappings and remapping model_type.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        Qwen2MoeTensorProcessor,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "glm4moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # 1. Register glm4moe as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("glm4moe")

    # 2. Add config mapping: GGUF key → Glm4MoeConfig field
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["glm4moe"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": "head_dim",
        "attention.value_length": None,
        "vocab_size": "vocab_size",
        "expert_count": "n_routed_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_shared_count": "n_shared_experts",
        "expert_group_count": "n_group",
        "expert_group_used_count": "topk_group",
        "expert_weights_scale": "routed_scaling_factor",
        "expert_weights_norm": "norm_topk_prob",
        "leading_dense_block_count": "first_k_dense_replace",
        "expert_feed_forward_length": "moe_intermediate_size",
    }

    # 3. Register BPE tokenizer converter (GLM4 uses the same tokenizer as Qwen2)
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "glm4moe" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["glm4moe"] = GGUFQwen2Converter

    # 4. Register MoE tensor processor (same merged gate+up pattern as Qwen2MoE)
    if "glm4moe" not in TENSOR_PROCESSORS:
        TENSOR_PROCESSORS["glm4moe"] = Qwen2MoeTensorProcessor

    # 5. Patch load_gguf_checkpoint: remap glm4moe → glm4_moe after loading
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "glm4moe":
            config["model_type"] = "glm4_moe"
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # 6. Patch get_gguf_hf_weights_map: add glm4_moe → glm4moe arch remapping
    orig_get_weights_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        effective_type = (
            model_type
            if model_type is not None
            else getattr(getattr(hf_model, "config", None), "model_type", None)
        )
        if effective_type == "glm4_moe":
            model_type = "glm4moe"
        return orig_get_weights_map(
            hf_model,
            processor,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map

    # Patch all modules that imported these functions directly
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


# Apply the monkey-patch at import time
_patch_transformers_glm4moe_gguf()


class ModelVariant(StrEnum):
    """Available Huihui GLM-4.5V Abliterated GGUF model variants for image to text."""

    HUIHUI_GLM_4_5V_ABLITERATED_MRADERMACHER_GGUF = (
        "huihui_glm_4_5v_abliterated_mradermacher_gguf"
    )


class ModelLoader(ForgeModel):
    """Huihui GLM-4.5V Abliterated GGUF model loader implementation for image to text tasks.

    The GGUF file contains the text-only MoE language model (glm4moe architecture).
    Inputs are text-only since the quantized GGUF does not include the vision encoder.
    """

    _VARIANTS = {
        ModelVariant.HUIHUI_GLM_4_5V_ABLITERATED_MRADERMACHER_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-GLM-4.5V-abliterated-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_GLM_4_5V_ABLITERATED_MRADERMACHER_GGUF

    _GGUF_FILES = {
        ModelVariant.HUIHUI_GLM_4_5V_ABLITERATED_MRADERMACHER_GGUF: "Huihui-GLM-4.5V-abliterated.Q4_K_M.gguf",
    }

    sample_text = "Describe the visual content you observe."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Huihui GLM-4.5V Abliterated GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @property
    def _gguf_file(self):
        """Get the GGUF filename for the current variant."""
        return self._GGUF_FILES.get(self._variant)

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["gguf_file"] = self._gguf_file
        model_kwargs |= kwargs

        # Load tokenizer from the base VLM since the GGUF repo has no tokenizer files
        self.tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.5V")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = Glm4MoeForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.5V")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            [self.sample_text] * batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)[
                    :batch_size
                ]

        return inputs
