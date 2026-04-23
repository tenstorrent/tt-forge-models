# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MediX R1 30B GGUF model loader implementation for image to text.
"""

from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
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

    The qwen3vlmoe GGUF format stores Qwen3-VL-MoE LLM weights only (vision
    encoder is not included). Transformers 5.x lacks GGUF loading support for
    the qwen3vlmoe architecture. This patch:
    1. Registers qwen3vlmoe as a supported GGUF architecture (same config
       fields as qwen3_moe).
    2. Patches load_gguf_checkpoint to set model_type = "qwen3_vl_moe".
    3. Patches get_gguf_hf_weights_map to handle the qwen3_vl_moe model type
       and the composite VL config that lacks a top-level num_hidden_layers.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        load_gguf_checkpoint as _orig_load,
        get_gguf_hf_weights_map as _orig_weights_map,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen3vlmoe" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # 1. Register qwen3vlmoe as a supported architecture (same fields as qwen3_moe)
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vlmoe")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vlmoe"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "moe_intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.key_length": "head_dim",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
        "expert_count": "num_experts",
        "expert_used_count": "num_experts_per_tok",
    }

    # 2. Register tokenizer converter (BPE-based, same as qwen3)
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "qwen3vlmoe" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3vlmoe"] = GGUFQwen2Converter

    # 3. Patch load_gguf_checkpoint to set model_type and restructure config.
    #    Qwen3VLMoeConfig expects LLM params nested under text_config, not flat.
    _LLM_FIELDS = {
        "hidden_size",
        "num_hidden_layers",
        "moe_intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "vocab_size",
        "rms_norm_eps",
        "rope_theta",
        "head_dim",
        "num_experts",
        "num_experts_per_tok",
    }

    def _patched_load_gguf_checkpoint(*args, **kwargs):
        result = _orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen3vlmoe":
            config["model_type"] = "qwen3_vl_moe"
            # Move flat LLM params into text_config so Qwen3VLMoeConfig picks
            # up the correct architecture instead of defaults.
            text_cfg = {k: config.pop(k) for k in _LLM_FIELDS if k in config}
            if text_cfg:
                config["text_config"] = text_cfg
                # vision_config.out_hidden_size must match LLM hidden_size.
                llm_hidden = text_cfg.get("hidden_size")
                if llm_hidden is not None:
                    config["vision_config"] = {"out_hidden_size": llm_hidden}
        return result

    gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    # 4. Patch get_gguf_hf_weights_map to handle qwen3_vl_moe composite config.
    #    Qwen3VLMoeConfig has no num_hidden_layers at the top level (it is
    #    nested in text_config). We bypass the full traversal and map
    #    language_model directly to avoid vision encoder collision.
    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type == "qwen3_vl_moe":
            model_type = "qwen3vlmoe"
        if num_layers is None:
            num_layers = getattr(hf_model.config, "num_hidden_layers", None)
            if num_layers is None:
                text_cfg = getattr(hf_model.config, "text_config", None)
                num_layers = (
                    getattr(text_cfg, "num_hidden_layers", 1) if text_cfg else 1
                )

        # For qwen3vlmoe VL models at the top-level call, build the map only
        # from model.language_model to avoid vision.merger.norm collision.
        if model_type == "qwen3vlmoe" and qual_name == "":
            model_sub = getattr(hf_model, "model", None)
            if model_sub is not None and hasattr(model_sub, "language_model"):
                return _orig_weights_map(
                    model_sub.language_model,
                    processor,
                    model_type=model_type,
                    num_layers=num_layers,
                    qual_name="model.language_model.",
                )

        return _orig_weights_map(
            hf_model,
            processor,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils

    for mod in (tok_auto, config_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = _patched_load_gguf_checkpoint


_patch_transformers_qwen3vlmoe_gguf()


class ModelVariant(StrEnum):
    """Available MediX R1 30B GGUF model variants for image to text."""

    MEDIX_R1_30B_Q4_K_M = "30b_q4_k_m"


class ModelLoader(ForgeModel):
    """MediX R1 30B GGUF model loader implementation for image to text tasks."""

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

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

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

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["gguf_file"] = gguf_file
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
        )

        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.sample_image,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs
