# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
bartowski browser-use BU-30B-A3B-Preview GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLMoeConfig,
    AutoProcessor,
)
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

    qwen3vlmoe is the GGUF architecture string for Qwen3 Vision-Language MoE
    models. transformers has Qwen3VLMoe model support but lacks the GGUF loader
    registration needed to load qwen3vlmoe GGUF checkpoints.
    """
    import re
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        Qwen2MoeTensorProcessor,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.integrations.ggml import (
        GGUF_CONFIG_MAPPING,
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    class _Qwen3VLMoeTensorProcessor(Qwen2MoeTensorProcessor):
        """MoE tensor processor for qwen3vlmoe GGUF checkpoints.

        Extends Qwen2MoeTensorProcessor to handle the extra `language_model`
        path segment in Qwen3VLMoe VL model weight names:
          model.language_model.layers.N.mlp.experts.*
        instead of:
          model.layers.N.mlp.experts.*
        """

        HF_VL_MOE_W13_PATTERN = re.compile(
            r"model\.language_model\.layers\.(?P<bid>\d+)\.mlp\.experts\.gate_up_proj"
        )
        HF_VL_MOE_DOWN_PATTERN = re.compile(
            r"model\.language_model\.layers\.(?P<bid>\d+)\.mlp\.experts\.down_proj"
        )

        def perform_fallback_tensor_mapping(
            self, gguf_to_hf_name_map, suffix, qual_name, hf_name
        ):
            super().perform_fallback_tensor_mapping(
                gguf_to_hf_name_map, suffix, qual_name, hf_name
            )
            if m := re.fullmatch(self.HF_VL_MOE_W13_PATTERN, hf_name):
                full_hf_name = qual_name + hf_name
                gguf_to_hf_name_map[
                    f"blk.{m['bid']}.ffn_gate_exps{suffix}"
                ] = full_hf_name
                gguf_to_hf_name_map[
                    f"blk.{m['bid']}.ffn_up_exps{suffix}"
                ] = full_hf_name
            elif m := re.fullmatch(self.HF_VL_MOE_DOWN_PATTERN, hf_name):
                full_hf_name = qual_name + hf_name
                gguf_to_hf_name_map[
                    f"blk.{m['bid']}.ffn_down_exps{suffix}"
                ] = full_hf_name

    if "qwen3vlmoe" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vlmoe")

    # qwen3vlmoe uses the same GGUF metadata field names as qwen3_moe for the
    # text part of the model.
    _QWEN3VLMOE_CONFIG_MAPPING = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
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

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vlmoe"] = _QWEN3VLMOE_CONFIG_MAPPING

    # Register the VL-aware MoE tensor processor so expert weights load correctly.
    TENSOR_PROCESSORS.setdefault("qwen3vlmoe", _Qwen3VLMoeTensorProcessor)

    if "qwen3_moe" in GGUF_CONFIG_MAPPING:
        GGUF_CONFIG_MAPPING.setdefault("qwen3vlmoe", GGUF_CONFIG_MAPPING["qwen3_moe"])

    if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3vlmoe", GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
        )
    else:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen3vlmoe", GGUFQwen2Converter)

    _QWEN3VLMOE_TEXT_CONFIG_KEYS = {
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "max_position_embeddings",
        "vocab_size",
        "head_dim",
        "num_experts",
        "num_experts_per_tok",
    }

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen3vlmoe":
            config["model_type"] = "qwen3_vl_moe"
            text_config = {}
            for key in list(config.keys()):
                if key in _QWEN3VLMOE_TEXT_CONFIG_KEYS:
                    text_config[key] = config.pop(key)
            if text_config:
                config["text_config"] = text_config
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    orig_weights_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type in ("qwen3vlmoe", "qwen3_vl_moe"):
            # Use the GGUF arch name so get_tensor_name_map generates the
            # correct VL tensor name map (which knows the language_model prefix).
            # num_hidden_layers lives in text_config for composite VL configs.
            model_type = "qwen3vlmoe"
            if num_layers is None:
                try:
                    tc = hf_model.config.text_config
                    nl = getattr(tc, "num_hidden_layers", None)
                    if nl is not None:
                        num_layers = nl
                except AttributeError:
                    pass
            if num_layers is None:
                try:
                    num_layers = len(hf_model.model.language_model.layers)
                except AttributeError:
                    pass
        return orig_weights_map(
            hf_model,
            processor,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


_patch_transformers_qwen3vlmoe_gguf()


class ModelVariant(StrEnum):
    """Available bartowski browser-use BU-30B-A3B-Preview GGUF model variants for image to text."""

    BARTOWSKI_BROWSER_USE_BU_30B_A3B_PREVIEW_GGUF = (
        "browser_use_bu_30b_a3b_preview_gguf"
    )


class ModelLoader(ForgeModel):
    """bartowski browser-use BU-30B-A3B-Preview GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.BARTOWSKI_BROWSER_USE_BU_30B_A3B_PREVIEW_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/browser-use_bu-30b-a3b-preview-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BARTOWSKI_BROWSER_USE_BU_30B_A3B_PREVIEW_GGUF

    GGUF_FILE = "browser-use_bu-30b-a3b-preview-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="bartowski browser-use BU-30B-A3B-Preview GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("browser-use/bu-30b-a3b-preview")

        # The GGUF only contains quantized text weights; vision config keys are
        # absent from GGUF metadata. Supply the full config from the base model
        # so the vision encoder is instantiated with correct dimensions.
        config = Qwen3VLMoeConfig.from_pretrained("browser-use/bu-30b-a3b-preview")
        model_kwargs["config"] = config

        # ignore_mismatched_sizes=True: GGUF files for VL-MoE models typically
        # lack vision encoder weights, which causes benign shape-mismatch
        # warnings for randomly-initialized vision tensors.
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
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
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
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
