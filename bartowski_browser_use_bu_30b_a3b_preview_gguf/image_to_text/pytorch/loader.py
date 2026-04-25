# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
bartowski browser-use BU-30B-A3B-Preview GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen3VLMoeForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
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

    Transformers 5.x has Qwen3VLMoeForConditionalGeneration but lacks GGUF loading
    support for the qwen3vlmoe architecture. We bridge the gap by registering
    qwen3vlmoe config/tensor mappings and converting the flat GGUF config into
    Qwen3VLMoe's nested text_config format.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "qwen3vlmoe" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # 1. Register qwen3vlmoe as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vlmoe")

    # 2. Add config mapping for qwen3vlmoe (text LLM MoE portion)
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vlmoe"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_sections": None,
        "rope.freq_base": "rope_theta",
        "attention.key_length": "head_dim",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
        "expert_count": "num_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_feed_forward_length": "moe_intermediate_size",
        "n_deepstack_layers": None,
        "attention.value_length": None,
    }

    # 3. Reuse qwen3moe tensor processor for MoE expert weight handling
    if "qwen3moe" in TENSOR_PROCESSORS:
        TENSOR_PROCESSORS["qwen3vlmoe"] = TENSOR_PROCESSORS["qwen3moe"]

    # 4. Register tokenizer converter (reuse qwen3_moe BPE scheme)
    if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3vlmoe"] = GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
        GGUF_TO_FAST_CONVERTERS["qwen3_vl_moe_text"] = GGUF_TO_FAST_CONVERTERS[
            "qwen3_moe"
        ]

    # 5. Patch load_gguf_checkpoint to restructure flat config into nested text_config
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen3vlmoe":
            # Move text LLM keys into nested text_config so Qwen3VLMoeConfig
            # can be constructed with the correct dimensions.
            text_keys = {
                "max_position_embeddings",
                "num_hidden_layers",
                "intermediate_size",
                "hidden_size",
                "rope_theta",
                "head_dim",
                "num_attention_heads",
                "num_key_value_heads",
                "rms_norm_eps",
                "vocab_size",
                "num_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
                "tie_word_embeddings",
            }
            text_config = {
                k: config.pop(k) for k in list(config.keys()) if k in text_keys
            }
            text_config["model_type"] = "qwen3_vl_moe_text"
            config["text_config"] = text_config
            config["model_type"] = "qwen3_vl_moe"
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # 6. Patch get_gguf_hf_weights_map to map only the language_model submodule,
    # avoiding collision with vision encoder tensor names.
    #
    # NOTE: existing patches in this codebase drop the `processor` arg from their
    # signature, so the call chain passes positional args shifted by one.  Passing
    # a non-None num_layers through the chain would arrive at the original function
    # as model_type (an integer), causing NotImplementedError.  For qwen3vlmoe we
    # therefore bypass the chain entirely and build the weight map directly via the
    # gguf library.
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, model_type=None, num_layers=None, qual_name=""
    ):
        # Always read the true model type from config to avoid the arg-shift issue
        # where model_type may actually hold a TensorProcessor object.
        actual_type = getattr(getattr(hf_model, "config", None), "model_type", None)

        if actual_type in ("qwen3_vl_moe", "qwen3vlmoe") and qual_name == "":
            from gguf import MODEL_ARCH, get_tensor_name_map

            n_layers = num_layers
            if n_layers is None:
                cfg = hf_model.config
                tc = getattr(cfg, "text_config", cfg)
                n_layers = getattr(tc, "num_hidden_layers", None)

            lang_model = None
            if hasattr(hf_model, "model") and hasattr(hf_model.model, "language_model"):
                lang_model = hf_model.model.language_model

            if lang_model is not None and n_layers is not None:
                import re as _re

                proc = TENSOR_PROCESSORS["qwen3vlmoe"]()
                name_map = get_tensor_name_map(MODEL_ARCH.QWEN3MOE, n_layers)

                # gate_up_proj is a merged (gate+up) tensor that has no direct
                # name_map entry; we need to emit two GGUF keys for it.
                _gate_up_re = _re.compile(r"layers\.(\d+)\.mlp\.experts\.gate_up_proj$")

                gguf_to_hf_name_map = {}
                for hf_name in lang_model.state_dict():
                    preprocessed = proc.preprocess_name(hf_name)
                    name, suffix = preprocessed, ""
                    if preprocessed.endswith(".weight") or preprocessed.endswith(
                        ".bias"
                    ):
                        name, suffix = preprocessed.rsplit(".", 1)
                        suffix = "." + suffix

                    full_hf = "model.language_model." + hf_name

                    # name_map expects the full model-root path ("model.layers....")
                    gguf_name = name_map.get_name("model." + name)
                    if gguf_name is not None:
                        gguf_to_hf_name_map[gguf_name + suffix] = full_hf
                        continue

                    # gate_up_proj is split into separate gate/up GGUF tensors
                    m = _gate_up_re.match(name)
                    if m:
                        bid = m.group(1)
                        gguf_to_hf_name_map[
                            f"blk.{bid}.ffn_gate_exps{suffix}"
                        ] = full_hf
                        gguf_to_hf_name_map[f"blk.{bid}.ffn_up_exps{suffix}"] = full_hf

                return gguf_to_hf_name_map

        # Fallback: pass through unchanged so we don't disturb the arg-shift
        # convention relied on by other patches in the chain.
        return orig_get_map(hf_model, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map
    if hasattr(modeling_utils, "get_gguf_hf_weights_map"):
        modeling_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


# Apply patch at import time so all downstream from_pretrained calls benefit
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

        # GGUF repos do not ship a processor or full config; use the base model for both.
        # The base config ensures vision_config dimensions match the GGUF checkpoint.
        self.processor = AutoProcessor.from_pretrained("browser-use/bu-30b-a3b-preview")
        model_kwargs["config"] = AutoConfig.from_pretrained(
            "browser-use/bu-30b-a3b-preview"
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
