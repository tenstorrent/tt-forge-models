# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
prithivMLmods FireRed-OCR GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoConfig,
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


def _patch_qwen3vl_gguf():
    """Monkey-patch transformers to add qwen3vl GGUF architecture support.

    Transformers 5.x has Qwen3VLForConditionalGeneration but lacks GGUF loading
    support for the qwen3vl architecture. The gguf library already knows about
    qwen3vl tensor names, so we only need to bridge transformers' config/tensor
    processing layer.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
    )
    from transformers.integrations.ggml import (
        GGUF_CONFIG_MAPPING,
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen3vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # 1. Register qwen3vl config key mapping
    GGUF_CONFIG_MAPPING["qwen3vl"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": "head_dim",
        "vocab_size": "vocab_size",
        "rope.dimension_sections": None,
        "attention.value_length": None,
        "n_deepstack_layers": None,
    }

    # 2. Register qwen3vl as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    # 3. Register tokenizer converter (BPE-based, same as qwen2/qwen3)
    if "qwen3vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3vl"] = GGUFQwen2Converter

    # 4. Patch load_gguf_checkpoint to post-process qwen3vl config
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen3vl":
            # Fix model_type to match Qwen3VLConfig.model_type
            config["model_type"] = "qwen3_vl"
            # Build text_config from flat GGUF params for Qwen3VLConfig nested layout
            text_keys = {
                "max_position_embeddings",
                "num_hidden_layers",
                "intermediate_size",
                "hidden_size",
                "rope_theta",
                "num_attention_heads",
                "num_key_value_heads",
                "rms_norm_eps",
                "head_dim",
                "vocab_size",
            }
            text_config = {}
            for k in list(config.keys()):
                if k in text_keys:
                    text_config[k] = config[k]
            if text_config:
                config["text_config"] = text_config
            # Keep num_hidden_layers at top level so get_gguf_hf_weights_map can
            # access hf_model.config.num_hidden_layers without going through text_config
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # 5. Patch get_gguf_hf_weights_map to handle qwen3_vl composite config:
    #    - map model_type "qwen3_vl" -> "qwen3vl" for gguf library lookup
    #    - resolve num_layers from text_config when not on the top-level config
    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None and hasattr(hf_model, "config"):
            model_type = hf_model.config.model_type
        if model_type == "qwen3_vl":
            model_type = "qwen3vl"
        if num_layers is None and hasattr(hf_model, "config"):
            cfg = hf_model.config
            if not hasattr(cfg, "num_hidden_layers") and hasattr(cfg, "text_config"):
                num_layers = cfg.text_config.num_hidden_layers
        return _orig_get_gguf_hf_weights_map(
            hf_model,
            processor,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map

    # Patch modules that imported load_gguf_checkpoint directly
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_qwen3vl_gguf()


class ModelVariant(StrEnum):
    """Available prithivMLmods FireRed-OCR GGUF model variants for image to text."""

    FIRERED_OCR_Q8_0_GGUF = "firered_ocr_q8_0_gguf"


class ModelLoader(ForgeModel):
    """prithivMLmods FireRed-OCR GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.FIRERED_OCR_Q8_0_GGUF: LLMModelConfig(
            pretrained_model_name="prithivMLmods/FireRed-OCR-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FIRERED_OCR_Q8_0_GGUF

    GGUF_FILE = "FireRed-OCR.Q8_0.gguf"
    BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="prithivMLmods FireRed-OCR GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_DOC_OCR,
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
        # Loading config from the base model ensures the vision config (including
        # out_hidden_size) is consistent with the GGUF text model weights.
        self.processor = AutoProcessor.from_pretrained(self.BASE_MODEL)
        config = AutoConfig.from_pretrained(self.BASE_MODEL)

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name,
            config=config,
            ignore_mismatched_sizes=True,
            **model_kwargs,
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
