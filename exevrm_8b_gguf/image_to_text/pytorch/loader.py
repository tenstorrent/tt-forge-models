# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ExeVRM 8B GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen3VLForConditionalGeneration,
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


def _patch_transformers_qwen3vl_gguf():
    """Monkey-patch transformers to add qwen3vl GGUF architecture support."""
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen3vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    from transformers.integrations.ggml import GGUF_CONFIG_MAPPING

    GGUF_CONFIG_MAPPING["qwen3vl"] = GGUF_CONFIG_MAPPING["qwen3"].copy()
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vl"] = GGUF_CONFIG_MAPPING["qwen3vl"]

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "qwen3vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3vl"] = GGUFQwen2Converter

    _text_config_keys = {
        "max_position_embeddings",
        "num_hidden_layers",
        "intermediate_size",
        "hidden_size",
        "rope_theta",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "vocab_size",
    }

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen3vl":
            text_config = {"model_type": "qwen3_vl_text"}
            for key in list(config.keys()):
                if key in _text_config_keys:
                    text_config[key] = config.pop(key)
            config["text_config"] = text_config
            config["model_type"] = "qwen3_vl"
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_transformers_qwen3vl_gguf()


class ModelVariant(StrEnum):
    """Available ExeVRM 8B GGUF model variants for image to text."""

    EXEVRM_8B_Q4_K_M_GGUF = "8b_q4_k_m_gguf"


class ModelLoader(ForgeModel):
    """ExeVRM 8B GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.EXEVRM_8B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/ExeVRM-8B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EXEVRM_8B_Q4_K_M_GGUF

    GGUF_FILE = "ExeVRM-8B.Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ExeVRM 8B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoConfig

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

        # Load config from base model to ensure vision and text configs are
        # consistent; the GGUF loader does not populate the vision config.
        model_kwargs["config"] = AutoConfig.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
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
