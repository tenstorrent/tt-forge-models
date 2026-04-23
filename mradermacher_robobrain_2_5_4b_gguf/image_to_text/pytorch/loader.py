# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mradermacher RoboBrain 2.5 4B GGUF model loader implementation for image to text.
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
    """Monkey-patch transformers to add qwen3vl GGUF architecture support.

    The qwen3vl architecture (Qwen3-VL vision-language models) is not yet
    registered in transformers' GGUF loader. We reuse the qwen3 config key
    mapping for the text portion and fix the model_type in the parsed config.
    We also patch get_gguf_hf_weights_map to translate qwen3_vl -> qwen3vl
    for the gguf-py tensor name lookup.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen3vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vl"] = GGUF_TO_TRANSFORMERS_MAPPING[
        "config"
    ]["qwen3"].copy()
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "qwen3vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3vl"] = GGUFQwen2Converter

    orig_load = gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen3vl":
            config["model_type"] = "qwen3_vl"
        return result

    gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    # Patch get_gguf_hf_weights_map to translate qwen3_vl -> qwen3vl for
    # gguf-py MODEL_ARCH_NAMES lookup (transformers uses underscores, gguf-py
    # uses camelCase without underscores for VL architectures)
    orig_weights_map = gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(hf_model, *args, **kwargs):
        # model_type may arrive as args[0] (positional) or kwargs["model_type"]
        if args:
            mt = args[0]
            if mt == "qwen3_vl":
                args = ("qwen3vl",) + args[1:]
        else:
            mt = kwargs.get("model_type")
            if mt is None and hasattr(hf_model, "config"):
                mt = hf_model.config.model_type
            if mt == "qwen3_vl":
                kwargs["model_type"] = "qwen3vl"
        return orig_weights_map(hf_model, *args, **kwargs)

    gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_transformers_qwen3vl_gguf()


class ModelVariant(StrEnum):
    """Available Mradermacher RoboBrain 2.5 4B GGUF model variants for image to text."""

    ROBOBRAIN_2_5_4B_GGUF = "robobrain_2_5_4b_gguf"


class ModelLoader(ForgeModel):
    """Mradermacher RoboBrain 2.5 4B GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.ROBOBRAIN_2_5_4B_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/RoboBrain2.5-4B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ROBOBRAIN_2_5_4B_GGUF

    GGUF_FILE = "RoboBrain2.5-4B.Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Mradermacher RoboBrain 2.5 4B GGUF",
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
        self.processor = AutoProcessor.from_pretrained("BAAI/RoboBrain2.5-4B")

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
