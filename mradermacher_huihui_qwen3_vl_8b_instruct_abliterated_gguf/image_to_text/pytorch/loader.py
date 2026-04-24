# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mradermacher Huihui Qwen3-VL 8B Instruct Abliterated GGUF model loader
implementation for image to text.
"""

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoConfig,
    AutoProcessor,
)
from typing import Optional


def _patch_transformers_qwen3vl_gguf():
    """Monkey-patch transformers to add qwen3vl GGUF architecture support.

    The qwen3vl GGUF architecture is not natively supported by transformers'
    GGUF loader. We map qwen3vl -> qwen3_vl for loading.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen3vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vl"] = dict(
        GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3"]
    )

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "qwen3vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3vl"] = GGUFQwen2Converter

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen3vl":
            config["model_type"] = "qwen3_vl"
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.configuration_utils as config_utils
    import transformers.tokenization_utils_tokenizers as tok_utils
    import transformers.models.auto.tokenization_auto as tok_auto

    config_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint
    tok_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint
    tok_auto.load_gguf_checkpoint = patched_load_gguf_checkpoint

    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type == "qwen3_vl":
            model_type = "qwen3vl"
            if num_layers is None:
                cfg = hf_model.config
                text_cfg = getattr(cfg, "text_config", cfg)
                num_layers = text_cfg.num_hidden_layers
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


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


class ModelVariant(StrEnum):
    """Available Mradermacher Huihui Qwen3-VL 8B Abliterated GGUF variants for image to text."""

    HUIHUI_QWEN3_VL_8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF = (
        "8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF"
    )


class ModelLoader(ForgeModel):
    """Mradermacher Huihui Qwen3-VL 8B Abliterated GGUF loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_QWEN3_VL_8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_QWEN3_VL_8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF

    _GGUF_FILES = {
        ModelVariant.HUIHUI_QWEN3_VL_8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF: "Huihui-Qwen3-VL-8B-Instruct-abliterated.Q4_K_M.gguf",
    }

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Mradermacher Huihui Qwen3-VL 8B Abliterated GGUF",
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
        _patch_transformers_qwen3vl_gguf()
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["gguf_file"] = self._gguf_file
        model_kwargs |= kwargs

        base_model = "Qwen/Qwen3-VL-8B-Instruct"
        self.processor = AutoProcessor.from_pretrained(base_model)
        config = AutoConfig.from_pretrained(base_model)

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
