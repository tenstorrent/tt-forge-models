# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Robo-Dopamine GRM 8B i1 GGUF model loader implementation for image to text.
"""
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
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


def _patch_transformers_qwen2vl_gguf():
    """Monkey-patch transformers to add qwen2vl GGUF architecture support.

    transformers 5.x does not support the qwen2vl GGUF architecture. This patch
    registers the architecture, reuses qwen2's config field mapping, uses the base
    TensorProcessor (which skips unmapped visual encoder tensors gracefully), and
    fixes get_gguf_hf_weights_map to handle the nested Qwen2VLConfig structure
    (num_hidden_layers lives in text_config).
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        TensorProcessor,
    )

    if "qwen2vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen2vl"] = GGUF_TO_TRANSFORMERS_MAPPING[
        "config"
    ]["qwen2"].copy()

    # Use base TensorProcessor: Qwen2VL has visual encoder tensors that the
    # standard qwen2 weight map cannot handle; the base processor skips them.
    TENSOR_PROCESSORS["qwen2vl"] = TensorProcessor

    orig_get_weights_map = gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = getattr(hf_model.config, "model_type", None)
        if model_type == "qwen2_vl":
            model_type = "qwen2vl"
        if num_layers is None:
            try:
                num_layers = hf_model.config.num_hidden_layers
            except AttributeError:
                try:
                    num_layers = hf_model.config.text_config.num_hidden_layers
                except AttributeError:
                    num_layers = 28
        return orig_get_weights_map(
            hf_model,
            processor,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_transformers_qwen2vl_gguf()


class ModelVariant(StrEnum):
    """Available Robo-Dopamine GRM 8B i1 GGUF model variants for image to text."""

    ROBO_DOPAMINE_GRM_8B_I1_GGUF = "8b_i1_gguf"


class ModelLoader(ForgeModel):
    """Robo-Dopamine GRM 8B i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.ROBO_DOPAMINE_GRM_8B_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Robo-Dopamine-GRM-8B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ROBO_DOPAMINE_GRM_8B_I1_GGUF

    GGUF_FILE = "Robo-Dopamine-GRM-8B.i1-Q4_K_M.gguf"

    # Processor source (the GGUF repo ships only quantized weights).
    _PROCESSOR_SOURCE = "tanhuajie2001/Robo-Dopamine-GRM-8B"

    # Canonical Qwen2-VL config for GGUF loading (qwen2vl arch not natively supported).
    _CONFIG_SOURCE = "Qwen/Qwen2-VL-7B-Instruct"

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Robo-Dopamine GRM 8B i1 GGUF",
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

        self.processor = AutoProcessor.from_pretrained(self._PROCESSOR_SOURCE)

        # qwen2vl GGUF architecture is not natively supported; load config from the
        # canonical Qwen2-VL-7B-Instruct model to ensure a compatible Qwen2VLConfig.
        base_config = AutoConfig.from_pretrained(self._CONFIG_SOURCE)
        model_kwargs["config"] = base_config
        model_kwargs |= kwargs

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(self._CONFIG_SOURCE)
        return self.config
