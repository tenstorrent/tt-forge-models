# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MediX R1 30B GGUF model loader implementation for image to text.
"""

from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor, AutoConfig
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

    transformers 5.x does not support the qwen3vlmoe GGUF architecture used by
    Qwen3-VL-MoE models. This patch registers the architecture, adds the config
    field mapping (reusing qwen3_moe's text config mapping), selects the right
    tensor processor, and fixes get_gguf_hf_weights_map to handle the nested
    Qwen3VLMoeConfig structure (num_hidden_layers lives in text_config).
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        Qwen2MoeTensorProcessor,
    )

    if "qwen3vlmoe" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vlmoe")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vlmoe"] = GGUF_TO_TRANSFORMERS_MAPPING[
        "config"
    ]["qwen3_moe"].copy()

    TENSOR_PROCESSORS["qwen3vlmoe"] = Qwen2MoeTensorProcessor

    orig_get_weights_map = gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            mt = getattr(hf_model.config, "model_type", None)
            if mt == "qwen3_vl_moe":
                model_type = "qwen3vlmoe"
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
        model_kwargs["ignore_mismatched_sizes"] = True
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
        )

        # Load config from base model to avoid GGUF config parsing which does
        # not support the nested Qwen3VLMoeConfig structure.
        base_config = AutoConfig.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")
        model_kwargs["config"] = base_config

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
