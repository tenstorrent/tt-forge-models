# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ASID Captioner 7B i1 GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen2VLForConditionalGeneration,
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


def _patch_transformers_qwen2vl_gguf():
    """Monkey-patch transformers to add qwen2vl GGUF architecture support.

    transformers 5.x does not support the qwen2vl GGUF architecture. This patch
    registers the architecture, reuses qwen2's config field mapping, and uses the
    base TensorProcessor (which skips unmapped visual encoder tensors gracefully).
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


_patch_transformers_qwen2vl_gguf()


class ModelVariant(StrEnum):
    """Available ASID Captioner 7B i1 GGUF model variants for image to text."""

    ASID_CAPTIONER_7B_I1_Q4_K_M_GGUF = "7b_i1_Q4_K_M_gguf"


class ModelLoader(ForgeModel):
    """ASID Captioner 7B i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.ASID_CAPTIONER_7B_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/ASID-Captioner-7B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ASID_CAPTIONER_7B_I1_Q4_K_M_GGUF

    GGUF_FILE = "ASID-Captioner-7B.i1-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ASID Captioner 7B i1 GGUF",
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

        # GGUF repos do not ship a processor; use the base model with use_fast=False
        # to avoid FutureWarning about Qwen2VLImageProcessor fast processor default.
        self.processor = AutoProcessor.from_pretrained(
            "AudioVisual-Caption/ASID-Captioner-7B", use_fast=False
        )

        # The GGUF architecture is qwen2vl; load config from the canonical
        # Qwen2-VL-7B model because the AudioVisual-Caption repo now returns a
        # Qwen2_5OmniConfig which is incompatible with Qwen2VLForConditionalGeneration.
        base_config = AutoConfig.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        # Set flat num_hidden_layers so GGUF weight-map lookup can find it without
        # diving into text_config (transformers' get_gguf_hf_weights_map expects it
        # at the top level of config).
        base_config.num_hidden_layers = base_config.text_config.num_hidden_layers

        # Use the GGUF arch name so get_gguf_hf_weights_map finds it in
        # gguf-py's MODEL_ARCH_NAMES ("qwen2vl" maps to MODEL_ARCH.QWEN2VL).
        base_config.model_type = "qwen2vl"

        model_kwargs["config"] = base_config
        model_kwargs |= kwargs

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        # Use text-only inputs: the GGUF checkpoint contains only LM weights;
        # the visual encoder is randomly initialized and not runnable on TT silicon.
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Describe a sunset over the ocean."}],
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
