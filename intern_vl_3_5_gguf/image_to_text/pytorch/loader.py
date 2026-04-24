# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InternVL3.5 GGUF model loader implementation for image to text.
"""

import inspect

from transformers import (
    AutoConfig,
    AutoProcessor,
    InternVLForConditionalGeneration,
    Qwen3ForCausalLM,
)
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tok
import transformers.tokenization_utils_tokenizers as _tok_utils
from typing import Optional


def _make_model_to_load_compat(fn):
    """Wrap fn to accept the model_to_load kwarg added in transformers 5.x.

    Some other loaders patch load_gguf_checkpoint with a version that lacks
    this kwarg, causing a TypeError when transformers 5.x passes it.
    """
    if "model_to_load" in inspect.signature(fn).parameters:
        return fn

    def _wrapper(gguf_path, return_tensors=False, model_to_load=None):
        return fn(gguf_path, return_tensors=return_tensors)

    return _wrapper


_gguf_utils.load_gguf_checkpoint = _make_model_to_load_compat(
    _gguf_utils.load_gguf_checkpoint
)
for _mod in (_config_utils, _auto_tok, _tok_utils):
    _mod.load_gguf_checkpoint = _gguf_utils.load_gguf_checkpoint

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
    """Available InternVL3.5 GGUF model variants for image to text."""

    INTERN_VL3_5_4B_Q4_K_M = "4b_q4_k_m"
    INTERN_VL3_5_4B_Q8_0 = "4b_q8_0"
    INTERN_VL3_5_14B_Q4_K_M = "14b_q4_k_m"
    INTERN_VL3_5_14B_Q8_0 = "14b_q8_0"


class ModelLoader(ForgeModel):
    """InternVL3.5 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.INTERN_VL3_5_4B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="bartowski/OpenGVLab_InternVL3_5-4B-GGUF",
            max_length=128,
        ),
        ModelVariant.INTERN_VL3_5_4B_Q8_0: LLMModelConfig(
            pretrained_model_name="bartowski/OpenGVLab_InternVL3_5-4B-GGUF",
            max_length=128,
        ),
        ModelVariant.INTERN_VL3_5_14B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="bartowski/OpenGVLab_InternVL3_5-14B-GGUF",
            max_length=128,
        ),
        ModelVariant.INTERN_VL3_5_14B_Q8_0: LLMModelConfig(
            pretrained_model_name="bartowski/OpenGVLab_InternVL3_5-14B-GGUF",
            max_length=128,
        ),
    }

    _GGUF_FILES = {
        ModelVariant.INTERN_VL3_5_4B_Q4_K_M: "OpenGVLab_InternVL3_5-4B-Q4_K_M.gguf",
        ModelVariant.INTERN_VL3_5_4B_Q8_0: "OpenGVLab_InternVL3_5-4B-Q8_0.gguf",
        ModelVariant.INTERN_VL3_5_14B_Q4_K_M: "OpenGVLab_InternVL3_5-14B-Q4_K_M.gguf",
        ModelVariant.INTERN_VL3_5_14B_Q8_0: "OpenGVLab_InternVL3_5-14B-Q8_0.gguf",
    }

    _HF_PROCESSORS = {
        ModelVariant.INTERN_VL3_5_4B_Q4_K_M: "OpenGVLab/InternVL3_5-4B-HF",
        ModelVariant.INTERN_VL3_5_4B_Q8_0: "OpenGVLab/InternVL3_5-4B-HF",
        ModelVariant.INTERN_VL3_5_14B_Q4_K_M: "OpenGVLab/InternVL3_5-14B-HF",
        ModelVariant.INTERN_VL3_5_14B_Q8_0: "OpenGVLab/InternVL3_5-14B-HF",
    }

    # HF model names for loading configs (GGUF files only contain text backbone)
    _HF_CONFIGS = {
        ModelVariant.INTERN_VL3_5_4B_Q4_K_M: "OpenGVLab/InternVL3_5-4B-HF",
        ModelVariant.INTERN_VL3_5_4B_Q8_0: "OpenGVLab/InternVL3_5-4B-HF",
        ModelVariant.INTERN_VL3_5_14B_Q4_K_M: "OpenGVLab/InternVL3_5-14B-HF",
        ModelVariant.INTERN_VL3_5_14B_Q8_0: "OpenGVLab/InternVL3_5-14B-HF",
    }

    DEFAULT_VARIANT = ModelVariant.INTERN_VL3_5_4B_Q4_K_M

    sample_image = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="InternVL3.5 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]
        hf_config_name = self._HF_CONFIGS[self._variant]

        self.processor = AutoProcessor.from_pretrained(
            self._HF_PROCESSORS[self._variant],
            trust_remote_code=True,
        )

        # GGUF files only contain the quantized text backbone (Qwen3).
        # Load the full InternVL config from HF, initialize the model (vision
        # encoder gets random weights), then replace the language model with the
        # GGUF-loaded text backbone.
        config = AutoConfig.from_pretrained(hf_config_name)

        lm_kwargs = {}
        if dtype_override is not None:
            lm_kwargs["torch_dtype"] = dtype_override
        lm_kwargs["gguf_file"] = gguf_file
        lm_kwargs |= kwargs

        language_model = Qwen3ForCausalLM.from_pretrained(
            pretrained_model_name,
            config=config.text_config,
            **lm_kwargs,
        )

        model = InternVLForConditionalGeneration(config)
        model.language_model = language_model
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
                    {"type": "text", "text": "What is shown in this image?"},
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
