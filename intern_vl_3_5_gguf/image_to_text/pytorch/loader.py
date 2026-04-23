# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InternVL3.5 GGUF model loader implementation for image to text.
"""

import os

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
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
            pretrained_model_name="OpenGVLab/InternVL3_5-4B-HF",
            max_length=128,
        ),
        ModelVariant.INTERN_VL3_5_4B_Q8_0: LLMModelConfig(
            pretrained_model_name="OpenGVLab/InternVL3_5-4B-HF",
            max_length=128,
        ),
        ModelVariant.INTERN_VL3_5_14B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="OpenGVLab/InternVL3_5-14B-HF",
            max_length=128,
        ),
        ModelVariant.INTERN_VL3_5_14B_Q8_0: LLMModelConfig(
            pretrained_model_name="OpenGVLab/InternVL3_5-14B-HF",
            max_length=128,
        ),
    }

    # bartowski GGUF repos lack config.json; download weights separately
    _GGUF_REPOS = {
        ModelVariant.INTERN_VL3_5_4B_Q4_K_M: "bartowski/OpenGVLab_InternVL3_5-4B-GGUF",
        ModelVariant.INTERN_VL3_5_4B_Q8_0: "bartowski/OpenGVLab_InternVL3_5-4B-GGUF",
        ModelVariant.INTERN_VL3_5_14B_Q4_K_M: "bartowski/OpenGVLab_InternVL3_5-14B-GGUF",
        ModelVariant.INTERN_VL3_5_14B_Q8_0: "bartowski/OpenGVLab_InternVL3_5-14B-GGUF",
    }

    _GGUF_FILES = {
        ModelVariant.INTERN_VL3_5_4B_Q4_K_M: "OpenGVLab_InternVL3_5-4B-Q4_K_M.gguf",
        ModelVariant.INTERN_VL3_5_4B_Q8_0: "OpenGVLab_InternVL3_5-4B-Q8_0.gguf",
        ModelVariant.INTERN_VL3_5_14B_Q4_K_M: "OpenGVLab_InternVL3_5-14B-Q4_K_M.gguf",
        ModelVariant.INTERN_VL3_5_14B_Q8_0: "OpenGVLab_InternVL3_5-14B-Q8_0.gguf",
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

    @staticmethod
    def _load_gguf_into_language_model(model, gguf_path, dtype=None):
        """Load qwen3 GGUF weights into InternVL's language_model submodule.

        The bartowski GGUF contains only the Qwen3 text backbone; transformers
        GGUF loading cannot handle InternVLConfig natively, so we manually map
        and load the language model weights with a 'language_model.' prefix.
        """
        from gguf import GGUFReader, dequantize
        from transformers.modeling_gguf_pytorch_utils import (
            TensorProcessor,
            get_gguf_hf_weights_map,
        )

        reader = GGUFReader(gguf_path)
        processor = TensorProcessor()

        # Map GGUF qwen3 tensor names → InternVL model parameter names
        # by using the language_model submodule with the qualifying prefix.
        tensor_key_mapping = get_gguf_hf_weights_map(
            model.language_model,
            processor,
            qual_name="language_model.",
        )

        state_dict = {}
        for tensor in reader.tensors:
            name = tensor.name
            weights = dequantize(tensor.data, tensor.tensor_type)
            if name not in tensor_key_mapping:
                continue
            hf_name = tensor_key_mapping[name]
            t = torch.from_numpy(np.copy(weights))
            if dtype is not None:
                t = t.to(dtype)
            state_dict[hf_name] = t

        # strict=False: vision encoder weights are absent from the GGUF
        model.load_state_dict(state_dict, strict=False)

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_path = hf_hub_download(
            repo_id=self._GGUF_REPOS[self._variant],
            filename=self._GGUF_FILES[self._variant],
        )

        config = AutoConfig.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            model = AutoModelForImageTextToText.from_config(config, **model_kwargs)
        else:
            model = AutoModelForImageTextToText.from_config(config, **model_kwargs)
            self._load_gguf_into_language_model(model, gguf_path, dtype=dtype_override)

        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
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
