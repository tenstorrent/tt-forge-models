# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cambrian-S model loader implementation for multimodal visual question answering.
"""

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from transformers.cache_utils import DynamicCache
from transformers.configuration_utils import PretrainedConfig
from typing import Optional

try:
    from cambrian import CambrianQwenConfig, CambrianQwenForCausalLM

    AutoConfig.register("cambrian_qwen", CambrianQwenConfig)
    AutoModelForCausalLM.register(CambrianQwenConfig, CambrianQwenForCausalLM)

    # CambrianQwenForCausalLM.__init__ sets config.rope_scaling = None which in
    # transformers 5.x clobbers rope_parameters via the property setter. Patch
    # the setter to be a no-op when value is None so rope_parameters is preserved.
    _orig_rope_scaling_setter = PretrainedConfig.rope_scaling.fset

    def _patched_rope_scaling_setter(self, value):
        if value is not None:
            _orig_rope_scaling_setter(self, value)

    PretrainedConfig.rope_scaling = PretrainedConfig.rope_scaling.setter(
        _patched_rope_scaling_setter
    )

    # cambrian-s uses DynamicCache.get_usable_length which was renamed to
    # get_seq_length in transformers 5.x. Add a backward-compatible alias.
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = DynamicCache.get_seq_length

    # In transformers 5.x, CambrianQwenModel.forward was written for transformers 4.x
    # and is incompatible (_attn_implementation moved to config, position_embeddings
    # now required in decoder layer forward, _prepare_4d_causal_attention_mask removed).
    # Patch CambrianQwenModel to use Qwen2Model.forward which handles the new API.
    from cambrian.model.language_model.cambrian_qwen2 import CambrianQwenModel
    from transformers import Qwen2Model

    CambrianQwenModel.forward = Qwen2Model.forward
except ImportError:
    pass

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.utils import get_file, cast_input_to_type


class ModelVariant(StrEnum):
    """Available Cambrian-S model variants."""

    CAMBRIAN_S_7B_S3 = "S_7B_S3"
    CAMBRIAN_S_3B = "S_3B"


class ModelLoader(ForgeModel):
    """Cambrian-S model loader for multimodal visual question answering."""

    _VARIANTS = {
        ModelVariant.CAMBRIAN_S_7B_S3: ModelConfig(
            pretrained_model_name="nyu-visionx/Cambrian-S-7B-S3",
        ),
        ModelVariant.CAMBRIAN_S_3B: ModelConfig(
            pretrained_model_name="nyu-visionx/Cambrian-S-3B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CAMBRIAN_S_7B_S3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Cambrian-S",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Cambrian-S model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for Cambrian-S."""
        if self.processor is None:
            self._load_processor()

        conversation = [
            {
                "role": "user",
                "content": "What is shown in this image?",
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        inputs = self.processor(text=text_prompt, return_tensors="pt")

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return dict(inputs)

    def decode_output(self, outputs, input_length=None):
        """Decode model outputs into human-readable text."""
        if self.processor is None:
            self._load_processor()

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return self.processor.decode(next_token_id)
