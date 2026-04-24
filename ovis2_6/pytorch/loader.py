# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ovis2.6 model loader implementation for multimodal visual question answering.
"""

import torch
from transformers import AutoImageProcessor, AutoModelForCausalLM, PreTrainedModel
from PIL import Image
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available Ovis2.6 model variants."""

    OVIS2_6_30B_A3B = "30B_A3B"


class ModelLoader(ForgeModel):
    """Ovis2.6 model loader for multimodal visual question answering tasks."""

    _VARIANTS = {
        ModelVariant.OVIS2_6_30B_A3B: ModelConfig(
            pretrained_model_name="AIDC-AI/Ovis2.6-30B-A3B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OVIS2_6_30B_A3B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Ovis2.6",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        # transformers 5.x removed is_parallelizable from PreTrainedModel;
        # modeling_ovis2_6.py accesses llm.is_parallelizable and vit.is_parallelizable.
        if not hasattr(PreTrainedModel, "is_parallelizable"):
            PreTrainedModel.is_parallelizable = False

        # transformers 5.x passes missing_keys/recompute_mapping to tie_weights(), but the
        # custom Ovis2.6 model defines tie_weights(self) without **kwargs.
        if not hasattr(PreTrainedModel, "_tie_weights_kwargs_patched"):
            _orig_finalize = PreTrainedModel._finalize_model_loading

            @staticmethod
            def _patched_finalize(model, load_config, loading_info):
                _orig_tie = model.tie_weights

                def _compat_tie_weights(**kw):
                    try:
                        return _orig_tie(**kw)
                    except TypeError:
                        return _orig_tie()

                model.tie_weights = _compat_tie_weights
                try:
                    return _orig_finalize(model, load_config, loading_info)
                finally:
                    model.tie_weights = _orig_tie

            PreTrainedModel._finalize_model_loading = _patched_finalize
            PreTrainedModel._tie_weights_kwargs_patched = True

        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
            "torch_dtype": torch.bfloat16,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        # transformers 5.x loads SiglipImageProcessor as a fast processor by default,
        # but modeling_ovis2_6.py calls preprocess(..., return_tensors="np") which
        # the fast processor does not support. Reload with use_fast=False.
        model.visual_tokenizer.image_processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name, do_center_crop=False, use_fast=False
        )

        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.model is None:
            raise RuntimeError(
                "Model must be loaded before inputs. Call load_model() first."
            )

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            }
        ]

        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=messages, add_generation_prompt=True
        )

        inputs = {
            "inputs": input_ids,
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
        }

        if dtype_override is not None:
            if pixel_values is not None:
                inputs["pixel_values"] = pixel_values.to(dtype_override)

        if batch_size > 1:
            inputs["inputs"] = input_ids.repeat_interleave(batch_size, dim=0)
            if pixel_values is not None:
                inputs["pixel_values"] = inputs["pixel_values"].repeat_interleave(
                    batch_size, dim=0
                )
            if grid_thws is not None:
                inputs["grid_thws"] = grid_thws.repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, input_length=None):
        if isinstance(outputs, str):
            return outputs

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return self.model.text_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return self.model.text_tokenizer.decode(next_token_id)
