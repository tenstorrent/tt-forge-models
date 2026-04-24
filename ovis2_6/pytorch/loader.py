# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ovis2.6 model loader implementation for multimodal visual question answering.
"""

import inspect
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from PIL import Image
from typing import Optional

# Ovis2.6 custom model code was written for transformers 4.x. In transformers 5.x:
# 1. is_parallelizable was removed from all models
# 2. all_tied_weights_keys must be set (via post_init), but custom models skip post_init
# 3. tie_weights() gained new kwargs (missing_keys, recompute_mapping)
# Patch _finalize_model_loading to lazily fix both (2) and (3) on any affected model.
if not hasattr(nn.Module, "is_parallelizable"):
    nn.Module.is_parallelizable = False

if hasattr(PreTrainedModel, "_finalize_model_loading"):
    _orig_finalize = PreTrainedModel._finalize_model_loading

    @staticmethod
    def _robust_finalize(model, load_config, loading_info):
        if not hasattr(model, "all_tied_weights_keys"):
            model.all_tied_weights_keys = model.get_expanded_tied_weights_keys(
                all_submodels=True
            )
        orig_tw = type(model).tie_weights
        if orig_tw is not PreTrainedModel.tie_weights:
            sig = inspect.signature(orig_tw)
            params = sig.parameters
            if "missing_keys" not in params and "kwargs" not in params:

                def _compat_tie_weights(
                    self, missing_keys=None, recompute_mapping=True, **kw
                ):
                    orig_tw(self)

                type(model).tie_weights = _compat_tie_weights
        return _orig_finalize(model, load_config, loading_info)

    PreTrainedModel._finalize_model_loading = _robust_finalize

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

        # transformers 5.x fast image processor doesn't support return_tensors="np";
        # the model's VisualTokenizer uses it, so swap in the slow processor.
        if hasattr(model, "visual_tokenizer") and hasattr(
            model.visual_tokenizer, "image_processor"
        ):
            model.visual_tokenizer.image_processor = AutoImageProcessor.from_pretrained(
                pretrained_model_name, do_center_crop=False, use_fast=False
            )

        # In transformers 5.x the loading process moves non-persistent buffers
        # (which are absent from the checkpoint) from meta device to an
        # uninitialised real tensor, corrupting their values.  Recompute
        # indicator_token_indices from the config so the VTE lookup is valid.
        if hasattr(model, "indicator_token_indices") and hasattr(
            model.config, "visual_vocab_size"
        ):
            n = model.indicator_token_indices.shape[0]
            correct = torch.arange(
                model.config.visual_vocab_size - n,
                model.config.visual_vocab_size,
                dtype=torch.long,
            )
            model.register_buffer("indicator_token_indices", correct, persistent=False)

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

        pad_token_id = self.model.text_tokenizer.pad_token_id
        attention_mask = torch.ne(input_ids, pad_token_id)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
        }

        if dtype_override is not None:
            if pixel_values is not None:
                inputs["pixel_values"] = pixel_values.to(dtype_override)

        if batch_size > 1:
            inputs["input_ids"] = input_ids.repeat_interleave(batch_size, dim=0)
            inputs["attention_mask"] = attention_mask.repeat_interleave(
                batch_size, dim=0
            )
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
