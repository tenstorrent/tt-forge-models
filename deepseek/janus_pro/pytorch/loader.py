# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek Janus-Pro model loader implementation for multimodal understanding.
"""

from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

# Import janus to register the "multi_modality" architecture type with
# AutoConfig / AutoModelForCausalLM.  The model checkpoint itself has no
# Python files, so trust_remote_code alone cannot resolve the class.
from janus.models import MultiModalityCausalLM, VLChatProcessor  # noqa: F401
from janus.models import siglip_vit as _janus_siglip_vit

# Fix: janus siglip_vit calls torch.linspace(...).item() during __init__.
# Transformers 5.x always initialises models inside torch.device("meta"), so
# every tensor created there is a meta tensor on which .item() is forbidden.
# Redirect linspace to CPU so the stochastic-depth rates are real floats.
_orig_linspace = torch.linspace


def _cpu_linspace(*args, **kwargs):
    kwargs["device"] = "cpu"
    return _orig_linspace(*args, **kwargs)


_janus_siglip_vit.torch.linspace = _cpu_linspace

# Fix: transformers 5.x _finalize_model_loading calls
# _adjust_tied_keys_with_tied_pointers which accesses self.all_tied_weights_keys.
# MultiModalityConfig has tie_word_embeddings unset, so
# get_expanded_tied_weights_keys returns {} without ever setting the attribute
# on the outer model.  Guard against the missing attribute.
_orig_adjust_tied = PreTrainedModel._adjust_tied_keys_with_tied_pointers


def _patched_adjust_tied(self, missing_keys):
    if not hasattr(self, "all_tied_weights_keys"):
        self.all_tied_weights_keys = {}
    _orig_adjust_tied(self, missing_keys)


PreTrainedModel._adjust_tied_keys_with_tied_pointers = _patched_adjust_tied

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ....tools.utils import cast_input_to_type, get_file


class JanusProForwardWrapper(nn.Module):
    """Wraps MultiModalityCausalLM to expose a flat forward() interface.

    prepare_inputs_embeds fuses the vision tokens into the sequence, then
    the language model runs the combined embedding through its decoder.
    """

    def __init__(self, model: MultiModalityCausalLM):
        super().__init__()
        self.janus = model

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask,
        images_seq_mask,
        images_emb_mask,
    ):
        inputs_embeds = self.janus.prepare_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask,
        )
        return self.janus.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )


class ModelVariant(StrEnum):
    """Available DeepSeek Janus-Pro model variants."""

    JANUS_PRO_1B = "Janus_Pro_1B"


class ModelLoader(ForgeModel):
    """DeepSeek Janus-Pro model loader for multimodal understanding."""

    _VARIANTS = {
        ModelVariant.JANUS_PRO_1B: ModelConfig(
            pretrained_model_name="deepseek-ai/Janus-Pro-1B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.JANUS_PRO_1B

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Janus-Pro model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="JanusPro",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        # use_fast=False avoids the transformers 5.x FutureWarning that
        # VLMImageProcessor is now loaded as a fast processor by default.
        self.processor = VLChatProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            use_fast=False,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Janus-Pro model instance."""
        model_name = self._variant_config.pretrained_model_name

        # Fix: the model config bakes in "_attn_implementation": "flash_attention_2"
        # which requires the flash_attn package (GPU-only).  Override to "eager".
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.language_config._attn_implementation = "eager"

        model_kwargs = {"trust_remote_code": True, "config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        janus_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        janus_model.eval()

        if self.processor is None:
            self._load_processor()

        return JanusProForwardWrapper(janus_model)

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for Janus-Pro."""
        if self.processor is None:
            self._load_processor()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file).convert("RGB")

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{self.sample_text}",
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        output = self.processor(
            conversations=conversation,
            images=[image],
            force_batchify=True,
        )

        inputs = {
            "input_ids": output.input_ids,
            "pixel_values": output.pixel_values,
            "attention_mask": output.attention_mask,
            "images_seq_mask": output.images_seq_mask,
            "images_emb_mask": output.images_emb_mask,
        }

        if dtype_override:
            inputs = {k: cast_input_to_type(v, dtype_override) for k, v in inputs.items()}

        return inputs
