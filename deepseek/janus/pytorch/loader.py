# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek Janus model loader implementation for multimodal understanding.
"""

from typing import Optional

import torch
from PIL import Image
from janus.models import (
    MultiModalityCausalLM,
    VLChatProcessor,
)  # registers multi_modality in transformers
import janus.models.siglip_vit as _siglip_vit
import janus.models.modeling_vlm as _modeling_vlm
from transformers import AutoModelForCausalLM

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
from ....tools.utils import get_file

# transformers 5.x always uses torch.device("meta") during from_pretrained, but
# siglip_vit.VisionTransformer.__init__ calls torch.linspace().item() which
# fails on meta tensors.  Patch the init to run inside a CPU device context
# so tensor constructors produce concrete tensors instead of meta placeholders.
_orig_vit_init = _siglip_vit.VisionTransformer.__init__


def _cpu_vit_init(self, *args, **kwargs):
    with torch.device("cpu"):
        return _orig_vit_init(self, *args, **kwargs)


_siglip_vit.VisionTransformer.__init__ = _cpu_vit_init

# language_config in the Janus config.json requests flash_attention_2 which is
# not installed.  Patch MultiModalityCausalLM.__init__ to downgrade it to eager
# before LlamaForCausalLM is constructed, and call post_init() which janus
# omits but transformers 5.x needs to set all_tied_weights_keys.
_orig_mm_init = _modeling_vlm.MultiModalityCausalLM.__init__


def _patched_mm_init(self, config, *args, **kwargs):
    lang_cfg = getattr(config, "language_config", None)
    if lang_cfg is not None:
        attn = getattr(lang_cfg, "_attn_implementation_internal", None)
        if attn == "flash_attention_2":
            lang_cfg._attn_implementation_internal = "eager"
    _orig_mm_init(self, config, *args, **kwargs)
    # janus doesn't call post_init() but transformers 5.x needs all_tied_weights_keys set
    if not hasattr(self, "all_tied_weights_keys"):
        self.post_init()


_modeling_vlm.MultiModalityCausalLM.__init__ = _patched_mm_init


# MultiModalityCausalLM has no forward() — add one that chains
# prepare_inputs_embeds + language_model so the test framework can call model(**inputs).
def _mm_forward(
    self,
    input_ids: torch.LongTensor,
    pixel_values: torch.FloatTensor,
    images_seq_mask: torch.LongTensor,
    images_emb_mask: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    inputs_embeds = self.prepare_inputs_embeds(
        input_ids, pixel_values, images_seq_mask, images_emb_mask
    )
    return self.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
    )


_modeling_vlm.MultiModalityCausalLM.forward = _mm_forward


class ModelVariant(StrEnum):
    """Available DeepSeek Janus model variants."""

    JANUS_1_3B = "Janus_1_3B"


class ModelLoader(ForgeModel):
    """DeepSeek Janus model loader for multimodal understanding."""

    _VARIANTS = {
        ModelVariant.JANUS_1_3B: ModelConfig(
            pretrained_model_name="deepseek-ai/Janus-1.3B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.JANUS_1_3B

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Janus model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Janus",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = VLChatProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Janus model instance."""
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(str(model_name), **model_kwargs)
        model.eval()

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for Janus."""
        if self.processor is None:
            self._load_processor()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{self.sample_text}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        raw = self.processor(
            conversations=conversation, images=[image], force_batchify=True
        )

        # Convert to plain dict, keeping only tensor fields that forward() accepts
        inputs = {k: raw[k] for k in raw.keys() if torch.is_tensor(raw[k])}

        if dtype_override:
            for key in list(inputs.keys()):
                if inputs[key].is_floating_point():
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
