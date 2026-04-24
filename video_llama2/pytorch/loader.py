# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VideoLLaMA2 model loader implementation for multimodal video understanding.
"""

from typing import Optional

import transformers

# videollama2 package references the removed transformers.TRANSFORMERS_CACHE constant
if not hasattr(transformers, "TRANSFORMERS_CACHE"):
    from huggingface_hub.constants import HF_HUB_CACHE

    transformers.TRANSFORMERS_CACHE = HF_HUB_CACHE

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from videollama2.model.videollama2_qwen2 import (
    Videollama2Qwen2Config,
    Videollama2Qwen2ForCausalLM,
)

AutoConfig.register("videollama2_qwen2", Videollama2Qwen2Config)
AutoModelForCausalLM.register(Videollama2Qwen2Config, Videollama2Qwen2ForCausalLM)

# encoder.py hardcodes flash_attention_2 for CLIP/SigLIP vision towers; patch to eager
import videollama2.model.encoder as _vl2_encoder
from transformers import (
    CLIPVisionConfig,
    CLIPVisionModel,
    SiglipImageProcessor,
    SiglipVisionConfig,
    SiglipVisionModel,
)


def _patched_siglip_init(self, vision_tower, args, load_pretrained=False):
    super(_vl2_encoder.SiglipVisionTower, self).__init__()
    self.vision_tower_name = vision_tower
    self.select_layer = args.mm_vision_select_layer
    self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
    self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
    config = SiglipVisionConfig.from_pretrained(self.vision_tower_name)
    config._attn_implementation = "eager"
    if not load_pretrained:
        self.vision_tower = SiglipVisionModel(config=config)
    else:
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)


_vl2_encoder.SiglipVisionTower.__init__ = _patched_siglip_init

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
from ...tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available VideoLLaMA2 model variants."""

    V2_1_7B_16F = "v2_1_7b_16f"


class ModelLoader(ForgeModel):
    """VideoLLaMA2 model loader for multimodal video understanding."""

    _VARIANTS = {
        ModelVariant.V2_1_7B_16F: ModelConfig(
            pretrained_model_name="DAMO-NLP-SG/VideoLLaMA2.1-7B-16F",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2_1_7B_16F

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize VideoLLaMA2 model loader."""
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="VideoLLaMA2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the VideoLLaMA2 model instance."""
        model_name = self._variant_config.pretrained_model_name
        kwargs.setdefault("trust_remote_code", True)
        kwargs.setdefault("attn_implementation", "eager")
        model = AutoModelForCausalLM.from_pretrained(str(model_name), **kwargs)
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.tokenizer is None:
            self._load_tokenizer()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for VideoLLaMA2."""
        if self.tokenizer is None:
            self._load_tokenizer()

        text_prompt = "<video>\nWhat is shown in this video?"

        inputs = self.tokenizer(text_prompt, return_tensors="pt")

        if dtype_override:
            inputs = {
                k: cast_input_to_type(v, dtype_override) for k, v in inputs.items()
            }

        return dict(inputs)
