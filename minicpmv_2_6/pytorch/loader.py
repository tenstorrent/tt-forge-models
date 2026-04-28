# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM-V 2.6 model loader implementation for multimodal visual question answering
"""

from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from transformers.integrations.tensor_parallel import ALL_PARALLEL_STYLES

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

# Fix parallel styles issue for torch 2.7.0+ compatibility
if ALL_PARALLEL_STYLES is None:
    import transformers.modeling_utils as mu

    mu.ALL_PARALLEL_STYLES = ["rowwise", "colwise", "headwise"]

def _apply_resampler_compat_patch():
    # Apply once: Resampler._initialize_weights shim for torch 2.7.0+.
    # Deferred to call time (not import time) so it doesn't mutate
    # nn.Module.__getattr__ before other models are traced by torch.compile,
    # which would cause SpeculationLog divergence across compilation passes.
    if getattr(nn.Module.__getattr__, "_resampler_compat", False):
        return
    _orig = nn.Module.__getattr__

    def _patched(self, name):
        if name == "_initialize_weights" and self.__class__.__name__ == "Resampler":

            def _initialize_weights(module_self):
                if hasattr(module_self, "_init_weights"):
                    module_self._init_weights(module_self)

            return _initialize_weights
        return _orig(self, name)

    _patched._resampler_compat = True
    nn.Module.__getattr__ = _patched


class ModelVariant(StrEnum):
    """Available MiniCPM-V 2.6 model variants."""

    TINY_RANDOM = "Tiny_Random"


class ModelLoader(ForgeModel):
    """MiniCPM-V 2.6 model loader for multimodal visual question answering."""

    _VARIANTS = {
        ModelVariant.TINY_RANDOM: ModelConfig(
            pretrained_model_name="optimum-intel-internal-testing/tiny-random-minicpmv-2_6",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MiniCPM-V 2.6",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        """Load and return the MiniCPM-V 2.6 model instance."""
        _apply_resampler_compat_patch()
        config = self._variant_config

        self.model = AutoModel.from_pretrained(
            config.pretrained_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            **kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model_name, trust_remote_code=True
        )

        self.model.eval()
        return self.model

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for the model."""
        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        question = "Describe this image in detail."

        return {"question": question, "image": image}

    def predict(self, inputs=None, **kwargs):
        """Run inference on the loaded model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_model() first.")

        if inputs is None:
            inputs = self.load_inputs()

        question = inputs.get("question", "Describe this image.")
        image = inputs.get("image")

        if image is None:
            raise ValueError("Image input is required for MiniCPM-V inference")

        msgs = [{"role": "user", "content": question}]

        with torch.no_grad():
            result = self.model.chat(
                image=image,
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=False,
                max_new_tokens=128,
                **kwargs,
            )

        return {"question": question, "response": result}

    @classmethod
    def decode_output(cls, outputs, **kwargs):
        """Decode model outputs into human-readable format."""
        return {
            "question": outputs.get("question", ""),
            "answer": outputs.get("response", ""),
        }
