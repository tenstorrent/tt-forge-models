# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NVIDIA Nemotron Nano 12B v2 VL NVFP4-QAD model loader implementation for image to text.
"""

from transformers import AutoModel, AutoProcessor
from transformers.modeling_utils import PreTrainedModel
from PIL import Image
from typing import Optional

from ...tools.utils import get_file
from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


# Fix: NemotronH_Nano_VL_V2 doesn't call post_init() so all_tied_weights_keys is never set.
# Patch the method that first accesses it to initialize the attribute if missing.
_orig_adjust_tied = PreTrainedModel._adjust_tied_keys_with_tied_pointers


def _safe_adjust_tied(self, missing_keys):
    if not hasattr(self, "all_tied_weights_keys"):
        self.all_tied_weights_keys = {}
    _orig_adjust_tied(self, missing_keys)


PreTrainedModel._adjust_tied_keys_with_tied_pointers = _safe_adjust_tied


class ModelVariant(StrEnum):
    """Available NVIDIA Nemotron Nano 12B v2 VL NVFP4-QAD model variants."""

    NEMOTRON_NANO_12B_V2_VL_NVFP4_QAD = "12b_v2_vl_nvfp4_qad"


class ModelLoader(ForgeModel):
    """NVIDIA Nemotron Nano 12B v2 VL NVFP4-QAD model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_NANO_12B_V2_VL_NVFP4_QAD: LLMModelConfig(
            pretrained_model_name="nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_NANO_12B_V2_VL_NVFP4_QAD

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="nemotron_nano_12b_v2_vl_nvfp4_qad",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Nemotron Nano 12B v2 VL NVFP4-QAD model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Nemotron Nano 12B v2 VL NVFP4-QAD model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
            "ignore_mismatched_sizes": True,
        }
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Nemotron Nano 12B v2 VL NVFP4-QAD model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        inputs = self.processor(
            images=image,
            text="<image>\nDescribe this image.",
            return_tensors="pt",
        )

        if batch_size > 1:
            for key, value in inputs.items():
                if hasattr(value, "repeat_interleave"):
                    inputs[key] = value.repeat_interleave(batch_size, dim=0)

        return inputs
