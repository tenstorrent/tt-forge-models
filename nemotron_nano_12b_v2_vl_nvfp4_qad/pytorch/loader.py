# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NVIDIA Nemotron Nano 12B v2 VL NVFP4-QAD model loader implementation for image to text.
"""

import os
from contextlib import contextmanager, nullcontext
from typing import Optional

import torch
import torch.cuda
import torch.distributed
from PIL import Image
from transformers import AutoModel, AutoProcessor
from transformers.modeling_utils import PreTrainedModel

# The NemotronH Mamba2 naive fallback wraps all block ops in torch.cuda.stream()
# purely to avoid multi-GPU NaN issues. In non-CUDA environments the context manager
# itself raises an AssertionError before running any GPU code. Replace it with a no-op.
if not torch.cuda.is_available():
    torch.cuda.default_stream = lambda device=None: None
    torch.cuda.stream = lambda stream: nullcontext()


@contextmanager
def _patch_tied_keys_for_missing_post_init():
    """Handle trust_remote_code models that don't call post_init(), leaving all_tied_weights_keys unset."""
    orig = PreTrainedModel._adjust_tied_keys_with_tied_pointers

    def patched(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(
                all_submodels=False
            )
        return orig(self, *args, **kwargs)

    PreTrainedModel._adjust_tied_keys_with_tied_pointers = patched
    try:
        yield
    finally:
        PreTrainedModel._adjust_tied_keys_with_tied_pointers = orig


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
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        with _patch_tied_keys_for_missing_post_init():
            model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)

        # vision_model.to(dtype) in __init__ corrupts the RADIO encoder's summary_idxs buffer
        # (an int64 index tensor that gets reinterpretted during dtype conversion).
        # Recompute the correct values from the model's vision config.
        if (
            hasattr(model, "vision_model")
            and hasattr(model.vision_model, "radio_model")
            and model.vision_model.radio_model is not None
        ):
            rm = model.vision_model.radio_model
            if rm.summary_idxs is not None:
                teachers = model.config.vision_config.args.get("teachers", [])
                correct_idxs = torch.tensor(
                    [i for i, t in enumerate(teachers) if t.get("use_summary", True)],
                    dtype=torch.long,
                )
                rm.register_buffer("summary_idxs", correct_idxs)

        # model.forward() calls torch.distributed.get_rank() unconditionally; initialize
        # a single-process group so it doesn't raise when running without distributed setup.
        if not torch.distributed.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
            torch.distributed.init_process_group(backend="gloo", world_size=1, rank=0)

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

        # num_patches: one entry per image indicating how many tiles the image was split into
        num_patches = inputs.pop("num_patches", None)

        # pixel_values must match the model dtype (vision encoder is cast to language model dtype)
        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        # image_flags marks which tiles are valid; shape must match pixel_values batch dim
        total_tiles = (
            int(num_patches.sum())
            if num_patches is not None
            else inputs["pixel_values"].shape[0]
        )
        inputs["image_flags"] = torch.ones(total_tiles, 1, dtype=torch.long)

        # forward() tries outputs.past_key_values but the LM returns cache_params;
        # return_dict=False makes the model return a plain tuple instead.
        inputs["return_dict"] = False

        if batch_size > 1:
            for key, value in inputs.items():
                if key != "image_flags" and hasattr(value, "repeat_interleave"):
                    inputs[key] = value.repeat_interleave(batch_size, dim=0)
            inputs["image_flags"] = inputs["image_flags"].repeat(batch_size, 1)

        return inputs
