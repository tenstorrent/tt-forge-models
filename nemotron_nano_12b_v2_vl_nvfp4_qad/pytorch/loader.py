# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NVIDIA Nemotron Nano 12B v2 VL NVFP4-QAD model loader implementation for image to text.
"""

import contextlib
import sys
import torch
import torch.distributed as dist
from transformers import AutoModel, AutoProcessor, PreTrainedModel
from PIL import Image
from typing import Optional

# NemotronH blocks use torch.cuda.stream for multi-GPU correctness (a no-op for single-device).
# Patch to work without CUDA (this environment uses XLA/TT backend, not CUDA).
if not torch.cuda.is_available():

    @contextlib.contextmanager
    def _null_cuda_stream(stream):
        yield

    torch.cuda.stream = _null_cuda_stream
    torch.cuda.default_stream = lambda device=None: None

# NemotronH_Nano_VL_V2.forward calls torch.distributed.get_rank() for a debug print but
# distributed may not be initialized in single-device inference. Patch to return 0 safely.
_original_get_rank = dist.get_rank


def _safe_get_rank(*args, **kwargs):
    try:
        return _original_get_rank(*args, **kwargs)
    except (ValueError, RuntimeError):
        return 0


dist.get_rank = _safe_get_rank

# NemotronH_Nano_VL_V2.__init__ does not call self.post_init(), so all_tied_weights_keys
# is never set. Patch _adjust_tied_keys_with_tied_pointers to initialize it if missing.
_original_adjust_tied = PreTrainedModel._adjust_tied_keys_with_tied_pointers


def _safe_adjust_tied(self, *args, **kwargs):
    if not hasattr(self, "all_tied_weights_keys"):
        self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(
            all_submodels=False
        )
    return _original_adjust_tied(self, *args, **kwargs)


PreTrainedModel._adjust_tied_keys_with_tied_pointers = _safe_adjust_tied

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

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)

        # NemotronHCausalLMOutput uses 'cache_params' instead of standard 'past_key_values'.
        # The VL model's forward accesses outputs.past_key_values on the language model output,
        # so patch NemotronHCausalLMOutput to expose past_key_values as an alias.
        for mod in sys.modules.values():
            if hasattr(mod, "NemotronHCausalLMOutput"):
                cls = mod.NemotronHCausalLMOutput
                if not hasattr(cls, "past_key_values"):
                    cls.past_key_values = property(lambda self: self.cache_params)
                break

        # summary_idxs is a buffer computed from config during __init__, but gets corrupted
        # when the model is initialized on meta device then moved to CPU without re-initialization.
        # Re-compute it from the RADIO vision config to restore correct values.
        radio_model = model.vision_model.radio_model
        if (
            hasattr(radio_model, "summary_idxs")
            and radio_model.summary_idxs is not None
        ):
            teachers = model.vision_model.config.args.get("teachers", [])
            correct_idxs = torch.tensor(
                [i for i, t in enumerate(teachers) if t.get("use_summary", True)],
                dtype=torch.int64,
            )
            with torch.no_grad():
                radio_model.summary_idxs.copy_(correct_idxs)

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

        inputs.pop("num_patches", None)
        inputs["image_flags"] = torch.ones(
            (inputs["pixel_values"].shape[0], 1), dtype=torch.long
        )

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        if batch_size > 1:
            for key, value in inputs.items():
                if hasattr(value, "repeat_interleave"):
                    inputs[key] = value.repeat_interleave(batch_size, dim=0)

        return inputs
