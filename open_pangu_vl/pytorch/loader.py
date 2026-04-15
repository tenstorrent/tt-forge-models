# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
openPangu-VL model loader implementation for vision-language tasks.
"""
import sys
import types

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from typing import Optional


def _install_compat_stubs():
    """Install compatibility stubs for the openPangu-VL HuggingFace model code.

    The model code has two compatibility issues:
    1. Unconditionally imports torch_npu (Huawei Ascend NPU) - we stub it so the
       model loads without NPU hardware.
    2. Imports LossKwargs from transformers.utils which was removed in transformers 5.x -
       we provide a TypedDict shim.
    """
    # Stub torch_npu as a package with submodules
    if "torch_npu" not in sys.modules:
        torch_npu = types.ModuleType("torch_npu")
        torch_npu.__path__ = []
        torch_npu.npu_fusion_attention = None
        torch_npu.npu_fused_infer_attention_score = None
        sys.modules["torch_npu"] = torch_npu

        # torch_npu.contrib with transfer_to_npu
        contrib = types.ModuleType("torch_npu.contrib")
        contrib.transfer_to_npu = None
        sys.modules["torch_npu.contrib"] = contrib
        torch_npu.contrib = contrib

        # The model code checks torch.npu.get_device_name() for "910"
        npu_stub = types.ModuleType("torch.npu")
        npu_stub.get_device_name = lambda *args, **kwargs: "stub"
        torch.npu = npu_stub

    # Shim 'default' rope type removed in transformers 5.x
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" not in ROPE_INIT_FUNCTIONS:

        def _compute_default_rope_parameters(
            config, device=None, seq_len=None, layer_type=None
        ):
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
            head_dim = getattr(
                config, "head_dim", config.hidden_size // config.num_attention_heads
            )
            dim = int(head_dim * partial_rotary_factor)
            inv_freq = 1.0 / (
                config.rope_theta
                ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
            )
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    # Shim LossKwargs for transformers 5.x compatibility
    import transformers.utils

    if not hasattr(transformers.utils, "LossKwargs"):
        from typing import TypedDict

        class LossKwargs(TypedDict, total=False):
            labels: torch.Tensor
            num_items_in_batch: int

        transformers.utils.LossKwargs = LossKwargs


_install_compat_stubs()


def _patch_config_defaults(config):
    """Patch config attributes that were default in transformers 4.x but removed in 5.x."""
    defaults = {
        "pad_token_id": 0,
        "initializer_range": 0.02,
    }
    for sub_config in [config] + [
        getattr(config, name)
        for name in ("text_config", "vision_config")
        if hasattr(config, name)
    ]:
        for attr, default in defaults.items():
            if not hasattr(sub_config, attr) or getattr(sub_config, attr) is None:
                setattr(sub_config, attr, default)


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
from .src.model import Wrapper


class ModelVariant(StrEnum):
    """Available openPangu-VL model variants for vision-language tasks."""

    OPEN_PANGU_VL_7B = "7B"


class ModelLoader(ForgeModel):
    """openPangu-VL model loader implementation for vision-language tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.OPEN_PANGU_VL_7B: LLMModelConfig(
            pretrained_model_name="FreedomIntelligence/openPangu-VL-7B",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.OPEN_PANGU_VL_7B

    # Shared configuration parameters
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Vision processing parameters
    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

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
        return ModelInfo(
            model="openPangu-VL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        # Initialize processor with vision parameters
        processor_kwargs = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }

        # Load the processor
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **processor_kwargs,
        )

        # Set pad_token if missing (required for padding=True in __call__)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the openPangu-VL model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Wrapped openPangu-VL model instance for vision-language tasks.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        # Load the model with dtype override if specified
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        # Pre-load config and patch missing attributes removed in transformers 5.x
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        _patch_config_defaults(config)
        model_kwargs["config"] = config

        # Pre-download processor files so the model's _parse_preprocess_params
        # can find them when loading with local_files_only=True
        self._load_processor()

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        model = Wrapper(model)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the openPangu-VL model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
                           If specified, converts pixel_values to the specified dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        # Apply chat template to get text prompt
        text = self.processor.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info

        # Process vision inputs
        image_inputs, video_inputs = process_vision_info(self.messages)

        # Process all inputs together
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Convert pixel_values to specified dtype if provided
        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
