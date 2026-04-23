# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Penguin-VL model loader implementation for multimodal visual question answering.
"""

import inspect

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import processing_utils
from typing import Optional

from ...tools.utils import get_file
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


def _patch_nested_from_pretrained():
    # In transformers 5.x, from_pretrained always wraps model init in a meta device context
    # (via get_init_context). The model's __init__ calls from_pretrained for the vision encoder
    # with device_map=None, which check_and_set_device_map now rejects inside a meta context.
    # We patch the module-level binding so the nested call loads to CPU instead of raising.
    import sys

    mu = sys.modules.get("transformers.modeling_utils")
    if mu is None:
        return lambda: None

    orig_check = mu.check_and_set_device_map

    def patched_check(device_map):
        if device_map is None:
            try:
                from transformers.modeling_utils import (
                    get_torch_context_manager_or_global_device,
                )

                if get_torch_context_manager_or_global_device() == torch.device("meta"):
                    return "cpu"
            except Exception:
                pass
        return orig_check(device_map)

    mu.check_and_set_device_map = patched_check

    def restore():
        mu.check_and_set_device_map = orig_check

    return restore


def _load_processor_compat(pretrained_model_name):
    # transformers 5.x calls _get_arguments_from_pretrained with processor_dict as a
    # positional argument, but this model's custom processor was written for the 4.x API
    # which only accepts pretrained_model_name_or_path. We patch ProcessorMixin.from_pretrained
    # to fix the override's signature when the mismatch is detected.
    orig = processing_utils.ProcessorMixin.from_pretrained.__func__

    @classmethod
    def _patched(cls, name, **call_kwargs):
        override = cls.__dict__.get("_get_arguments_from_pretrained")
        if override is not None:
            fn = getattr(override, "__func__", override)
            params = inspect.signature(fn).parameters
            has_positional_dict = "processor_dict" in params or any(
                p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values()
            )
            if not has_positional_dict:

                @classmethod
                def compat(klass, n, processor_dict=None, **kw):
                    return fn(klass, n, **kw)

                cls._get_arguments_from_pretrained = compat
        return orig(cls, name, **call_kwargs)

    processing_utils.ProcessorMixin.from_pretrained = _patched
    try:
        return AutoProcessor.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
    finally:
        processing_utils.ProcessorMixin.from_pretrained = classmethod(orig)


class ModelVariant(StrEnum):
    """Available Penguin-VL model variants."""

    PENGUIN_VL_2B = "2B"
    PENGUIN_VL_8B = "8B"


class ModelLoader(ForgeModel):
    """Penguin-VL model loader implementation for multimodal visual question answering tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.PENGUIN_VL_2B: ModelConfig(
            pretrained_model_name="tencent/Penguin-VL-2B",
        ),
        ModelVariant.PENGUIN_VL_8B: ModelConfig(
            pretrained_model_name="tencent/Penguin-VL-8B",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.PENGUIN_VL_8B

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="Penguin-VL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Penguin-VL model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Penguin-VL model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["device_map"] = "cpu"
        model_kwargs["trust_remote_code"] = True
        model_kwargs |= kwargs

        self.processor = _load_processor_compat(pretrained_model_name)

        restore = _patch_nested_from_pretrained()
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
        finally:
            restore()
        model.eval()
        self.model = model

        return model

    def load_inputs(self, batch_size=1):
        """Load and return sample inputs for the Penguin-VL model.

        Args:
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self.processor = _load_processor_compat(
                self._variant_config.pretrained_model_name
            )

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": {"image_path": image_file}},
                    {"type": "text", "text": "Describe this image."},
                ],
            },
        ]

        inputs = self.processor(
            conversation=conversation,
            return_tensors="pt",
        )

        if self.model is not None:
            inputs = {
                k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        return inputs

    def decode_output(self, outputs, input_length=None):
        """Decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass or generated token IDs.
            input_length: Optional length of input tokens to slice from output.

        Returns:
            str: Decoded output text.
        """
        if self.processor is None:
            self.processor = _load_processor_compat(
                self._variant_config.pretrained_model_name
            )

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return self.processor.decode(outputs[0], skip_special_tokens=True)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        return self.processor.decode(next_token_id, skip_special_tokens=True)
