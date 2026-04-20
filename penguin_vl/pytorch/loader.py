# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Penguin-VL model loader implementation for multimodal visual question answering.
"""
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from typing import Optional


def _compute_default_rope_parameters(config, device, seq_len=None, **kwargs):
    if "dim" in kwargs:
        dim = kwargs["dim"]
        base = kwargs.get("base", 10000)
    else:
        base = getattr(config, "rope_theta", 10000)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        partial = getattr(config, "partial_rotary_factor", 1.0)
        dim = int(head_dim * partial)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    return inv_freq, 1.0


if "default" not in ROPE_INIT_FUNCTIONS:
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

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


class ModelVariant(StrEnum):
    """Available Penguin-VL model variants."""

    PENGUIN_VL_8B = "8B"


class ModelLoader(ForgeModel):
    """Penguin-VL model loader implementation for multimodal visual question answering tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
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

    def _load_processor(self):
        pretrained_model_name = self._variant_config.pretrained_model_name
        try:
            return AutoProcessor.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
        except TypeError:
            # Workaround: custom processor incompatible with transformers >=5.2.0
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            image_processor = AutoImageProcessor.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            ProcessorClass = get_class_from_dynamic_module(
                "processing_penguinvl.PenguinVLQwen3Processor",
                pretrained_model_name,
                trust_remote_code=True,
            )
            return ProcessorClass(
                image_processor=image_processor,
                tokenizer=tokenizer,
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

        model_kwargs["device_map"] = "auto"
        model_kwargs["trust_remote_code"] = True
        model_kwargs |= kwargs

        self.processor = self._load_processor()

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
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
            self.processor = self._load_processor()

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
            self.processor = self._load_processor()

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return self.processor.decode(outputs[0], skip_special_tokens=True)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        return self.processor.decode(next_token_id, skip_special_tokens=True)
