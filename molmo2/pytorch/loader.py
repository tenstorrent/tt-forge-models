# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Molmo2 model loader implementation for multimodal image-text-to-text tasks.

Molmo2-8B (allenai/Molmo2-8B) is a vision-language model: a Qwen3-8B style
decoder-only text backbone paired with a SigLIP-style ViT vision tower and a
pooling adapter that projects image features into the text embedding space.
The checkpoint ships as custom code on the Hub, so it is loaded with
``trust_remote_code=True`` via ``AutoModelForImageTextToText``.
"""
import torch
from PIL import Image
from typing import Optional
from transformers import AutoProcessor, AutoModelForImageTextToText

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
from ...tools.utils import get_file, cast_input_to_type
from .src.model import Wrapper


def _install_transformers_compat():
    """Make Molmo2's transformers-4.57-era remote code work on transformers 5.x.

    The checkpoint ships custom code (configuration/modeling/processing) pinned to
    transformers 4.57.1. Two backward-incompatible changes in transformers 5.x
    break it; both are restored here with minimal, behaviour-preserving shims:

    1. ``ROPE_INIT_FUNCTIONS`` lost its ``"default"`` entry (the standard,
       non-scaled RoPE). The text config uses ``rope_type="default"``. We
       re-register the historical default-RoPE initializer.
    2. ``ProcessorMixin.__init__`` now rejects any keyword that is not a sub-
       processor attribute, whereas it used to accept and store extra config
       fields. Molmo2Processor forwards several such fields
       (``image_use_col_tokens`` etc.) to ``super().__init__``. We wrap the init
       to set unknown kwargs as plain attributes (the old behaviour).
    """
    import torch as _torch
    import transformers.modeling_rope_utils as _rope_utils
    from transformers.processing_utils import ProcessorMixin

    if "default" not in _rope_utils.ROPE_INIT_FUNCTIONS:

        def _compute_default_rope_parameters(
            config, device=None, seq_len=None, layer_type=None
        ):
            base = getattr(config, "rope_theta", None) or 10000.0
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
            head_dim = getattr(config, "head_dim", None) or (
                config.hidden_size // config.num_attention_heads
            )
            dim = int(head_dim * partial_rotary_factor)
            inv_freq = 1.0 / (
                base
                ** (
                    _torch.arange(0, dim, 2, dtype=_torch.int64).float().to(device)
                    / dim
                )
            )
            return inv_freq, 1.0  # (inv_freq, attention_scaling)

        _rope_utils.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    if not getattr(ProcessorMixin.__init__, "_molmo2_compat", False):
        _orig_init = ProcessorMixin.__init__

        def _patched_init(self, *args, **kwargs):
            attrs = set(self.get_attributes())
            extra = {
                k: kwargs.pop(k)
                for k in list(kwargs)
                if k not in attrs and k != "chat_template"
            }
            _orig_init(self, *args, **kwargs)
            for k, v in extra.items():
                setattr(self, k, v)

        _patched_init._molmo2_compat = True
        ProcessorMixin.__init__ = _patched_init


class ModelVariant(StrEnum):
    """Available Molmo2 model variants."""

    MOLMO2_8B = "8B"


class ModelLoader(ForgeModel):
    """Molmo2 model loader implementation for multimodal image-text-to-text tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.MOLMO2_8B: ModelConfig(
            pretrained_model_name="allenai/Molmo2-8B",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MOLMO2_8B

    # Sample prompt for inference
    prompt = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.tokenizer = None
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
            model="molmo2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor and tokenizer for the current variant.

        Returns:
            The loaded processor instance
        """
        _install_transformers_compat()
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        return self.processor

    def load_model(self, dtype_override=None):
        """Load and return the Molmo2 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model uses its checkpoint dtype (float32).

        Returns:
            torch.nn.Module: The wrapped Molmo2 model instance returning logits.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        _install_transformers_compat()

        # Ensure processor is loaded
        if self.processor is None:
            self._load_processor()

        model_kwargs = {
            "trust_remote_code": True,
            "_attn_implementation": "eager",
            # The fp32 checkpoint is ~35GB; stream shards into the target dtype
            # to keep host peak memory bounded.
            "low_cpu_mem_usage": True,
        }
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.config.use_cache = False
        model.eval()

        self.model = Wrapper(model)
        return self.model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Molmo2 model.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' float dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        # Load image from URL and resize to a single ViT crop (378x378) so the
        # multimodal sequence length stays small for bringup.
        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file).convert("RGB").resize((378, 378))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        # Keep only the tensors the wrapped forward consumes.
        keep = [
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_token_pooling",
            "image_grids",
            "image_num_crops",
        ]
        inputs = {k: inputs[k] for k in keep if k in inputs}

        # Convert float tensors (pixel_values) to the requested dtype; leave
        # integer tensors (ids, grids, pooling indices) untouched.
        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self, **kwargs):
        """Return the model config (text sub-config carries layer counts)."""
        if self.model is not None:
            return self.model.model.config
        return None
