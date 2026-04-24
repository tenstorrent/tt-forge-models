# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLaVA-Phi-3-mini GGUF model loader implementation for multimodal conditional generation.
"""

from typing import Optional

from datasets import load_dataset
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoConfig

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available LLaVA-Phi-3-mini GGUF model variants."""

    LLAVA_PHI_3_MINI_F16 = "Phi_3_mini_F16"


class ModelLoader(ForgeModel):
    """LLaVA-Phi-3-mini GGUF model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.LLAVA_PHI_3_MINI_F16: ModelConfig(
            pretrained_model_name="xtuner/llava-phi-3-mini-gguf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAVA_PHI_3_MINI_F16

    _GGUF_FILES = {
        ModelVariant.LLAVA_PHI_3_MINI_F16: "llava-phi-3-mini-f16.gguf",
    }

    _PROCESSOR_NAME = "xtuner/llava-phi-3-mini-hf"

    sample_text = "What's shown in this image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize LLaVA-Phi-3-mini GGUF model loader."""
        super().__init__(variant)
        self.processor = None
        self.config = None

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LLaVA-Phi-3-mini GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(self._PROCESSOR_NAME)
        if self.processor.patch_size is None:
            self.processor.patch_size = 14
        return self.processor

    @staticmethod
    def _make_llava_weight_map(hf_model):
        """Build GGUF→HF weight name mapping for LLaVA models.

        The GGUF file stores only the language-model tensors using llama naming
        (blk.N.*, token_embd.weight, output.weight, output_norm.weight).
        The HF LLaVA model wraps the LM under model.language_model.*, so this
        function strips that prefix for the gguf-py name lookup and maps each
        GGUF tensor back to the full HF path.
        """
        from gguf import MODEL_ARCH, get_tensor_name_map

        n_layers = hf_model.config.text_config.num_hidden_layers
        name_map = get_tensor_name_map(MODEL_ARCH.LLAMA, n_layers)

        gguf_to_hf = {}
        LM_PREFIX = "model.language_model."
        for hf_name in hf_model.state_dict():
            if hf_name.startswith(LM_PREFIX):
                # 'model.language_model.layers.0.X' -> 'model.layers.0.X'
                lookup = "model." + hf_name[len(LM_PREFIX) :]
            elif hf_name == "lm_head.weight":
                lookup = hf_name
            else:
                continue  # vision tower / projector weights absent from GGUF

            if lookup.endswith(".weight"):
                base, suffix = lookup[:-7], ".weight"
            elif lookup.endswith(".bias"):
                base, suffix = lookup[:-5], ".bias"
            else:
                base, suffix = lookup, ""

            gguf_name = name_map.get_name(base)
            if gguf_name is not None:
                gguf_to_hf[gguf_name + suffix] = hf_name

        return gguf_to_hf

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the LLaVA-Phi-3-mini GGUF model instance."""
        import transformers.modeling_gguf_pytorch_utils as _gguf_utils

        _saved_map_fn = _gguf_utils.get_gguf_hf_weights_map

        def _patched_map_fn(hf_model, processor, model_type=None, **kw):
            mt = model_type or getattr(
                getattr(hf_model, "config", None), "model_type", None
            )
            if mt == "llava":
                return self._make_llava_weight_map(hf_model)
            return _saved_map_fn(hf_model, processor, model_type=model_type, **kw)

        _gguf_utils.get_gguf_hf_weights_map = _patched_map_fn

        try:
            pretrained_model_name = self._variant_config.pretrained_model_name

            model_kwargs = {}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs
            model_kwargs["gguf_file"] = self.gguf_file

            # The GGUF repo lacks config.json; load architecture config from the HF
            # version so the model is built with the correct Phi-3-mini dimensions.
            config = AutoConfig.from_pretrained(self._PROCESSOR_NAME)
            model_kwargs["config"] = config

            model = LlavaForConditionalGeneration.from_pretrained(
                pretrained_model_name,
                ignore_mismatched_sizes=True,
                **model_kwargs,
            ).eval()
        finally:
            _gguf_utils.get_gguf_hf_weights_map = _saved_map_fn

        self.config = model.config

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for LLaVA-Phi-3-mini GGUF."""
        if self.processor is None:
            self._load_processor()

        # Build prompt
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]

        if self.processor.chat_template is not None:
            text_prompt = self.processor.apply_chat_template(
                conversation, padding=True, add_generation_prompt=True
            )
        else:
            text_prompt = f"USER: <image>\n{self.sample_text} ASSISTANT:"

        # Load dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        # Preprocess
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]

        if dtype_override:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attention_mask = cast_input_to_type(attention_mask, dtype_override)
            pixel_values = cast_input_to_type(pixel_values, dtype_override)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
