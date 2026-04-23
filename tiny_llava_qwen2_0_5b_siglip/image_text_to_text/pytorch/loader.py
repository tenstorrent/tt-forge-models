# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TinyLLaVA-Qwen2-0.5B-SigLIP model loader implementation for image-text-to-text tasks.
"""

import importlib
import os
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor


def _register_tinyllava():
    """Register TinyLLaVA with transformers auto classes.

    The tinyllava package was designed for transformers 4.x; patch compatibility
    shims before importing so it works with transformers 5.x.
    """
    import transformers
    import transformers.modeling_utils

    # apply_chunking_to_forward moved from modeling_utils to pytorch_utils in transformers 5.x
    if not hasattr(transformers.modeling_utils, "apply_chunking_to_forward"):
        from transformers.pytorch_utils import apply_chunking_to_forward

        transformers.modeling_utils.apply_chunking_to_forward = (
            apply_chunking_to_forward
        )

    # find_pruneable_heads_and_indices and prune_linear_layer were removed in transformers 5.x;
    # only used by the QFormer connector which TinyLLaVA-Qwen2 does not use.
    if not hasattr(transformers.modeling_utils, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(*args, **kwargs):
            raise NotImplementedError(
                "find_pruneable_heads_and_indices removed in transformers 5.x"
            )

        def prune_linear_layer(*args, **kwargs):
            raise NotImplementedError("prune_linear_layer removed in transformers 5.x")

        transformers.modeling_utils.find_pruneable_heads_and_indices = (
            find_pruneable_heads_and_indices
        )
        transformers.modeling_utils.prune_linear_layer = prune_linear_layer

    # Patch import_modules to skip connectors with missing dependencies (e.g. qformer, resampler)
    import tinyllava.utils as _tu

    def _import_modules_safe(models_dir, namespace):
        for file in os.listdir(models_dir):
            if (
                not file.startswith("_")
                and not file.startswith(".")
                and file.endswith(".py")
            ):
                model_name = file[: file.find(".py")]
                try:
                    importlib.import_module(namespace + "." + model_name)
                except (ImportError, ModuleNotFoundError):
                    pass

    _tu.import_modules = _import_modules_safe

    import tinyllava.model
    from tinyllava.model import TinyLlavaConfig, TinyLlavaForConditionalGeneration
    from transformers import AutoConfig

    # Patch tie_weights to accept new recompute_mapping kwarg added in transformers 5.x
    _orig_tie_weights = TinyLlavaForConditionalGeneration.tie_weights

    def _patched_tie_weights(self, **kwargs):
        return _orig_tie_weights(self)

    TinyLlavaForConditionalGeneration.tie_weights = _patched_tie_weights

    AutoConfig.register("tinyllava", TinyLlavaConfig, exist_ok=True)
    AutoModelForCausalLM.register(
        TinyLlavaConfig, TinyLlavaForConditionalGeneration, exist_ok=True
    )


def _tokenizer_image_token(
    prompt, tokenizer, image_token_index=-200, return_tensors=None
):
    """Tokenize prompt, replacing <image> with image_token_index placeholders."""

    def _insert_separator(x, sep):
        return [ele for sublist in zip(x, [sep] * len(x)) for ele in sublist][:-1]

    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in _insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    return input_ids


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
    """Available TinyLLaVA-Qwen2-0.5B-SigLIP model variants."""

    TINY_LLAVA_QWEN2_0_5B_SIGLIP = "tiny_llava_qwen2_0_5b_siglip"


class ModelLoader(ForgeModel):
    """TinyLLaVA-Qwen2-0.5B-SigLIP model loader for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.TINY_LLAVA_QWEN2_0_5B_SIGLIP: ModelConfig(
            pretrained_model_name="Zhang199/TinyLLaVA-Qwen2-0.5B-SigLIP",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_LLAVA_QWEN2_0_5B_SIGLIP

    sample_text = "What is shown in this image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize TinyLLaVA-Qwen2-0.5B-SigLIP model loader."""
        super().__init__(variant)
        self.processor = None
        self.image_processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="TinyLLaVA-Qwen2-0.5B-SigLIP",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the TinyLLaVA-Qwen2-0.5B-SigLIP model instance."""
        _register_tinyllava()

        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()

        if self.processor is None:
            self._load_processor()

        self.image_processor = model.vision_tower._image_processor

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for TinyLLaVA-Qwen2-0.5B-SigLIP."""
        if self.processor is None:
            self._load_processor()

        conversation = [
            {
                "role": "user",
                "content": f"<image>\n{self.sample_text}",
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        input_ids = _tokenizer_image_token(
            text_prompt, self.processor, return_tensors="pt"
        ).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        if self.image_processor is not None:
            pixel_values = self.image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ]
        else:
            import torchvision.transforms.functional as TF

            pixel_values = TF.to_tensor(image.convert("RGB")).unsqueeze(0)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": pixel_values,
        }

        if dtype_override is not None:
            inputs = {
                k: cast_input_to_type(v, dtype_override) if v.is_floating_point() else v
                for k, v in inputs.items()
            }

        if batch_size > 1:
            inputs = {
                k: v.repeat_interleave(batch_size, dim=0) for k, v in inputs.items()
            }

        return inputs
