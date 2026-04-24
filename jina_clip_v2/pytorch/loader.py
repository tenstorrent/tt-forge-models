# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Jina CLIP v2 model loader implementation for image-text similarity.
"""
import torch
from transformers import AutoModel, AutoProcessor, PreTrainedModel
from typing import Optional

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
from datasets import load_dataset


class ModelVariant(StrEnum):
    """Available Jina CLIP v2 model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Jina CLIP v2 model loader for image-text similarity tasks."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="jinaai/jina-clip-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="JINA_CLIP_V2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant."""
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Jina CLIP v2 model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Jina CLIP v2 model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True, "return_dict": False}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # transformers 5.x always initializes models on meta device via get_init_context,
        # but EVAVisionTransformer calls .item() during __init__ which fails on meta tensors.
        # Temporarily patch to exclude meta device during loading.
        _orig_get_init_context = PreTrainedModel.__dict__["get_init_context"]

        @classmethod  # type: ignore[misc]
        def _get_init_context_no_meta(cls, dtype, is_quantized, _is_ds_init_called):
            contexts = _orig_get_init_context.__func__(
                cls, dtype, is_quantized, _is_ds_init_called
            )
            return [
                c
                for c in contexts
                if not (isinstance(c, torch.device) and c.type == "meta")
            ]

        PreTrainedModel.get_init_context = _get_init_context_no_meta
        try:
            model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        finally:
            PreTrainedModel.get_init_context = _orig_get_init_context
        model.eval()

        # The XLM-Roberta text encoder explicitly casts tokens to float32 internally
        # for numerical stability, causing text_embeds to be float32 while
        # image_embeds stays bfloat16, which breaks torch.matmul in CLIP forward.
        # Wrap get_text_features to cast its output back to the target dtype.
        if dtype_override is not None:
            _dtype = dtype_override
            _orig_get_text_features = model.get_text_features

            def _cast_text_features(*args, **kwargs):
                result = _orig_get_text_features(*args, **kwargs)
                return result.to(_dtype) if isinstance(result, torch.Tensor) else result

            model.get_text_features = _cast_text_features

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Jina CLIP v2 model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size (default 1).

        Returns:
            dict: Input tensors containing pixel values, input IDs, and attention masks.
        """
        if self.processor is None:
            self._load_processor()

        # Load image from HuggingFace dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        self.text_prompts = ["a photo of a cat", "a photo of a dog"]

        inputs = self.processor(
            text=self.text_prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        # Replicate tensors for batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs

    def post_process(self, outputs):
        """Post-process Jina CLIP v2 model outputs to extract similarity scores.

        Args:
            outputs: Raw model output tuple.
        """
        if self.text_prompts is None:
            self.text_prompts = ["a photo of a cat", "a photo of a dog"]

        logits_per_image = outputs[0]
        probs = logits_per_image.softmax(dim=1)

        for i, text in enumerate(self.text_prompts):
            print(f"Probability of '{text}':", probs[0, i].item())

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        Args:
            fwd_output: Output from the model's forward pass (tuple)

        Returns:
            torch.Tensor: Concatenated flattened outputs for backward pass
        """
        if isinstance(fwd_output, tuple):
            tensors = []
            for item in fwd_output:
                if isinstance(item, torch.Tensor):
                    tensors.append(item.flatten())
                elif hasattr(item, "last_hidden_state"):
                    tensors.append(item.last_hidden_state.flatten())
            if tensors:
                return torch.cat(tensors, dim=0)
        return fwd_output
