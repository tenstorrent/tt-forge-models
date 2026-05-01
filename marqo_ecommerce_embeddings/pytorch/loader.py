# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Marqo Ecommerce Embeddings model loader implementation for image-text similarity.
"""
import torch
from transformers import AutoModel, AutoProcessor
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
    """Available Marqo Ecommerce Embeddings model variants."""

    LARGE = "Large"


class ModelLoader(ForgeModel):
    """Marqo Ecommerce Embeddings model loader for image-text similarity tasks."""

    _VARIANTS = {
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="Marqo/marqo-ecommerce-embeddings-L",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Marqo-Ecommerce-Embeddings",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant."""
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Marqo Ecommerce Embeddings model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The model instance for image-text similarity.
        """
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from transformers.modeling_utils import local_torch_dtype, init

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # transformers 5.x wraps __init__ in torch.device("meta"), causing
        # open_clip.create_model's model.to('cpu') to fail on meta tensors.
        # Patch get_init_context on the remote class to skip the meta device.
        #
        # The remote __init__ also omits post_init(), so all_tied_weights_keys
        # is never set. Patch __init__ to initialize it to {} when absent.
        model_cls = get_class_from_dynamic_module(
            "marqo_fashionSigLIP.MarqoFashionSigLIP",
            pretrained_model_name,
            trust_remote_code=True,
        )

        @classmethod
        def _no_meta_get_init_context(cls, dtype, is_quantized, _is_ds_init_called):
            return [local_torch_dtype(dtype, cls.__name__), init.no_tie_weights()]

        orig_cls_init = model_cls.__init__

        def _patched_init(self, config):
            orig_cls_init(self, config)
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}

        orig_ctx = model_cls.__dict__.get("get_init_context")
        model_cls.get_init_context = _no_meta_get_init_context
        model_cls.__init__ = _patched_init
        try:
            model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        finally:
            model_cls.__init__ = orig_cls_init
            if orig_ctx is not None:
                model_cls.get_init_context = orig_ctx
            else:
                del model_cls.get_init_context

        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size (default 1).

        Returns:
            dict: Input tensors containing pixel values and text tokens.
        """
        if self.processor is None:
            self._load_processor()

        # Load image from HuggingFace dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        self.text_prompts = ["a cat", "a dog", "a shoe"]

        inputs = self.processor(
            text=self.text_prompts,
            images=[image],
            padding="max_length",
            return_tensors="pt",
        )

        # Replicate tensors for batch size
        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        # forward() only accepts input_ids and pixel_values; drop attention_mask etc.
        return {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"],
        }

    def post_process(self, outputs):
        """Post-process model outputs to extract similarity scores.

        Args:
            outputs: Raw model output (image_features, text_features, logit_scale)
        """
        if self.text_prompts is None:
            self.text_prompts = ["a cat", "a dog", "a shoe"]

        image_features, text_features, logit_scale = outputs
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        for i, text in enumerate(self.text_prompts):
            print(f"Probability of '{text}':", text_probs[0, i].item())

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        Args:
            fwd_output: Output from the model's forward pass (tuple of tensors)

        Returns:
            torch.Tensor: Concatenated flattened outputs for backward pass
        """
        if isinstance(fwd_output, tuple):
            tensors = []
            for item in fwd_output:
                if isinstance(item, torch.Tensor):
                    tensors.append(item.flatten())
            if tensors:
                return torch.cat(tensors, dim=0)
        return fwd_output
