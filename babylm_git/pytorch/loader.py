# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BabyLM GIT model loader implementation for multimodal causal language modeling.
"""

from typing import Optional
import torch
import transformers

# transformers 5.x removed ViTFeatureExtractor; the remote modeling_git.py imports it
# at module level (but never calls it). Inject into _objects because the _LazyModule
# uses its own dict (_objects), not the module's __dict__, for attribute resolution.
if not hasattr(transformers, "ViTFeatureExtractor"):
    import sys
    from transformers import ViTImageProcessor
    _tm = sys.modules["transformers"]
    if hasattr(_tm, "_objects"):
        _tm._objects["ViTFeatureExtractor"] = ViTImageProcessor
    else:
        _tm.ViTFeatureExtractor = ViTImageProcessor

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, PreTrainedModel

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
from ...tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available BabyLM GIT model variants."""

    MULTIMODAL_BASELINE = "multimodal_baseline"


class ModelLoader(ForgeModel):
    """BabyLM GIT model loader for multimodal causal language modeling."""

    _VARIANTS = {
        ModelVariant.MULTIMODAL_BASELINE: ModelConfig(
            pretrained_model_name="BabyLM-community/babylm-multimodal-baseline-git",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MULTIMODAL_BASELINE

    sample_text = "A photo of"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize BabyLM GIT model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BabyLM GIT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CAUSAL_LM,
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
        """Load and return the BabyLM GIT model instance."""
        model_name = self._variant_config.pretrained_model_name

        # The remote modeling_git.py calls ViTModel.from_pretrained() inside __init__,
        # which fails under transformers 5.x's meta-device init context. Strip the
        # meta-device context so the nested from_pretrained can allocate real tensors.
        _orig_get_init_context = PreTrainedModel.get_init_context

        @classmethod  # type: ignore[misc]
        def _no_meta_get_init_context(cls, dtype, is_quantized, _is_ds_init_called):
            contexts = _orig_get_init_context.__func__(cls, dtype, is_quantized, _is_ds_init_called)
            return [c for c in contexts if not isinstance(c, torch.device)]

        PreTrainedModel.get_init_context = _no_meta_get_init_context
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                **kwargs,
            )
        finally:
            PreTrainedModel.get_init_context = _orig_get_init_context

        # transformers 5.x _move_missing_keys_from_meta_to_device() replaces all
        # persistent=False buffers with torch.empty_like() (uninitialized), even when
        # not using meta device. Re-initialize position_ids from its definition.
        model.git.embeddings.position_ids = torch.arange(
            model.config.max_position_embeddings
        ).expand((1, -1))

        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for BabyLM GIT."""
        if self.processor is None:
            self._load_processor()

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        inputs = self.processor(
            images=image,
            text=self.sample_text,
            return_tensors="pt",
        )

        if dtype_override:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return dict(inputs)
