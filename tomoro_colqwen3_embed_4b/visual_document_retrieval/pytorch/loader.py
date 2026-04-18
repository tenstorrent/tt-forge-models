# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tomoro ColQwen3 Embed 4B model loader implementation for visual document retrieval.
"""
import sys
import torch
from transformers import AutoModel, AutoProcessor
from typing import Optional

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


class ModelVariant(StrEnum):
    """Available Tomoro ColQwen3 Embed 4B model variants."""

    TOMORO_COLQWEN3_EMBED_4B = "tomoro-colqwen3-embed-4b"


class ModelLoader(ForgeModel):
    """Tomoro ColQwen3 Embed 4B model loader for visual document retrieval."""

    _VARIANTS = {
        ModelVariant.TOMORO_COLQWEN3_EMBED_4B: ModelConfig(
            pretrained_model_name="TomoroAI/tomoro-colqwen3-embed-4b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TOMORO_COLQWEN3_EMBED_4B

    sample_queries = [
        "What is the revenue growth for Q3 2024?",
    ]

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Tomoro ColQwen3 Embed 4B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            max_num_visual_tokens=1280,
        )
        return self.processor

    @staticmethod
    def _patch_colqwen3_tie_weights():
        import inspect

        for mod in sys.modules.values():
            cls = getattr(mod, "ColQwen3", None)
            if not inspect.isclass(cls):
                continue
            if not hasattr(cls, "tie_weights"):
                continue
            sig = inspect.signature(cls.tie_weights)
            if "kwargs" in sig.parameters or "recompute_mapping" in sig.parameters:
                continue
            original = cls.tie_weights

            def _tie_weights(self, **kwargs):
                return original(self)

            cls.tie_weights = _tie_weights

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        try:
            model = AutoModel.from_pretrained(
                pretrained_model_name,
                **model_kwargs,
            )
        except TypeError:
            self._patch_colqwen3_tie_weights()
            model = AutoModel.from_pretrained(
                pretrained_model_name,
                **model_kwargs,
            )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.processor is None:
            self._load_processor()

        inputs = self.processor.process_texts(
            texts=self.sample_queries,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    inputs[key] = value.to(dtype_override)

        return inputs

    def decode_output(self, outputs, inputs=None):
        if hasattr(outputs, "embeddings"):
            embeddings = outputs.embeddings
        elif isinstance(outputs, (tuple, list)):
            embeddings = outputs[0]
        else:
            embeddings = outputs

        return embeddings.tolist()
