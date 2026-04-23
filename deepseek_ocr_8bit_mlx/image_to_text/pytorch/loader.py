# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-OCR 8-bit MLX model loader implementation for document OCR tasks.
"""
import transformers.models.llama.modeling_llama as _llama_modeling
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Optional

# LlamaFlashAttention2 was removed in transformers 5.x; the model's custom code still references it
if not hasattr(_llama_modeling, "LlamaFlashAttention2"):
    _llama_modeling.LlamaFlashAttention2 = _llama_modeling.LlamaAttention

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
from ....tools.utils import get_file

# Reuse preprocessing utilities from DeepSeek-OCR
from ....deepseek.deepseek_ocr.pytorch.src.model_utils import preprocess


class ModelVariant(StrEnum):
    """Available DeepSeek-OCR 8-bit MLX model variants."""

    DEEPSEEK_OCR_8BIT = "DeepSeek_OCR_8bit"


class ModelLoader(ForgeModel):
    """DeepSeek-OCR 8-bit MLX model loader implementation for document OCR tasks."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_OCR_8BIT: ModelConfig(
            pretrained_model_name="mlx-community/DeepSeek-OCR-8bit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_OCR_8BIT

    sample_prompt = "<image>\n<|grounding|>Convert the document to markdown. "

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DeepSeek-OCR 8-bit MLX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True, "use_safetensors": True}
        # MLX quantized variants may have mismatched weight shapes
        model_kwargs["ignore_mismatched_sizes"] = True
        model_kwargs |= kwargs

        # Load config first and strip MLX quantization (incompatible with transformers 5.x)
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if hasattr(config, "quantization_config"):
            del config.quantization_config
        model_kwargs["config"] = config

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)

        model.config.return_dict = False
        model.config.use_cache = False

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        image_file = get_file("test_images/doc.png")

        inputs = preprocess(
            tokenizer=self.tokenizer,
            prompt=self.sample_prompt,
            image_file=image_file,
            base_size=1024,
            image_size=640,
            crop_mode=True,
        )

        if dtype_override is not None:
            for idx, (images_crop, images_ori) in enumerate(inputs["images"]):
                inputs["images"][idx] = (
                    images_crop.to(dtype_override),
                    images_ori.to(dtype_override),
                )

        return inputs
