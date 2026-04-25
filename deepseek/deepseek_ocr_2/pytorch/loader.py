# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek OCR-2 model loader implementation for document OCR tasks.
"""
import glob
import inspect
import os
import sys
import textwrap
from transformers import AutoTokenizer, AutoModel
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
from ....tools.utils import get_file

# Reuse preprocessing utilities from DeepSeek-OCR
from ...deepseek_ocr.pytorch.src.model_utils import preprocess


class ModelVariant(StrEnum):
    """Available DeepSeek OCR-2 model variants."""

    DEEPSEEK_OCR_2 = "Ocr2"
    DEEPSEEK_OCR_2_UNSLOTH = "Ocr2-Unsloth"


class ModelLoader(ForgeModel):
    """DeepSeek OCR-2 model loader implementation for document OCR tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DEEPSEEK_OCR_2: ModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-OCR-2",
        ),
        ModelVariant.DEEPSEEK_OCR_2_UNSLOTH: ModelConfig(
            pretrained_model_name="unsloth/DeepSeek-OCR-2",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_OCR_2

    # Shared configuration parameters
    sample_prompt = "<image>\n<|grounding|>Convert the document to markdown. "

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
    ):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DeepSeek",
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

    @staticmethod
    def _patch_cached_ocr2_modules():
        """Patch cached HF modules to replace .cuda() with device-agnostic calls."""
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        pattern = os.path.join(
            hf_home,
            "modules",
            "transformers_modules",
            "deepseek_hyphen_ai",
            "DeepSeek_hyphen_OCR_hyphen_2",
            "*",
            "modeling_deepseekocr2.py",
        )
        old = "images_seq_mask[idx].unsqueeze(-1).cuda()"
        new = "images_seq_mask[idx].unsqueeze(-1).to(inputs_embeds[idx].device)"
        for cached_file in glob.glob(pattern):
            with open(cached_file) as f:
                content = f.read()
            if old in content:
                with open(cached_file, "w") as f:
                    f.write(content.replace(old, new))
                stale = [k for k in sys.modules if "modeling_deepseekocr2" in k]
                for k in stale:
                    del sys.modules[k]

    @staticmethod
    def _patch_model_forward(model):
        """In-memory patch: replace .cuda() in DeepseekOCR2Model.forward."""
        cls = type(model.model)
        src = inspect.getsource(cls.forward)
        old = "images_seq_mask[idx].unsqueeze(-1).cuda()"
        new = "images_seq_mask[idx].unsqueeze(-1).to(inputs_embeds[idx].device)"
        if old not in src:
            return
        src = textwrap.dedent(src).replace(old, new)
        mod = inspect.getmodule(cls)
        globs = {**vars(mod), "__builtins__": __builtins__}
        exec(src, globs)  # noqa: S102
        cls.forward = globs["forward"]

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Patch cached module files before loading (for future imports)
        self._patch_cached_ocr2_modules()

        model = AutoModel.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            use_safetensors=True,
            **kwargs,
        )

        # In-memory patch in case the module was already imported before patching
        self._patch_model_forward(model)

        model.config.return_dict = False
        model.config.use_cache = False

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def _create_synthetic_inputs(self, dtype_override=None):
        """Synthetic inputs with zero images so the model skips image encoding."""
        import torch

        text = "Convert the document to markdown."
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        input_ids = torch.LongTensor([[0] + tokens])
        seq_len = input_ids.shape[1]
        dtype = dtype_override or torch.float32
        return {
            "input_ids": input_ids,
            # Zero image_ori causes forward to skip masked_scatter_ (sum==0 guard)
            "images": [
                (
                    torch.zeros((1, 3, 1024, 1024), dtype=dtype),
                    torch.zeros((1, 3, 640, 640), dtype=dtype),
                )
            ],
            "images_seq_mask": torch.zeros(1, seq_len, dtype=torch.bool),
            "images_spatial_crop": torch.zeros((1, 2), dtype=torch.long),
        }

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        try:
            image_file = get_file("test_images/doc.png")
        except (ValueError, RuntimeError):
            return self._create_synthetic_inputs(dtype_override)

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
