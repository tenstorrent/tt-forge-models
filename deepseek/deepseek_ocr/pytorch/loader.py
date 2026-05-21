# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek OCR — Hugging Face hub load with minimal local patches (transformers==4.46.3).

1. ``snapshot_download`` checkpoint + remote ``modeling_*.py`` into ``DeepSeek_OCR_hub/``.
2. Patch ``modeling_deepseekocr.py``: ``masked_scatter_`` + ``.cuda()`` → ``masked_scatter``
   (same as ported forge) so tt-xla ``decompositions.masked_scatter`` applies.
3. ``AutoModel.from_pretrained`` on that directory; ``src/model.py`` applies patches + CUDA workaround.

Per-model ``requirements.txt`` pins ``transformers==4.46.3``; ``RequirementsManager`` handles
install/rollback during tt-xla tests.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

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
from .src.model import ensure_hub_cpu_cuda_workaround, prepare_hub_snapshot
from .src.model_utils import preprocess

DEFAULT_HF_REVISION = "9f30c71f441d010e5429c532364a86705536c53a"
_HUB_DIR_NAME = "DeepSeek_OCR_hub"


class ModelVariant(StrEnum):
    DEEPSEEK_OCR = "Ocr"


class ModelLoader(ForgeModel):
    """DeepSeek OCR via patched HF hub snapshot + ``AutoModel``."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_OCR: ModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-OCR",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_OCR
    sample_prompt = "<image>\n<|grounding|>Convert the document to markdown. "

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _hub_revision(cls) -> str:
        return os.environ.get("DEEPSEEK_OCR_HF_REVISION", DEFAULT_HF_REVISION)

    @classmethod
    def _hub_dir(cls) -> Path:
        return Path(__file__).resolve().parent / _HUB_DIR_NAME

    @classmethod
    def _ensure_hub_snapshot(cls, repo_id: str) -> Path:
        hub_dir = cls._hub_dir()
        hub_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            revision=cls._hub_revision(),
            local_dir=str(hub_dir),
        )
        if not (hub_dir / "modeling_deepseekocr.py").is_file():
            raise FileNotFoundError(
                f"Expected modeling_deepseekocr.py under {hub_dir} after "
                f"snapshot_download of {repo_id!r}"
            )
        prepare_hub_snapshot(hub_dir)
        return hub_dir

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DeepSeek",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _hub_kwargs(self) -> dict:
        return {
            "trust_remote_code": True,
            "revision": self._hub_revision(),
        }

    def _load_tokenizer(self):
        repo_id = self._variant_config.pretrained_model_name
        self._ensure_hub_snapshot(repo_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self._hub_dir()),
            local_files_only=True,
            **self._hub_kwargs(),
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        repo_id = self._variant_config.pretrained_model_name
        hub_dir = self._ensure_hub_snapshot(repo_id)

        ensure_hub_cpu_cuda_workaround()

        load_kwargs = {
            "local_files_only": True,
            "trust_remote_code": True,
            **kwargs,
        }
        if dtype_override is not None:
            load_kwargs["torch_dtype"] = dtype_override

        model = AutoModel.from_pretrained(str(hub_dir), **load_kwargs)

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
