# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LanguageBind Video model loader implementation for video-text similarity.
"""
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import torch
from typing import Optional

_LANGUAGEBIND_REPO = "https://github.com/PKU-YuanGroup/LanguageBind.git"
_LANGUAGEBIND_COMMIT = "7070c53375661cdb235801176b564b45f96f0648"


def _ensure_languagebind():
    """Clone LanguageBind repo if not importable (repo has no setup.py/pyproject.toml)."""
    try:
        import languagebind  # noqa: F401

        return
    except ImportError:
        pass

    cache_dir = os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "languagebind_repo",
    )
    if not os.path.isdir(cache_dir):
        subprocess.run(
            ["git", "clone", "--filter=blob:none", _LANGUAGEBIND_REPO, cache_dir],
            check=True,
        )
        subprocess.run(
            ["git", "-C", cache_dir, "checkout", _LANGUAGEBIND_COMMIT],
            check=True,
        )
    if cache_dir not in sys.path:
        sys.path.insert(0, cache_dir)


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


class ModelVariant(StrEnum):
    """Available LanguageBind Video model variants."""

    LANGUAGEBIND_VIDEO_MERGE = "LanguageBind_Video_merge"
    LANGUAGEBIND_VIDEO = "LanguageBind_Video"


class ModelLoader(ForgeModel):
    """LanguageBind Video model loader for video-text similarity tasks."""

    _VARIANTS = {
        ModelVariant.LANGUAGEBIND_VIDEO_MERGE: ModelConfig(
            pretrained_model_name="LanguageBind/LanguageBind_Video_merge",
        ),
        ModelVariant.LANGUAGEBIND_VIDEO: ModelConfig(
            pretrained_model_name="LanguageBind/LanguageBind_Video",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LANGUAGEBIND_VIDEO_MERGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="LanguageBind_Video",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        _ensure_languagebind()
        from languagebind import LanguageBindVideoTokenizer, LanguageBindVideoProcessor

        pretrained_model_name = self._variant_config.pretrained_model_name
        tokenizer = LanguageBindVideoTokenizer.from_pretrained(pretrained_model_name)
        self.processor = LanguageBindVideoProcessor(
            self._load_model_config(), tokenizer
        )
        return self.processor

    def _load_model_config(self):
        _ensure_languagebind()
        from languagebind import LanguageBindVideoConfig

        return LanguageBindVideoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        _ensure_languagebind()
        from languagebind import LanguageBindVideo

        pretrained_model_name = self._variant_config.pretrained_model_name
        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = LanguageBindVideo.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        return model

    @staticmethod
    def _create_synthetic_video(num_frames=8, height=224, width=224):
        """Create a temporary synthetic video file and return its path."""
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp.name, fourcc, 8, (width, height))
        for _ in range(num_frames):
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()
        tmp.close()
        return tmp.name

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        # Generate a synthetic video file (8 frames of 224x224 RGB)
        video_path = self._create_synthetic_video()

        self.text_prompts = ["a dog playing in the park", "a person riding a bicycle"]

        data = self.processor([video_path], self.text_prompts, return_tensors="pt")

        # Replicate tensors for batch size
        for key in data:
            if torch.is_tensor(data[key]):
                data[key] = data[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            if "pixel_values" in data:
                data["pixel_values"] = data["pixel_values"].to(dtype_override)

        return data

    def post_process(self, outputs):
        if self.text_prompts is None:
            self.text_prompts = [
                "a dog playing in the park",
                "a person riding a bicycle",
            ]

        logits_per_image = outputs[0]
        probs = logits_per_image.softmax(dim=1)
        for i, text in enumerate(self.text_prompts):
            print(f"Probability of '{text}':", probs[0, i].item())

    def unpack_forward_output(self, fwd_output):
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
