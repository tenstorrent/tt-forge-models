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


def _patch_clip_tokenizer_vocab_args():
    """Fix CLIPTokenizer compat for transformers>=5.

    transformers 5.x changed CLIPTokenizer.__init__ signature:
      - Renamed vocab_file->vocab and merges_file->merges
      - Removed 'errors' as a named positional parameter

    LanguageBind calls super().__init__ with 7 positional args matching the old
    signature (vocab_file, merges_file, errors, unk_token, bos_token, eos_token,
    pad_token). This shim maps them to the new 6-param signature.
    """
    from transformers import CLIPTokenizer
    import inspect

    sig = inspect.signature(CLIPTokenizer.__init__)
    params = list(sig.parameters.keys())
    if "vocab_file" in params:
        return

    original_init = CLIPTokenizer.__init__
    _OLD_POSITIONAL = (
        "vocab",
        "merges",
        "errors",
        "unk_token",
        "bos_token",
        "eos_token",
        "pad_token",
    )

    def _compat_init(self, *args, **kwargs):
        if len(args) >= 3:
            named = dict(zip(_OLD_POSITIONAL, args))
            kwargs = {**named, **kwargs}
            args = args[len(_OLD_POSITIONAL) :]
        if "vocab_file" in kwargs and "vocab" not in kwargs:
            kwargs["vocab"] = kwargs.pop("vocab_file")
        if "merges_file" in kwargs and "merges" not in kwargs:
            kwargs["merges"] = kwargs.pop("merges_file")
        kwargs.pop("errors", None)
        original_init(self, *args, **kwargs)

    CLIPTokenizer.__init__ = _compat_init


def _patch_torchaudio_audio_backend():
    """Stub torchaudio.set_audio_backend removed in torchaudio>=2.1 that LanguageBind uses."""
    try:
        import torchaudio

        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda backend: None
    except ImportError:
        pass


def _patch_torchvision_functional_tensor():
    """Provide torchvision.transforms.functional_tensor removed in torchvision>=0.16."""
    import sys
    import types
    import torchvision.transforms.functional as F

    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    mod = types.ModuleType("torchvision.transforms.functional_tensor")
    for name in dir(F):
        setattr(mod, name, getattr(F, name))
    sys.modules["torchvision.transforms.functional_tensor"] = mod


def _patch_transformers_expand_mask():
    """Restore _expand_mask removed in transformers>=4.36 that LanguageBind still uses."""
    import torch
    import transformers.models.clip.modeling_clip as clip_module

    if hasattr(clip_module, "_expand_mask"):
        return

    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len=None):
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        inverted = 1.0 - expanded
        return inverted.masked_fill(inverted.to(torch.bool), torch.finfo(dtype).min)

    clip_module._expand_mask = _expand_mask


def _patch_languagebind_video_processor():
    """Fix LanguageBindVideoProcessor compat for transformers>=5.

    transformers 5.x ProcessorMixin.__init__ validates that components named
    in tokenizer_class are passed as kwargs. LanguageBindVideoProcessor calls
    super().__init__(**kwargs) without forwarding tokenizer, so validation fails.
    This shim replaces __init__ to call ProcessorMixin.__init__ with tokenizer.
    """
    from languagebind.video import processing_video
    from transformers import ProcessorMixin

    if getattr(
        processing_video.LanguageBindVideoProcessor.__init__, "_patched_t5", False
    ):
        return

    def _patched_init(self, config, tokenizer=None, **kwargs):
        ProcessorMixin.__init__(self, tokenizer=tokenizer, **kwargs)
        self.config = config
        self.transform = processing_video.get_video_transform(config)
        self.image_processor = processing_video.load_and_transform_video
        self.tokenizer = tokenizer

    _patched_init._patched_t5 = True
    processing_video.LanguageBindVideoProcessor.__init__ = _patched_init


def _ensure_languagebind():
    """Clone LanguageBind repo if not importable (repo has no setup.py/pyproject.toml)."""
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
    _patch_torchaudio_audio_backend()
    _patch_torchvision_functional_tensor()
    _patch_transformers_expand_mask()
    _patch_clip_tokenizer_vocab_args()


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
        _patch_languagebind_video_processor()
        from languagebind.video.tokenization_video import LanguageBindVideoTokenizer
        from languagebind.video.processing_video import LanguageBindVideoProcessor

        pretrained_model_name = self._variant_config.pretrained_model_name
        tokenizer = LanguageBindVideoTokenizer.from_pretrained(pretrained_model_name)
        self.processor = LanguageBindVideoProcessor(
            self._load_model_config(), tokenizer
        )
        return self.processor

    def _load_model_config(self):
        _ensure_languagebind()
        from languagebind.video.configuration_video import LanguageBindVideoConfig

        return LanguageBindVideoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        _ensure_languagebind()
        from languagebind.video.modeling_video import LanguageBindVideo

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
