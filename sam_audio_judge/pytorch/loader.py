# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Facebook SAM Audio Judge model loader for audio quality evaluation.
"""

import importlib
import os
import sys
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import torch

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


def _import_sam_audio_package():
    """Import the real sam_audio pip package, bypassing local directory shadowing.

    The local tt_forge_models/sam_audio/ model directory shadows the pip-installed
    sam_audio package. This function reorders sys.path and mocks CUDA-only
    transitive dependencies (xformers, torchcodec, dacvae) so the package can
    be imported on CPU-only compile systems.
    """
    site_packages = [p for p in sys.path if "site-packages" in p]
    if not site_packages:
        raise ImportError("No site-packages found in sys.path")

    sp_base = site_packages[0]

    core_mods = set()
    core_dir = os.path.join(sp_base, "core")
    if os.path.isdir(core_dir):
        for root, _dirs, files in os.walk(core_dir):
            rel = root[len(sp_base) + 1 :].replace("/", ".")
            core_mods.add(rel)
            for f in files:
                if f.endswith(".py") and f != "__init__.py":
                    core_mods.add(f"{rel}.{f[:-3]}")

    mock_mods = list(core_mods) + [
        "xformers",
        "xformers.ops",
        "xformers.ops.fmha",
        "xformers.flash_attn_3",
        "torchcodec",
        "torchcodec.decoders",
        "torchcodec.encoders",
        "torchcodec.samplers",
        "torchcodec.transforms",
        "torchcodec._core",
        "torchcodec._core.ops",
        "torchcodec._core._metadata",
        "torchcodec._core._decoder_utils",
        "torchcodec._internally_replaced_utils",
        "dacvae",
        "audiotools",
        "imagebind",
        "imagebind.data",
        "imagebind.models",
        "imagebind.models.imagebind_model",
    ]

    for name in mock_mods:
        if name not in sys.modules:
            sys.modules[name] = MagicMock()

    for k in list(sys.modules):
        if k == "sam_audio" or k.startswith("sam_audio."):
            del sys.modules[k]

    this_dir = os.path.dirname(os.path.abspath(__file__))
    worktree_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
    other = [
        p
        for p in sys.path
        if "site-packages" not in p and p != worktree_root and p != ""
    ]
    orig_path = sys.path[:]
    sys.path = site_packages + other

    try:
        return importlib.import_module("sam_audio")
    finally:
        sys.path[:] = orig_path


@dataclass
class SAMAudioJudgeOutput:
    overall: Optional[torch.Tensor] = None
    recall: Optional[torch.Tensor] = None
    precision: Optional[torch.Tensor] = None
    faithfulness: Optional[torch.Tensor] = None


class _TransformerBlock(torch.nn.Module):
    def __init__(self, dim, n_heads, n_layers):
        super().__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x, padding_mask=None):
        return self.encoder(x, src_key_padding_mask=padding_mask)


class _AudioEncoder(torch.nn.Module):
    def __init__(self, codebook_dim, hop_length):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            1, codebook_dim, kernel_size=hop_length, stride=hop_length
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.conv(x)


class StandaloneSAMAudioJudge(torch.nn.Module):
    """Standalone SAM Audio Judge model for compile-only environments.

    Mirrors the real SAMAudioJudgeModel architecture with standard PyTorch
    modules, avoiding CUDA-only dependencies (xformers, dacvae, etc.).
    """

    def __init__(
        self,
        codebook_dim=128,
        hidden_size=768,
        n_heads=12,
        n_layers=6,
        bottleneck_dim=256,
        text_hidden_size=768,
        hop_length=1920,
    ):
        super().__init__()
        self.hop_length = hop_length
        self.audio_codec = _AudioEncoder(codebook_dim, hop_length)
        self.data_proj = torch.nn.Linear(codebook_dim, hidden_size)
        self.transformer = _TransformerBlock(hidden_size, n_heads, n_layers)
        ft_n_heads = 8
        self.finetune_transformer = _TransformerBlock(
            bottleneck_dim, ft_n_heads, max(n_layers // 2, 1)
        )
        from transformers import ModernBertConfig, AutoModel

        text_config = ModernBertConfig(
            hidden_size=text_hidden_size,
            num_attention_heads=12,
            num_hidden_layers=6,
            intermediate_size=text_hidden_size * 4,
        )
        self.text_model = AutoModel.from_config(text_config)
        self.cat_audio_proj = torch.nn.Linear(2 * hidden_size, bottleneck_dim)
        self.text_proj1 = torch.nn.Linear(text_hidden_size, hidden_size, bias=False)
        self.text_proj2 = torch.nn.Linear(hidden_size, bottleneck_dim)
        self.layer_norm = torch.nn.LayerNorm(bottleneck_dim)
        self.proj_audio_and_text = torch.nn.Linear(2 * bottleneck_dim, bottleneck_dim)
        self.finetune_data_proj = torch.nn.Linear(bottleneck_dim, bottleneck_dim)
        self.head = torch.nn.Linear(bottleneck_dim, 4, bias=False)
        self.mean = torch.nn.Parameter(torch.zeros(4, requires_grad=False))
        self.std = torch.nn.Parameter(torch.ones(4, requires_grad=False))

    def forward(
        self,
        input_ids: torch.Tensor,
        input_values: torch.Tensor,
        separated_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> SAMAudioJudgeOutput:
        text_output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        text_features = self.text_proj1(text_output.hidden_states[-2][:, 0])

        stacked_audios = torch.cat([input_values, separated_values], dim=0)
        stacked_codec_features = self.audio_codec(stacked_audios)

        feature_padding_mask = None
        if padding_mask is not None:
            feature_padding_mask = padding_mask[:, :: self.hop_length]

        stacked_features = self.transformer(
            self.data_proj(stacked_codec_features.transpose(1, 2)),
            padding_mask=feature_padding_mask,
        )
        input_features, hyp_features = stacked_features.chunk(2, 0)
        audio_features = self.cat_audio_proj(
            torch.cat([hyp_features, input_features], dim=2)
        )
        expanded_text = (
            self.layer_norm(self.text_proj2(text_features))
            .unsqueeze(1)
            .expand_as(audio_features)
        )
        audio_and_text = self.proj_audio_and_text(
            torch.cat([audio_features, expanded_text], dim=2)
        )
        finetune_output = self.finetune_transformer(
            self.finetune_data_proj(audio_and_text),
            padding_mask=feature_padding_mask,
        )
        result = self.head(finetune_output)
        pooled = result.mean(dim=1)
        de_normalized = pooled * self.std + self.mean
        return SAMAudioJudgeOutput(*de_normalized.chunk(4, dim=1))


class SAMAudioJudgeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        output = self.model(**kwargs)
        return output.overall


class ModelVariant(StrEnum):
    """Available SAM Audio Judge model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """Facebook SAM Audio Judge model loader for audio quality evaluation."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="facebook/sam-audio-judge",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SAMAudioJudge",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_AUDIO_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _use_random_weights(self):
        return os.environ.get("TT_RANDOM_WEIGHTS", "") == "1"

    def _load_processor(self):
        if self._use_random_weights():
            return self._load_random_processor()
        sam_audio = _import_sam_audio_package()
        self._processor = sam_audio.SAMAudioJudgeProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self._processor

    def _load_random_processor(self):
        from transformers import AutoTokenizer

        class _FakeProcessor:
            def __init__(self):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "answerdotai/ModernBERT-base"
                )
                self.sample_rate = 16000

            def __call__(self, text, input_audio, separated_audio):
                tokens = self.tokenizer(
                    text, return_tensors="pt", padding=True, truncation=True
                )
                return {
                    "input_ids": tokens["input_ids"],
                    "attention_mask": tokens["attention_mask"],
                    "input_values": input_audio[0]
                    if isinstance(input_audio, list)
                    else input_audio,
                    "separated_values": separated_audio[0]
                    if isinstance(separated_audio, list)
                    else separated_audio,
                }

        self._processor = _FakeProcessor()
        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        if self._use_random_weights():
            model = StandaloneSAMAudioJudge()
        else:
            sam_audio = _import_sam_audio_package()
            SAMAudioJudgeModel = sam_audio.SAMAudioJudgeModel

            model_kwargs = {}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs

            model = SAMAudioJudgeModel.from_pretrained(
                self._variant_config.pretrained_model_name, **model_kwargs
            )

        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return SAMAudioJudgeWrapper(model)

    def load_inputs(self, dtype_override=None):
        if self._processor is None:
            self._load_processor()

        sampling_rate = 16000
        duration_seconds = 1
        num_samples = sampling_rate * duration_seconds

        input_audio = torch.randn(1, num_samples)
        separated_audio = torch.randn(1, num_samples)
        description = "A person speaking"

        inputs = self._processor(
            text=[description],
            input_audio=[input_audio],
            separated_audio=[separated_audio],
        )

        if dtype_override is not None:
            inputs = {
                k: (
                    v.to(dtype_override)
                    if isinstance(v, torch.Tensor) and v.is_floating_point()
                    else v
                )
                for k, v in inputs.items()
            }

        return inputs
