# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VibeVoice model loader implementation.

VibeVoice (microsoft/VibeVoice-1.5B) is a long-form, generation-based
text-to-speech model. The HuggingFace repo ships weights only; the model
code lives in the standalone https://github.com/microsoft/VibeVoice repo,
which is vendored as a pinned git submodule under
``third_party/VibeVoice/`` (rather than copy-pasting the source). The two
compat needs — a ``tie_weights`` signature widened for transformers>=5 and an
idempotent ``AutoModel.register`` — are applied as runtime patches here so the
upstream files are used unmodified.

The entry class ``VibeVoiceForConditionalGeneration`` is built from the real
1.5B ``config.json`` (vendored alongside this loader) with random weights —
no multi-GB safetensors download. The bringup forward path uses
``speech_tensors=None`` so the model reduces to: embed(input_ids) -> Qwen2.5
decoder -> lm_head -> logits, which is a clean tensor-in / tensor-out forward.

Note: the full model is ~2.7B params (the "1.5B" refers to the Qwen LLM
backbone only; the acoustic/semantic VAE tokenizers, diffusion head and
connectors add the rest).
"""

import importlib
import os
import types
from typing import Optional

import torch

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _vibevoice_pkg_dir():
    """Absolute path to the upstream ``vibevoice`` package in the submodule.

    Resolved from this file's location: the loader lives at
    ``<repo>/vibevoice/pytorch/loader.py`` and the submodule at
    ``<repo>/third_party/VibeVoice``. Returns ``None`` if the submodule has
    not been checked out.
    """
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    pkg = os.path.join(repo_root, "third_party", "VibeVoice", "vibevoice")
    return pkg if os.path.isdir(pkg) else None


def _register_bare_package(name, path):
    """Register a bare package at ``path`` in ``sys.modules`` (no ``__init__``).

    Upstream ``vibevoice/__init__.py`` and ``vibevoice/modular/__init__.py``
    eagerly import the streaming/processor stack (gradio/av/aiortc) that this
    bringup does not need, so we point ``__path__`` straight at the submodule
    dirs and skip those ``__init__`` files. Setting ``__path__`` explicitly also
    shadows the same-named tt-forge-models ``vibevoice/`` model package that is
    on ``sys.path`` (avoids resolving ``import vibevoice.*`` to this loader's
    own package).
    """
    import sys

    existing = sys.modules.get(name)
    if existing is None or list(getattr(existing, "__path__", []) or [])[:1] != [path]:
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        sys.modules[name] = mod


def _patch_tie_weights(cls):
    """Widen ``tie_weights`` to accept transformers>=5's ``recompute_mapping``.

    transformers>=5 calls ``tie_weights(recompute_mapping=...)`` from
    ``init_weights``; upstream (pinned transformers<5.0.0) declares
    ``tie_weights(self)``. Wrap the original so extra args are ignored — keeps
    the upstream method body untouched.
    """
    orig = cls.__dict__.get("tie_weights")
    if orig is None or getattr(orig, "_tt_widened", False):
        return

    def tie_weights(self, *args, **kwargs):
        return orig(self)

    tie_weights._tt_widened = True
    cls.tie_weights = tie_weights


def _import_vibevoice():
    """Import the VibeVoice entry class + config from the pinned submodule.

    Returns ``(VibeVoiceForConditionalGeneration, VibeVoiceConfig)``.
    """
    pkg_dir = _vibevoice_pkg_dir()
    if pkg_dir is None:
        raise ImportError(
            "The VibeVoice submodule is not checked out. Run:\n"
            "  git submodule update --init third_party/VibeVoice"
        )

    importlib.invalidate_caches()
    _register_bare_package("vibevoice", pkg_dir)
    _register_bare_package("vibevoice.modular", os.path.join(pkg_dir, "modular"))
    _register_bare_package("vibevoice.schedule", os.path.join(pkg_dir, "schedule"))

    _patch_idempotent_register()
    from vibevoice.modular import modeling_vibevoice as _modeling
    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig

    _patch_tie_weights(_modeling.VibeVoiceForConditionalGeneration)
    return _modeling.VibeVoiceForConditionalGeneration, VibeVoiceConfig


def _patch_idempotent_register():
    """Make AutoModel(.ForCausalLM).register idempotent.

    transformers>=5 pre-registers the VibeVoice sub-configs, so the explicit
    ``AutoModel.register(...)`` calls at module import time would raise
    "already used by a Transformers model". The registrations are identical
    config->model pairs, so forcing ``exist_ok=True`` is safe. Upstream pins
    transformers<5.0.0; this lets the ported code import under transformers 5.
    """
    import transformers

    for cls_name in ("AutoModel", "AutoModelForCausalLM"):
        cls = getattr(transformers, cls_name)
        if getattr(cls.register, "_tt_idempotent", False):
            continue
        orig_bound = cls.register

        def _make(orig):
            def register(config_class, model_class=None, exist_ok=False):
                return orig(config_class, model_class, exist_ok=True)

            register._tt_idempotent = True
            return staticmethod(register)

        setattr(cls, "register", _make(orig_bound))


class _VibeVoiceLogitsWrapper(torch.nn.Module):
    """Wrap VibeVoice so forward() returns only the logits tensor.

    The inference test harness compares the *raw* forward output between CPU
    and TT with a pytree comparator (it does not call unpack_forward_output —
    that is training-only). VibeVoice's native output
    ``VibeVoiceCausalLMOutputWithPast`` contains a non-tensor leaf
    (``speech_token_num``, a Python int), which makes the comparator's
    ``torch.equal()`` fail. Returning a single tensor keeps the device
    computation identical while giving the comparator a clean tensor on both
    sides.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        kwargs.setdefault("return_dict", True)
        out = self.model(*args, **kwargs)
        return out.logits


class ModelVariant(StrEnum):
    """Available VibeVoice model variants."""

    VIBEVOICE_1_5B = "1.5B"


class ModelLoader(ForgeModel):
    """VibeVoice model loader implementation."""

    _VARIANTS = {
        ModelVariant.VIBEVOICE_1_5B: ModelConfig(
            pretrained_model_name="microsoft/VibeVoice-1.5B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIBEVOICE_1_5B

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="VibeVoice",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def _load_config(self):
        """Build the real 1.5B VibeVoiceConfig from the vendored config.json."""
        _, VibeVoiceConfig = _import_vibevoice()
        self.config = VibeVoiceConfig.from_pretrained(os.path.dirname(__file__))
        return self.config

    def load_model(self, dtype_override=torch.bfloat16, **kwargs):
        """Load and return the VibeVoice model instance (random weights).

        Args:
            dtype_override: dtype to cast the whole model to. The upstream
                model is internally mixed-precision (tokenizers/connectors/head
                follow ``config.torch_dtype`` while the Qwen LM stays fp32);
                casting the whole module to one dtype is required for a clean
                forward. Defaults to bfloat16.

        Returns:
            torch.nn.Module: The VibeVoice model instance.
        """
        VibeVoiceForConditionalGeneration, _ = _import_vibevoice()

        config = self._load_config()
        model = VibeVoiceForConditionalGeneration(config)
        if dtype_override is not None:
            model = model.to(dtype_override)
        # Wrap so forward() returns only the logits tensor (the inference
        # harness compares raw forward outputs and cannot handle the
        # non-tensor leaves in VibeVoiceCausalLMOutputWithPast).
        return _VibeVoiceLogitsWrapper(model.eval()).eval()

    def load_inputs(self, batch_size=1, seq_len=32, dtype_override=torch.bfloat16):
        """Load and return sample inputs for the VibeVoice model.

        The bringup forward path keeps ``speech_tensors=None`` so the model
        behaves as a Qwen2.5 causal LM with a single (unused) semantic
        connector call. ``speech_semantic_tensors`` is therefore required but
        its result is not consumed downstream.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.config is None:
            self._load_config()

        vocab_size = self.config.decoder_config.vocab_size
        semantic_vae_dim = self.config.semantic_vae_dim

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        speech_semantic_tensors = torch.randn(
            batch_size, 1, semantic_vae_dim, dtype=dtype_override
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "speech_semantic_tensors": speech_semantic_tensors,
            "return_dict": True,
        }

    def unpack_forward_output(self, output):
        """Return the logits tensor.

        load_model() wraps the model to already return a bare logits tensor,
        so pass tensors through unchanged; fall back to ``.logits`` if a raw
        dataclass output is ever handed in.
        """
        if torch.is_tensor(output):
            return output
        return output.logits
