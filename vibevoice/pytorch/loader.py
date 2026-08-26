# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VibeVoice model loader implementation.

VibeVoice (microsoft/VibeVoice-1.5B) is a long-form, generation-based
text-to-speech model. The HuggingFace repo ships weights only; the model code
lives in a standalone GitHub repo, vendored as a pinned git submodule under
``third_party/VibeVoice/`` (rather than copy-pasting the source). Every
transformers>=5 incompatibility is handled as a runtime patch in
``compat.py`` so the upstream files are used unmodified.

**Which repo is pinned, and why it is not Microsoft's.** Microsoft removed the
VibeVoice-TTS inference stack from ``microsoft/VibeVoice`` on 2025-09-05; the
file ``modeling_vibevoice_inference.py`` does not exist anywhere in that
repo's history. What survives there is a *training* forward returning a
diffusion loss — no ``generate()``, no denoise sampling loop, no VAE
decode-to-waveform. The submodule therefore pins
``vibevoice-community/VibeVoice`` @ ``631804b``, a preserved pre-removal fork
that still carries the full inference path and the ``VibeVoiceProcessor`` TTS
conditioning path. The model weights are still first-party
(``microsoft/VibeVoice-1.5B`` on HuggingFace).

Two entry paths are exposed:

``load_model()`` / ``load_inputs()`` — the compiler-bringup path. Builds
``VibeVoiceForConditionalGeneration`` from the real 1.5B ``config.json``
(vendored alongside this loader) with **random weights**, so no multi-GB
safetensors download. The forward passes ``speech_tensors=None`` so the model
reduces to embed(input_ids) -> Qwen2.5 decoder -> lm_head -> logits, a clean
tensor-in / tensor-out forward the generic inference harness can compare.

``load_tts_model()`` / ``load_processor()`` / ``load_tts_inputs()`` — the real
TTS path. Downloads the pretrained checkpoint and returns
``VibeVoiceForConditionalGenerationInference``, whose ``generate()`` runs the
acoustic + semantic VAE tokenizers, both connectors, the diffusion-head
sampling loop, AR generation with KV cache and the acoustic decode to
waveform. Used by the e2e demo and the per-component tests, not by the
generic harness (``generate()`` is not a single forward).

Note: the full model is ~2.7B params (the "1.5B" refers to the Qwen LLM
backbone only; the acoustic/semantic VAE tokenizers, diffusion head and
connectors add the rest).
"""

import importlib
import os
import types
from typing import Optional

import torch

from . import compat
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


def _register_vibevoice_packages():
    """Put the submodule's ``vibevoice`` subpackages on ``sys.modules``.

    Shared by both entry paths. Raises if the submodule is missing.
    """
    pkg_dir = _vibevoice_pkg_dir()
    if pkg_dir is None:
        raise ImportError(
            "The VibeVoice submodule is not checked out. Run:\n"
            "  git submodule update --init third_party/VibeVoice"
        )

    importlib.invalidate_caches()
    _register_bare_package("vibevoice", pkg_dir)
    for sub in ("modular", "schedule", "processor"):
        _register_bare_package(f"vibevoice.{sub}", os.path.join(pkg_dir, sub))
    return pkg_dir


def _import_vibevoice():
    """Import the VibeVoice entry class + config from the pinned submodule.

    Returns ``(VibeVoiceForConditionalGeneration, VibeVoiceConfig)``.
    """
    _register_vibevoice_packages()

    compat.apply_pre_import()
    from vibevoice.modular import modeling_vibevoice as _modeling
    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig

    _patch_tie_weights(_modeling.VibeVoiceForConditionalGeneration)
    return _modeling.VibeVoiceForConditionalGeneration, VibeVoiceConfig


def _import_vibevoice_tts():
    """Import the TTS inference class + processor from the pinned submodule.

    Returns ``(VibeVoiceForConditionalGenerationInference, VibeVoiceProcessor)``
    with every transformers>=5 patch already applied.
    """
    _register_vibevoice_packages()

    compat.apply_pre_import()
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    compat.patch_runtime()
    compat.patch_inference_class(VibeVoiceForConditionalGenerationInference)
    return VibeVoiceForConditionalGenerationInference, VibeVoiceProcessor


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

    # ------------------------------------------------------------------
    # Real TTS path (pretrained weights, full generate()).
    #
    # Separate from load_model()/load_inputs() above because generate() is a
    # sampling loop, not a single forward, so the generic inference harness
    # cannot drive it. Used by the e2e demo and per-component tests.
    # ------------------------------------------------------------------

    #: Prompt used by the demo and reference runs when none is given. The
    #: "Speaker 1:" prefix is required - the processor parses speaker turns out
    #: of the script and assigns each one a voice sample.
    DEFAULT_TEXT = (
        "Speaker 1: Hello from Tenstorrent. This is a VibeVoice end to end test."
    )

    @staticmethod
    def default_voice_sample():
        """Path to a voice-prompt wav shipped with the submodule.

        VibeVoice clones the timbre of a reference clip rather than selecting
        from fixed speaker IDs, so a wav is a required input. Returns ``None``
        if the submodule is not checked out.
        """
        pkg_dir = _vibevoice_pkg_dir()
        if pkg_dir is None:
            return None
        wav = os.path.join(
            os.path.dirname(pkg_dir), "demo", "voices", "en-Alice_woman.wav"
        )
        return wav if os.path.isfile(wav) else None

    def load_processor(self):
        """Load the ``VibeVoiceProcessor`` for the pretrained checkpoint.

        Handles the text tokenizer and the voice-prompt audio conditioning
        (db-normalise, tokenize into acoustic frames, build the speech masks).
        """
        _, VibeVoiceProcessor = _import_vibevoice_tts()
        return VibeVoiceProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )

    def load_tts_model(self, dtype_override=torch.float32, ddpm_steps=10):
        """Load the full TTS inference model with **pretrained** weights.

        Args:
            dtype_override: dtype for the whole model. Defaults to float32:
                the CPU reference is the PCC golden for the TT run, so it
                should not itself be quantised.
            ddpm_steps: denoise steps per acoustic frame in the diffusion head.
                Upstream's demo default is 10; more steps cost linear time.

        Returns:
            torch.nn.Module: eval-mode ``VibeVoiceForConditionalGenerationInference``.
        """
        InferenceCls, _ = _import_vibevoice_tts()

        model = InferenceCls.from_pretrained(
            self._variant_config.pretrained_model_name,
            dtype=dtype_override,
            attn_implementation="eager",
        )
        model.eval()
        model.set_ddpm_inference_steps(num_steps=ddpm_steps)
        # Buffers registered persistent=False are absent from the checkpoint,
        # and transformers 5 materialises from the checkpoint only - so they
        # come back as uninitialised memory rather than their __init__ value.
        compat.restore_nonpersistent_buffers(model)
        # Three ways this checkpoint loads into a model that generates fluent
        # audio saying nothing like the input text, none of which raise:
        # an untied lm_head, _init_weights() clobbering the speech connectors,
        # and a garbage fix_std. Assert against all of them.
        compat.assert_lm_head_tied(model)
        compat.assert_speech_stack_loaded(model)
        return model

    def load_tts_inputs(self, text=None, voice_samples=None, processor=None):
        """Build the conditioning inputs for ``load_tts_model().generate()``.

        Args:
            text: script string, ``"Speaker 1: ..."`` form. Defaults to
                :attr:`DEFAULT_TEXT`.
            voice_samples: list of voice-prompt wav paths, one per speaker.
                Defaults to the submodule's ``en-Alice_woman.wav``.
            processor: reuse an already-loaded processor; loaded on demand
                otherwise.

        Returns:
            dict: ``input_ids``, ``attention_mask``, ``speech_input_mask``,
            ``speech_tensors``, ``speech_masks``.
        """
        if processor is None:
            processor = self.load_processor()
        if text is None:
            text = self.DEFAULT_TEXT
        if voice_samples is None:
            voice = self.default_voice_sample()
            if voice is None:
                raise FileNotFoundError(
                    "No voice sample available; pass voice_samples=[<path.wav>] "
                    "or check out the VibeVoice submodule."
                )
            voice_samples = [voice]

        return processor(
            text=[text],
            voice_samples=[voice_samples],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
