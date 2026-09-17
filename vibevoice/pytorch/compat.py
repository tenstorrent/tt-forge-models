# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""transformers>=5 compatibility shims for the vendored VibeVoice source.

The vendored fork pins ``transformers==4.51.3``; tt-forge-models runs
transformers 5.x. Everything here is a runtime patch applied from the loader so
the vendored files stay unmodified, which keeps the submodule a clean upstream
checkout.

Split into three groups by when they must run:

* :func:`apply_pre_import` - before the vendored package is imported at all.
* :func:`patch_runtime` - after import, on module-level singletons.
* :func:`patch_inference_class` - after import, on the inference model class.

Every patch is idempotent, so calling these repeatedly (once per loader
instantiation) is safe.
"""

import sys
import types

# --------------------------------------------------------------------------
# pre-import
# --------------------------------------------------------------------------


def _alias_qwen2_fast_tokenizer():
    """Restore the ``tokenization_qwen2_fast`` module path removed in transformers 5.

    transformers 5 dropped the slow/fast tokenizer split: ``Qwen2Tokenizer`` is
    now itself the tokenizers-backed implementation and ``Qwen2TokenizerFast``
    survives only as a top-level deprecation alias. VibeVoice's text tokenizer
    imports the class from the old submodule path, so stand up a shim module
    exposing both names under that path.
    """
    name = "transformers.models.qwen2.tokenization_qwen2_fast"
    if name in sys.modules:
        return
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

    shim = types.ModuleType(name)
    shim.Qwen2Tokenizer = Qwen2Tokenizer
    shim.Qwen2TokenizerFast = Qwen2Tokenizer
    sys.modules[name] = shim


def patch_idempotent_register():
    """Make ``AutoModel(.ForCausalLM).register`` idempotent.

    transformers>=5 pre-registers the VibeVoice sub-configs, so the explicit
    ``AutoModel.register(...)`` calls at module import time would raise
    "already used by a Transformers model". The registrations are identical
    config->model pairs, so forcing ``exist_ok=True`` is safe.
    """
    import transformers

    for cls_name in ("AutoModel", "AutoModelForCausalLM"):
        cls = getattr(transformers, cls_name)
        if getattr(cls.register, "_tt_idempotent", False):
            continue

        def _make(orig):
            def register(config_class, model_class=None, exist_ok=False):
                return orig(config_class, model_class, exist_ok=True)

            register._tt_idempotent = True
            return staticmethod(register)

        setattr(cls, "register", _make(cls.register))


def apply_pre_import():
    """Patches that must land before the vendored package is imported."""
    _alias_qwen2_fast_tokenizer()
    patch_idempotent_register()


# --------------------------------------------------------------------------
# post-import runtime patches
# --------------------------------------------------------------------------


def patch_scheduler_real_device():
    """Build the DPM-Solver scheduler on a real device, not under meta init.

    transformers 5 constructs models inside a ``torch.device("meta")`` context.
    ``DPMSolverMultistepScheduler.__init__`` precomputes its betas/sigmas as
    plain tensors and then calls ``self.sigmas.to("cpu")``, which raises
    "Cannot copy out of meta tensor; no data!". The scheduler holds no
    parameters and is not part of the checkpoint, so forcing its construction
    onto CPU is safe.
    """
    import torch

    from vibevoice.schedule.dpm_solver import DPMSolverMultistepScheduler

    orig = DPMSolverMultistepScheduler.__init__
    if getattr(orig, "_tt_real_device", False):
        return

    def __init__(self, *args, **kwargs):
        with torch.device("cpu"):
            return orig(self, *args, **kwargs)

    __init__._tt_real_device = True
    DPMSolverMultistepScheduler.__init__ = __init__


def patch_get_text_config():
    """Point ``VibeVoiceConfig.get_text_config()`` at the Qwen decoder config.

    transformers 5 builds the generation cache from
    ``self.config.get_text_config(decoder=True)``, which recognises a nested
    text config only under the names ``decoder``/``generator``/``text_config``.
    VibeVoice calls its own ``decoder_config``, so the base implementation
    falls through to the composite config and ``DynamicCache`` dies on
    ``'VibeVoiceConfig' object has no attribute 'num_hidden_layers'``.
    """
    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig

    if getattr(VibeVoiceConfig.get_text_config, "_tt_decoder_config", False):
        return

    def get_text_config(self, decoder=None, encoder=None):
        return self.decoder_config

    get_text_config._tt_decoder_config = True
    VibeVoiceConfig.get_text_config = get_text_config


def patch_dynamic_cache_legacy_views():
    """Re-expose ``DynamicCache.key_cache`` / ``.value_cache``.

    transformers 5 moved the per-layer tensors behind ``cache.layers[i].keys``
    and ``.values``. VibeVoice's CFG path walks
    ``zip(past_key_values.key_cache, past_key_values.value_cache)`` and mutates
    the entries in place to fix up the negative-branch KV cache. The properties
    return the live tensor objects, so in-place writes still land on the cache.
    """
    from transformers.cache_utils import DynamicCache

    if hasattr(DynamicCache, "key_cache"):
        return

    DynamicCache.key_cache = property(
        lambda self: [layer.keys for layer in self.layers if layer.keys is not None]
    )
    DynamicCache.value_cache = property(
        lambda self: [layer.values for layer in self.layers if layer.values is not None]
    )


def patch_init_weights_respects_loaded():
    """Stop ``_init_weights`` from overwriting weights already loaded.

    The single worst defect on this path, and it raises nothing.

    ``VibeVoicePreTrainedModel._init_weights`` re-randomises any ``nn.Linear``
    it is handed::

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

    transformers 4.51 only ever called it on modules the checkpoint did not
    cover. transformers 5 calls it on *every* module after loading and expects
    each ``_init_weights`` implementation to honour the ``_is_hf_initialized``
    marker that the loader stamps onto restored parameters. The vendored
    implementation predates that contract, so under transformers 5 it clobbers
    the two ``SpeechConnector`` blocks — ``model.{acoustic,semantic}_connector``
    ``fc1``/``fc2`` weights back to N(0, 0.02) and their biases to exactly zero.

    Every other submodule survives because it is built through
    ``AutoModel.from_config`` and so carries its own ``_init_weights``; the
    connectors are the only plain ``nn.Module`` children of ``VibeVoiceModel``.
    Their ``norm`` survives too, since ``LlamaRMSNorm`` matches neither
    ``nn.Linear`` nor ``nn.LayerNorm``.

    The consequence is that the voice-prompt latents are projected into the LM's
    embedding space by an untrained matrix, so the conditioning carries no
    speaker or content information. Generation still produces fluent speech —
    it just has nothing to do with the prompt, which is why it survived every
    signal-level check. See :func:`assert_speech_stack_loaded`.
    """
    from vibevoice.modular.modeling_vibevoice import VibeVoicePreTrainedModel

    orig = VibeVoicePreTrainedModel._init_weights
    if getattr(orig, "_tt_respects_loaded", False):
        return

    def _init_weights(self, module):
        params = list(module.parameters(recurse=False))
        if params and all(
            getattr(p, "_is_hf_initialized", False) for p in params if p is not None
        ):
            return
        return orig(self, module)

    _init_weights._tt_respects_loaded = True
    VibeVoicePreTrainedModel._init_weights = _init_weights


def patch_runtime():
    """Post-import patches that apply to module-level singletons."""
    patch_scheduler_real_device()
    patch_get_text_config()
    patch_dynamic_cache_legacy_views()
    patch_init_weights_respects_loaded()


# --------------------------------------------------------------------------
# inference-class patches
# --------------------------------------------------------------------------


def patch_tie_weights(cls):
    """Restore lm_head <-> embed_tokens tying, which silently breaks under tf5.

    Two separate problems, and the second one corrupts output rather than
    raising:

    1. transformers>=5 calls ``tie_weights(recompute_mapping=..., missing_keys=...)``
       while the vendored method declares ``tie_weights(self)``.
    2. The vendored guard reads ``self.config.tie_word_embeddings``, but
       transformers 5 dropped that attribute from ``PreTrainedConfig`` and
       ``VibeVoiceConfig`` never sets it, so the lookup yields ``None`` and the
       method returns without tying. VibeVoice-1.5B ships no ``lm_head.weight``
       in its checkpoint (it is tied), so the head stays at its random
       initialisation and generation produces noise with no error raised. The
       real flag lives on ``config.decoder_config.tie_word_embeddings``.

    Callers must assert the tie actually took - see
    :func:`assert_lm_head_tied`.
    """
    if getattr(cls.__dict__.get("tie_weights"), "_tt_decoder_tying", False):
        return

    def tie_weights(self, *args, **kwargs):
        decoder_config = getattr(self.config, "decoder_config", None)
        if not getattr(decoder_config, "tie_word_embeddings", False):
            return
        embed_tokens = getattr(self.model.language_model, "embed_tokens", None)
        if embed_tokens is None or not hasattr(self, "lm_head"):
            return
        self.lm_head.weight = embed_tokens.weight

    tie_weights._tt_decoder_tying = True
    cls.tie_weights = tie_weights


def assert_lm_head_tied(model):
    """Fail loudly if ``lm_head`` did not end up sharing the embedding table.

    An untied head is not an error anywhere in the stack - it is simply random,
    and the generated audio is noise. Compare storage identity rather than
    trusting the config.
    """
    embed = model.get_input_embeddings().weight
    head = model.lm_head.weight
    if head.data_ptr() != embed.data_ptr():
        raise RuntimeError(
            "lm_head is not tied to embed_tokens: the head is at its random "
            "initialisation and generation would be noise. VibeVoice-1.5B ships "
            "no lm_head.weight in its checkpoint, so tying is mandatory."
        )


def restore_nonpersistent_buffers(model):
    """Re-materialise buffers registered ``persistent=False``.

    The acoustic tokenizer holds its VAE sampling scale as::

        self.register_buffer("fix_std", torch.tensor(config.fix_std),
                             persistent=False)

    Being non-persistent it is absent from the checkpoint by design, and
    transformers 5 builds models under a ``torch.device("meta")`` context and
    then materialises them from the checkpoint only. Nothing re-runs
    ``__init__``, so this buffer is handed back as *uninitialised memory*: it
    read 8986389.0, 4.3e-17 and 2.6e-06 on three consecutive loads of the same
    checkpoint, against a true value of 0.5.

    It is not a dead field. ``std_dist_type`` is ``"gaussian"`` for this
    checkpoint, so the acoustic encode does::

        std = torch.randn(batch) * (self.std / 0.8)
        latents = mean + std * torch.randn_like(mean)

    which scales the noise added to the voice-prompt latents by the garbage
    value. That both destroys the conditioning and makes every run differ,
    which is the run-to-run variation in generated length seen under a fixed
    seed.

    Reset from the config, which is where the true value lives.
    """
    import torch

    for name in ("acoustic_tokenizer", "semantic_tokenizer"):
        tok = getattr(model.model, name, None)
        buf = getattr(tok, "fix_std", None) if tok is not None else None
        if buf is None:
            continue
        want = getattr(tok.config, "fix_std", None)
        if want is None:
            continue
        tok.fix_std = torch.tensor(want, dtype=buf.dtype, device=buf.device)


def assert_speech_stack_loaded(model):
    """Fail loudly if the speech conditioning path is not the trained one.

    Guards the two defects that produce fluent-but-unrelated audio with nothing
    raised anywhere in the stack — see
    :func:`patch_init_weights_respects_loaded` and
    :func:`restore_nonpersistent_buffers`. Both are cheap to check and neither
    is detectable from the generated waveform's signal statistics.
    """
    for name in ("acoustic_connector", "semantic_connector"):
        connector = getattr(model.model, name)
        for fc_name in ("fc1", "fc2"):
            bias = getattr(connector, fc_name).bias
            if bias is not None and not bias.any():
                raise RuntimeError(
                    f"{name}.{fc_name}.bias is all zeros, which is the signature of "
                    "_init_weights() having overwritten the checkpoint. The speech "
                    "connectors are at their random initialisation, so the voice "
                    "prompt conditions nothing and the generated audio will be "
                    "fluent speech unrelated to the input text."
                )

    for name in ("acoustic_tokenizer", "semantic_tokenizer"):
        tok = getattr(model.model, name, None)
        buf = getattr(tok, "fix_std", None) if tok is not None else None
        want = getattr(tok.config, "fix_std", None) if tok is not None else None
        if buf is None or want is None:
            continue
        if abs(float(buf) - float(want)) > 1e-9:
            raise RuntimeError(
                f"{name}.fix_std is {float(buf)!r} but the config says {want!r}. "
                "This buffer is registered persistent=False, so transformers 5 "
                "hands back uninitialised memory; it scales the noise added to "
                "the voice-prompt latents."
            )


def patch_prepare_inputs_for_generation(cls):
    """Guarantee ``inputs_embeds`` is present in the prepared-inputs dict.

    transformers 4 always emitted the key (possibly ``None``); transformers 5
    omits it when unused. VibeVoice's CFG branch reads
    ``negative_model_inputs['inputs_embeds']`` unconditionally and raises
    ``KeyError``.
    """
    orig = cls.prepare_inputs_for_generation
    if getattr(orig, "_tt_embeds_key", False):
        return

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_inputs = orig(self, *args, **kwargs)
        model_inputs.setdefault("inputs_embeds", None)
        return model_inputs

    prepare_inputs_for_generation._tt_embeds_key = True
    cls.prepare_inputs_for_generation = prepare_inputs_for_generation


def patch_generation_mixin(cls):
    """Adapt the two GenerationMixin helpers whose signatures changed in tf5.

    * ``_prepare_generation_config`` lost the positional ``use_model_defaults``
      flag that the vendored code passes as ``True``.
    * ``_prepare_cache_for_generation`` lost its trailing ``device`` argument.

    Both are re-exposed on the model class with the old calling convention,
    forwarding to the current implementation.
    """
    from transformers.generation import GenerationMixin

    base_prepare_config = GenerationMixin._prepare_generation_config

    def _prepare_generation_config(self, generation_config, *args, **kwargs):
        return base_prepare_config(self, generation_config, **kwargs)

    cls._prepare_generation_config = _prepare_generation_config

    base_prepare_cache = GenerationMixin._prepare_cache_for_generation

    def _prepare_cache_for_generation(
        self,
        generation_config,
        model_kwargs,
        generation_mode,
        batch_size,
        max_cache_length,
        device=None,
    ):
        return base_prepare_cache(
            self,
            generation_config,
            model_kwargs,
            generation_mode,
            batch_size,
            max_cache_length,
        )

    cls._prepare_cache_for_generation = _prepare_cache_for_generation


def patch_inference_class(cls):
    """Apply every patch that targets the TTS inference model class."""
    patch_tie_weights(cls)
    patch_generation_mixin(cls)
    patch_prepare_inputs_for_generation(cls)
