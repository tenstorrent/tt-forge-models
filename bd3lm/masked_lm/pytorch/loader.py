# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BD3LM (Block Discrete Denoising Diffusion Language Model) loader
implementation for masked language modeling.
"""
import torch
from typing import Optional

from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available BD3LM model variants."""

    BD3LM_OWT_BLOCK_SIZE_4 = "owt-block_size4"


class ModelLoader(ForgeModel):
    """BD3LM model loader implementation for masked language modeling."""

    # BD3LM uses the gpt2 tokenizer (vocab_size 50258 = gpt2 50257 + 1 mask token).
    _TOKENIZER_NAME = "gpt2"

    _VARIANTS = {
        ModelVariant.BD3LM_OWT_BLOCK_SIZE_4: LLMModelConfig(
            pretrained_model_name="kuleshov-group/bd3lm-owt-block_size4",
            # BD3LM uses cross attention between noised and target blocks, which
            # expects an input length of 2 * model_length (2 * 1024).
            max_length=2048,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BD3LM_OWT_BLOCK_SIZE_4

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BD3LM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self._TOKENIZER_NAME)
        # GPT-2 tokenizer has no pad token by default; BD3LM adds one
        # (vocab_size = 50257 + 1 mask token).  Use eos_token as pad so
        # tokenizer padding calls work correctly.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    @staticmethod
    def _patch_bd3lm_post_init(pretrained_model_name: str) -> None:
        """Apply two patches needed to load and compile BD3LM on transformers 5.x.

        Patch 1 — post_init():
          Transformers 5.x requires every PreTrainedModel subclass to call
          self.post_init() at the end of __init__ to initialise
          all_tied_weights_keys (and other fields).  The BD3LM remote-code
          class was written for an older transformers API and does not do so,
          causing AttributeError in _finalize_model_loading.

        Patch 2 — modulate_fused():
          BD3LM defines modulate_fused as a @torch.jit.script function that
          internally calls a module-level modulate() helper.  The module also
          defines a *second* modulate() at a later line with an extra
          .unsqueeze(1) call, which shadows the first.  torch.compile/dynamo
          traces through the ScriptFunction via Python name lookup and picks
          up the shadowing definition, producing a 4-D tensor where a 3-D one
          is expected, which breaks einops.rearrange inside get_qkv.
          Replace modulate_fused with a plain Python wrapper that uses the
          correct (no-unsqueeze) modulate logic so dynamo can trace it."""
        import sys

        try:
            BD3LM = get_class_from_dynamic_module(
                "modeling_bd3lm.BD3LM",
                pretrained_model_name,
            )
        except Exception:
            # If the class cannot be loaded (e.g. no network), skip patching
            # and let from_pretrained surface the real error.
            return

        if getattr(BD3LM, "_post_init_patched_for_transformers5", False):
            return

        # Patch 1: add post_init() call to __init__
        original_init = BD3LM.__init__

        def _patched_init(self, config):
            original_init(self, config)
            if not hasattr(self, "all_tied_weights_keys"):
                self.post_init()

        BD3LM.__init__ = _patched_init

        # Patch 2: replace modulate_fused with a plain Python function that
        # correctly implements x * (1 + scale) + shift (no unsqueeze).
        mod_name = BD3LM.__module__
        if mod_name in sys.modules:
            bd3lm_mod = sys.modules[mod_name]

            def _modulate_fused_python(
                x: torch.Tensor,
                shift: torch.Tensor,
                scale: torch.Tensor,
            ) -> torch.Tensor:
                # Correct implementation: scale is already the right shape.
                # Do NOT call .unsqueeze(1) here; that extra dim comes from
                # the shadowing modulate() at the bottom of modeling_bd3lm.py.
                return x * (1 + scale) + shift

            bd3lm_mod.modulate_fused = _modulate_fused_python

            # Patch 3: Rotary embedding device-aware cache.
            # Rotary.forward() caches cos/sin by sequence length only, not by
            # device.  After the CPU reference run the cache holds CPU tensors.
            # When torch.compile then traces the model for the XLA device with
            # the same seq_len, the old CPU cache is returned, causing a
            # "found two different devices xla:0, cpu" error in FakeTensor
            # propagation.  Patch forward() to additionally check the current
            # device before reusing the cache.
            Rotary = bd3lm_mod.Rotary
            _orig_rotary_forward = Rotary.forward

            def _device_aware_rotary_forward(self, x, seq_dim=1):
                seq_len = x.shape[seq_dim]
                current_device = x.device
                if (
                    seq_len != self.seq_len_cached
                    or self.cos_cached is None
                    or self.cos_cached.device != current_device
                ):
                    # Force recalculation with the current device
                    self.seq_len_cached = None  # invalidate cache
                return _orig_rotary_forward(self, x, seq_dim)

            Rotary.forward = _device_aware_rotary_forward

        BD3LM._post_init_patched_for_transformers5 = True

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        # Transformers 5.x requires PreTrainedModel.__init__ to call
        # self.post_init() so that all_tied_weights_keys is set before
        # _finalize_model_loading runs.  Patch the remote-code class.
        self._patch_bd3lm_post_init(self._variant_config.pretrained_model_name)

        # BD3LM's flex attention backend requires CUDA and recent PyTorch;
        # override to sdpa so the loader works on CPU/XLA runtimes.
        model_kwargs = {"attn_backend": "sdpa"}
        # BD3LM's TimestepEmbedder.timestep_embedding() is hardcoded to
        # produce float32 output (t.float() * freqs[None]) regardless of the
        # model dtype.  Passing bfloat16 here would make the MLP weights
        # bfloat16 while the embedding stays float32, causing a dtype mismatch
        # on F.linear.  Keep float32 as the model dtype.
        if dtype_override is not None and dtype_override != torch.float32:
            # Do not forward bfloat16 dtype_override to this model
            pass
        elif dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )

        # Transformers 5.x initialises the model on meta device during
        # from_pretrained's lazy-load phase.  DITBackbone.gen_mask() sets
        # self.mask as a plain attribute (not register_buffer), so it is
        # never materialised from meta.  Re-generate it now that the model
        # is on CPU with real data.
        backbone = model.backbone
        if hasattr(backbone, "mask") and backbone.mask.is_meta:
            backbone.gen_mask(
                backbone.n,
                backbone.block_size,
                attn_backend=backbone.config.attn_backend,
            )

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            "The quick brown fox jumps over the lazy dog.",
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        # timesteps is a continuous-valued noise level (sigma).  The model's
        # TimestepEmbedder.timestep_embedding() always casts to float32
        # internally, so the MLP weights remain float32 regardless of
        # dtype_override.  Always create timesteps in float32 to match.
        timesteps = torch.zeros(1, dtype=torch.float32)

        return {
            "input_ids": inputs["input_ids"],
            "timesteps": timesteps,
        }

    def decode_output(self, outputs):
        if self.tokenizer is None:
            self._load_tokenizer()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        predicted_ids = logits.argmax(dim=-1)
        decoded = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        return decoded
