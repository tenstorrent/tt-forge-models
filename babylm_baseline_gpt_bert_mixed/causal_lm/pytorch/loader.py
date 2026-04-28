# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BabyLM Baseline GPT-BERT Mixed model loader implementation for causal language modeling.
"""

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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


class ModelVariant(StrEnum):
    """Available BabyLM Baseline GPT-BERT Mixed model variants."""

    BABYLM_100M = "100M"


class ModelLoader(ForgeModel):
    """BabyLM Baseline GPT-BERT Mixed model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.BABYLM_100M: ModelConfig(
            pretrained_model_name="BabyLM-community/babylm-baseline-100m-gpt-bert-mixed",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BABYLM_100M

    sample_text = (
        "In a shocking finding, scientists discovered a herd of unicorns living in"
    )

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BabyLM Baseline GPT-BERT Mixed",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BabyLM GPT-BERT Mixed model instance."""
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "trust_remote_code": True,
        }
        model_kwargs |= kwargs

        if self.num_layers is not None:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        _original_default = json.JSONEncoder.default

        def _dtype_aware_default(self, obj):
            if isinstance(obj, torch.dtype):
                return str(obj)
            return _original_default(self, obj)

        import torch.nn as nn
        from transformers import PreTrainedModel as _PTM

        _orig_finalize = _PTM._finalize_model_loading

        @staticmethod
        def _patched_finalize(model, load_config, loading_info):
            # GPTBERTForCausalLM skips self.post_init() (transformers 5.x requirement),
            # so all_tied_weights_keys is never initialised. Seed it to an empty dict so
            # _adjust_tied_keys_with_tied_pointers can populate it via pointer detection.
            if not hasattr(model, "all_tied_weights_keys"):
                model.all_tied_weights_keys = {}
            for m in model.modules():
                # _init_weights unconditionally accesses LayerNorm.bias, but layers
                # using elementwise_affine=False have no bias. Pre-mark as initialised
                # so _initialize_missing_keys skips them.
                if isinstance(m, nn.LayerNorm) and not m.elementwise_affine:
                    m._is_hf_initialized = True
            result = _orig_finalize(model, load_config, loading_info)
            # _move_missing_keys_from_meta_to_device fills all non-persistent buffers
            # with torch.empty_like (garbage). Recompute position_indices after the
            # finalize is done. The Attention config attribute stores what we need.
            for m in model.modules():
                if (
                    hasattr(m, "position_indices")
                    and hasattr(m, "make_log_bucket_position")
                    and hasattr(m, "config")
                ):
                    cfg = m.config
                    pos_idx = (
                        torch.arange(cfg.max_position_embeddings, dtype=torch.long).unsqueeze(1)
                        - torch.arange(cfg.max_position_embeddings, dtype=torch.long).unsqueeze(0)
                    )
                    pos_idx = m.make_log_bucket_position(
                        pos_idx, cfg.position_bucket_size, cfg.max_position_embeddings
                    )
                    pos_idx = cfg.position_bucket_size - 1 + pos_idx
                    m.register_buffer("position_indices", pos_idx, persistent=False)
            return result

        _PTM._finalize_model_loading = _patched_finalize
        json.JSONEncoder.default = _dtype_aware_default
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        finally:
            json.JSONEncoder.default = _original_default
            _PTM._finalize_model_loading = _orig_finalize

        # DWAModules uses InPlaceSetSlice which calls torch.Tensor().set_() to create
        # aliased storage. torch.compile/dynamo cannot trace set_() — the FakeTensor
        # retains its empty shape, causing tensordot to see size-0 inputs.
        # sys.modules search is unreliable because torch.ops and torch.classes also
        # expose a DWAModules attribute and appear earlier in iteration order.  Get the
        # class directly from the model instance and replace forward/init_accumulator
        # with cat-based implementations that contain no reference to InPlaceSetSlice.
        _DWA = type(model.model.dwa_modules)

        def _dwa_init_accumulator(self, x):
            self.accumulator = (None, x.unsqueeze(0))

        def _dwa_forward(self, x, block_idx):
            assert self.accumulator is not None, "call init_accumulator first"
            _, last_slice = self.accumulator
            v = x.unsqueeze(0)
            new_slice = torch.cat([last_slice, v], dim=0)
            self.accumulator = (None, new_slice)
            return torch.tensordot(self.alphas[block_idx], new_slice, dims=1)

        _DWA.init_accumulator = _dwa_init_accumulator
        _DWA.forward = _dwa_forward

        model.eval()

        if self.tokenizer is None:
            self._load_tokenizer()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model."""
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=512,
            padding=True,
            truncation=True,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return dict(inputs)

    def decode_output(self, outputs, inputs=None):
        """Decode model outputs into human-readable text."""
        if self.tokenizer is None:
            self._load_tokenizer()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        token_ids = torch.argmax(logits, dim=-1)
        decoded = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return decoded[0] if len(decoded) == 1 else decoded
