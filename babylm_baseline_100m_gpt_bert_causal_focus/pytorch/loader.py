# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BabyLM Baseline 100M GPT-BERT Causal Focus model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available BabyLM Baseline 100M GPT-BERT Causal Focus model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """BabyLM Baseline 100M GPT-BERT Causal Focus loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.DEFAULT: LLMModelConfig(
            pretrained_model_name="BabyLM-community/babylm-baseline-100m-gpt-bert-causal-focus",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    sample_text = "This is a sample text from "

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
            model="BabyLM Baseline 100M GPT-BERT Causal Focus",
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
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "trust_remote_code": True,
            "use_cache": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

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
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        finally:
            _PTM._finalize_model_loading = _orig_finalize

        # DWAModules uses InPlaceSetSlice which calls torch.Tensor().set_() to create
        # aliased storage. torch.compile/dynamo cannot trace set_() — the FakeTensor
        # retains its empty shape, causing tensordot to see size-0 inputs.
        # Get the class directly from the model instance and replace forward/init_accumulator
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

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        vocab_size = 16384  # From model config

        input_ids = torch.cat(
            [
                torch.randint(1, vocab_size, (1, 255)),
                torch.zeros(1, 1, dtype=torch.int64),
            ],
            dim=-1,
        ).to(torch.int64)

        return {"input_ids": input_ids}

    def decode_output(self, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text."""
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        generated_ids = logits.argmax(-1)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
