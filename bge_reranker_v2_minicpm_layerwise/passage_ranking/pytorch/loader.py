# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BGE Reranker v2 MiniCPM Layerwise model loader implementation for passage ranking.
"""
import torch
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
    """Available BGE Reranker v2 MiniCPM Layerwise model variants for passage ranking."""

    LAYERWISE = "layerwise"


class ModelLoader(ForgeModel):
    """BGE Reranker v2 MiniCPM Layerwise model loader implementation for passage ranking.

    This reranker uses a MiniCPM-2B causal LM backbone with layerwise scoring
    (selectable cutoff layers 8-40) to determine query-passage relevance. The
    last-token logit of the final selected layer is used as the relevance score.
    """

    _VARIANTS = {
        ModelVariant.LAYERWISE: ModelConfig(
            pretrained_model_name="BAAI/bge-reranker-v2-minicpm-layerwise",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LAYERWISE

    # Sample query-passage pairs for testing
    sample_pairs = [
        ("what is panda?", "hi"),
        (
            "what is panda?",
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
        ),
    ]

    _PROMPT_TEMPLATE = "A: {query}\nB: {passage}\nPredict whether passage B contains an answer to query A."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="BGE-Reranker-v2-MiniCPM-Layerwise",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoConfig, AutoModelForCausalLM

        # transformers 5.x removed is_torch_fx_available; the cached remote code
        # still imports it at module level. Provide a compatibility shim before
        # the dynamic module is loaded.
        import transformers.utils.import_utils as _iu
        if not hasattr(_iu, "is_torch_fx_available"):
            _iu.is_torch_fx_available = lambda: True

        pretrained_model_name = self._variant_config.pretrained_model_name

        # transformers 5.x normalizes rope_scaling=null in config.json to
        # {'rope_type': 'default', ...}.  The custom model code (trust_remote_code)
        # expects None or {'type': ...} (transformers 4.x format), causing
        # KeyError: 'type' during _init_rope.  Load config and translate.
        config = AutoConfig.from_pretrained(pretrained_model_name, trust_remote_code=True)
        if isinstance(config.rope_scaling, dict) and "type" not in config.rope_scaling:
            rope_type = config.rope_scaling.get("rope_type", "default")
            if rope_type == "default":
                config.rope_scaling = None
            else:
                config.rope_scaling = dict(config.rope_scaling, type=rope_type)

        # transformers 5.x changed _tied_weights_keys from list to dict format.
        # This model's class attribute is a list (4.x format), which causes
        # AttributeError in post_init → get_expanded_tied_weights_keys.
        # For head_type=simple the lm_head is a ModuleList with no vocab tying,
        # so disabling tie_word_embeddings is correct.
        config.tie_word_embeddings = False

        model_kwargs = {"trust_remote_code": True, "config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            **model_kwargs,
        )

        # transformers>=5.0 uses meta device during from_pretrained, leaving
        # non-persistent buffers (inv_freq, cos_cached, sin_cached) uninitialized.
        # Reinitialize them explicitly to get valid RoPE values.
        import torch
        for module in model.modules():
            if (
                hasattr(module, "inv_freq")
                and hasattr(module, "base")
                and hasattr(module, "dim")
                and hasattr(module, "_set_cos_sin_cache")
            ):
                inv_freq = 1.0 / (
                    module.base
                    ** (torch.arange(0, module.dim, 2).float() / module.dim)
                )
                module.register_buffer("inv_freq", inv_freq, persistent=False)
                module._set_cos_sin_cache(
                    seq_len=module.max_seq_len_cached,
                    device=inv_freq.device,
                    dtype=torch.float32,
                )

        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        texts = [
            self._PROMPT_TEMPLATE.format(query=query, passage=passage)
            for query, passage in self.sample_pairs
        ]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.float32:
                        inputs[key] = value.to(dtype_override)

        return inputs
