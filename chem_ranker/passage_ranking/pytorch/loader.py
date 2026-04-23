# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ChemRanker model loader implementation for SMILES pair reranking.

ChemRanker is a Cross Encoder built on the ModChemBERT chemical language model
and fine-tuned for molecular reranking. Given a pair of SMILES strings it emits
a relevance score suitable for reranking candidate molecules against an anchor.

Reference: https://huggingface.co/Derify/ChemRanker-alpha-sim
"""
import torch
import transformers.models.modernbert.modeling_modernbert as _modernbert
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.models.modernbert.configuration_modernbert import (
    ModernBertConfig as _ModernBertConfig,
)
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertPreTrainedModel as _ModernBertPreTrainedModel,
)
from typing import Optional

# Compatibility shims for API changes in transformers 5.x.
# MODERNBERT_ATTENTION_FUNCTION was a dict of old-style attention functions that accepted a
# combined qkv tensor.  ALL_ATTENTION_FUNCTIONS uses a new signature (query, key, value
# separately, heads already transposed).  We provide a wrapper that adapts the old
# calling convention to plain torch SDPA so the frozen cached model file still works.
if not hasattr(_modernbert, "MODERNBERT_ATTENTION_FUNCTION"):
    import torch.nn.functional as _F

    def _compat_sdpa_attn_forward(
        module,
        qkv=None,
        attention_mask=None,
        sliding_window_mask=None,
        position_ids=None,
        local_attention=(-1, -1),
        bs=None,
        dim=None,
        **kwargs,
    ):
        # qkv: (bs, seq_len, 3, num_heads, head_dim)
        query, key, value = qkv.unbind(dim=2)
        # -> (bs, num_heads, seq_len, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if hasattr(module, "rotary_emb") and position_ids is not None:
            layer_types = getattr(module.rotary_emb, "layer_types", [])
            layer_type = next(
                (
                    lt
                    for lt in layer_types
                    if "full" in lt.lower() or "global" in lt.lower()
                ),
                (layer_types[0] if layer_types else "full_attention"),
            )
            cos, sin = module.rotary_emb(query, position_ids, layer_type=layer_type)
            query, key = _modernbert.apply_rotary_pos_emb(
                query, key, cos, sin, unsqueeze_dim=1
            )

        attn_out = _F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=query.shape[-1] ** -0.5,
        )
        # -> (bs, seq_len, dim)
        bs_actual, _, seq_len, _ = attn_out.shape
        attn_out = attn_out.transpose(1, 2).contiguous().view(bs_actual, seq_len, -1)
        return (attn_out,)

    _modernbert.MODERNBERT_ATTENTION_FUNCTION = {"sdpa": _compat_sdpa_attn_forward}

if not hasattr(_modernbert, "_unpad_modernbert_input"):

    def _unpad_modernbert_input(inputs, attention_mask, position_ids=None, labels=None):
        raise NotImplementedError("flash_attention_2 unpadding not supported")

    _modernbert._unpad_modernbert_input = _unpad_modernbert_input

if not hasattr(_modernbert, "_pad_modernbert_output"):

    def _pad_modernbert_output(inputs, indices, batch, seqlen):
        raise NotImplementedError("flash_attention_2 padding not supported")

    _modernbert._pad_modernbert_output = _pad_modernbert_output

# global_rope_theta was replaced by default_theta dict in transformers 5.x
if not hasattr(_ModernBertConfig, "global_rope_theta"):

    @property
    def _global_rope_theta(self):
        dt = getattr(self, "default_theta", {})
        if isinstance(dt, dict):
            return dt.get("global", 160000.0)
        return float(dt)

    _ModernBertConfig.global_rope_theta = _global_rope_theta

# _maybe_set_compile was removed from ModernBertPreTrainedModel in transformers 5.x
if not hasattr(_ModernBertPreTrainedModel, "_maybe_set_compile"):
    _ModernBertPreTrainedModel._maybe_set_compile = lambda self: None

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
    """Available ChemRanker model variants for SMILES pair reranking."""

    ALPHA_SIM = "alpha-sim"


class ModelLoader(ForgeModel):
    """ChemRanker model loader implementation for SMILES pair reranking."""

    _VARIANTS = {
        ModelVariant.ALPHA_SIM: ModelConfig(
            pretrained_model_name="Derify/ChemRanker-alpha-sim",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ALPHA_SIM

    # Sample SMILES pairs (anchor, candidate) for reranking.
    sample_pairs = [
        (
            "c1snnc1C[NH2+]Cc1cc2c(s1)CCC2",
            "c1snnc1CCC[NH2+]Cc1cc2c(s1)CCC2",
        ),
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ChemRanker",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False, "trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        anchors = [pair[0] for pair in self.sample_pairs]
        candidates = [pair[1] for pair in self.sample_pairs]

        inputs = self.tokenizer(
            anchors,
            candidates,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
