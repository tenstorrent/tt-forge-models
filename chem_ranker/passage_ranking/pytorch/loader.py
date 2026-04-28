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
import torch.nn.functional as _F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers.models.modernbert.modeling_modernbert import (
    apply_rotary_pos_emb as _apply_rotary_pos_emb,
)
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


def _modchembert_sdpa_forward(
    module,
    *,
    qkv,
    attention_mask,
    sliding_window_mask,
    position_ids,
    local_attention,
    bs,
    dim,
    **kwargs,
):
    """Compatibility shim for the pre-transformers-5.x ModernBERT sdpa attention API.

    modeling_modchembert.py (Derify/ChemRanker-alpha-sim) was written against an older
    version of transformers that exposed MODERNBERT_ATTENTION_FUNCTION['sdpa'] with this
    calling convention. In transformers 5.x the dict was removed; this shim restores it.
    """
    # qkv shape: (bs, seq_len, 3, num_heads, head_dim)
    q, k, v = qkv.unbind(dim=2)
    q = q.transpose(1, 2)  # (bs, num_heads, seq_len, head_dim)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Pooling attention always uses global (full_attention) rotary embeddings.
    cos, sin = module.rotary_emb(q, position_ids, layer_type="full_attention")
    q, k = _apply_rotary_pos_emb(q, k, cos, sin)

    dropout = module.attention_dropout if module.training else 0.0
    attn_output = _F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attention_mask,
        dropout_p=dropout,
    )
    attn_output = attn_output.transpose(1, 2).reshape(bs, -1, dim)
    return (attn_output, None)


def _modchembert_pad_stub(inputs, indices, batch, seqlen):
    raise NotImplementedError(
        "_pad_modernbert_output is only used with flash_attention_2, "
        "which is not supported in this environment"
    )


def _modchembert_unpad_stub(inputs, attention_mask, **kwargs):
    raise NotImplementedError(
        "_unpad_modernbert_input is only used with flash_attention_2, "
        "which is not supported in this environment"
    )


def _patch_modernbert_for_modchembert():
    """Add symbols removed in transformers 5.x that modeling_modchembert.py still imports.

    The custom model code at Derify/ChemRanker-alpha-sim was written against an older
    transformers that exported MODERNBERT_ATTENTION_FUNCTION, _pad_modernbert_output,
    _unpad_modernbert_input, and PreTrainedModel._maybe_set_compile. This function patches
    the installed modules so the trust_remote_code import and forward pass succeed.
    """
    import transformers.models.modernbert.modeling_modernbert as _mb
    from transformers import PreTrainedModel

    if not hasattr(_mb, "MODERNBERT_ATTENTION_FUNCTION"):
        _mb.MODERNBERT_ATTENTION_FUNCTION = {
            "sdpa": _modchembert_sdpa_forward,
            "eager": _modchembert_sdpa_forward,
        }
        _mb._pad_modernbert_output = _modchembert_pad_stub
        _mb._unpad_modernbert_input = _modchembert_unpad_stub

    # _maybe_set_compile was removed in transformers 5.x; forward() still calls it.
    if not hasattr(PreTrainedModel, "_maybe_set_compile"):
        PreTrainedModel._maybe_set_compile = lambda self: None


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
    # Two pairs are required so that the batch output has numel() > 1; the PCC
    # evaluator returns 0.0 for single-element tensors (batch=1, num_labels=1).
    sample_pairs = [
        (
            "c1snnc1C[NH2+]Cc1cc2c(s1)CCC2",
            "c1snnc1CCC[NH2+]Cc1cc2c(s1)CCC2",
        ),
        (
            "CC(=O)Oc1ccccc1C(=O)O",
            "CC(=O)Nc1ccc(O)cc1",
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
        # Patch transformers to restore symbols removed in 5.x that the custom
        # modeling_modchembert.py still imports (MODERNBERT_ATTENTION_FUNCTION, etc.)
        _patch_modernbert_for_modchembert()

        pretrained_model_name = self._variant_config.pretrained_model_name

        # Load config separately so we can add global_rope_theta, which transformers 5.x
        # absorbs into rope_parameters via convert_rope_params_to_dict() and no longer
        # stores as a standalone attribute — but modeling_modchembert.py line 118 still
        # reads config.global_rope_theta directly.
        config = AutoConfig.from_pretrained(pretrained_model_name, trust_remote_code=True)
        if not hasattr(config, "global_rope_theta"):
            rope_params = getattr(config, "rope_parameters", {})
            config.global_rope_theta = rope_params.get("full_attention", {}).get(
                "rope_theta", 160000.0
            )
        model_kwargs = {"trust_remote_code": True, "config": config}
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
