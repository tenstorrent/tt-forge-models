# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ruRoPEBert model loader implementation for feature extraction.
"""
import torch
from typing import Optional

from transformers import AutoTokenizer, AutoModel


def _patch_transformers_compat():
    """Patch missing transformers functions needed by ruRoPEBert custom code.

    The ruRoPEBert model's custom code imports find_pruneable_heads_and_indices
    which was removed in transformers 5.x.
    """
    from transformers import pytorch_utils

    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned):
            mask = torch.ones(n_heads, head_size)
            for head in heads:
                if head not in already_pruned:
                    mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        pytorch_utils.find_pruneable_heads_and_indices = (
            find_pruneable_heads_and_indices
        )


from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available ruRoPEBert model variants for feature extraction."""

    TOCHKA_AI_RUROPEBERT_CLASSIC_BASE_512 = "Tochka-AI/ruRoPEBert-classic-base-512"


class ModelLoader(ForgeModel):
    """ruRoPEBert model loader implementation for feature extraction."""

    _VARIANTS = {
        ModelVariant.TOCHKA_AI_RUROPEBERT_CLASSIC_BASE_512: LLMModelConfig(
            pretrained_model_name="Tochka-AI/ruRoPEBert-classic-base-512",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TOCHKA_AI_RUROPEBERT_CLASSIC_BASE_512

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="ruRoPEBert",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            model_name = self._variant_config.pretrained_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        _patch_transformers_compat()

        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="eager",
            **model_kwargs,
        )
        model.eval()

        self._patch_extended_attention_mask(model)
        self._patch_rotary_pos_emb(model)

        self.model = model
        return model

    @staticmethod
    def _patch_extended_attention_mask(model):
        """Patch get_extended_attention_mask to produce [B,1,S,S] instead of [B,1,1,S].

        The TT-MLIR compiler requires the SDPA attention mask dim 2 to match
        the query sequence length. The default 2D→4D conversion produces
        [B,1,1,S] which has dim 2 = 1. This patch creates [B,1,S,S] so
        dim 2 = S matches the query length.
        """
        orig_fn = model.get_extended_attention_mask

        def _patched_get_extended_attention_mask(
            attention_mask, input_shape, *args, **kwargs
        ):
            if attention_mask.dim() == 2:
                batch_size, seq_length = attention_mask.shape
                mask_col = attention_mask[:, None, None, :]
                mask_row = attention_mask[:, None, :, None]
                extended = mask_row * mask_col
                extended = extended.to(dtype=model.dtype)
                extended = (1.0 - extended) * torch.finfo(model.dtype).min
                return extended
            return orig_fn(attention_mask, input_shape, *args, **kwargs)

        model.get_extended_attention_mask = _patched_get_extended_attention_mask

    @staticmethod
    def _patch_rotary_pos_emb(model):
        """Replace gather-based RoPE with slice-based to avoid unsupported gather op.

        The TT-MLIR compiler cannot lower the gather used in
        apply_rotary_pos_emb(cos[position_ids]). Since position_ids is
        always sequential [0..seq_len-1] for inference and the cos/sin
        tensors are already sliced to seq_len, we replace the gather
        with a simple unsqueeze+expand.
        """
        import sys

        model_mod = None
        for name, mod in list(sys.modules.items()):
            if (
                mod is not None
                and hasattr(mod, "apply_rotary_pos_emb")
                and "rope_bert" in name
            ):
                model_mod = mod
                break

        if model_mod is None:
            return

        rotate_half = model_mod.rotate_half

        def _apply_rotary_pos_emb_no_gather(
            q, k, cos, sin, position_ids=None, unsqueeze_dim=1
        ):
            cos = cos.unsqueeze(0).unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(0).unsqueeze(unsqueeze_dim)
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            return q_embed, k_embed

        model_mod.apply_rotary_pos_emb = _apply_rotary_pos_emb_no_gather

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            "Привет, как дела? Давай поговорим о чём-нибудь интересном.",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def output_postprocess(self, output, inputs=None):
        if inputs is None:
            inputs = self.load_inputs()

        attention_mask = inputs["attention_mask"]

        if isinstance(output, (tuple, list)):
            token_embeddings = output[0]
        elif hasattr(output, "last_hidden_state"):
            token_embeddings = output.last_hidden_state
        else:
            token_embeddings = output

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sentence_embeddings

    def decode_output(self, outputs, inputs=None):
        return self.output_postprocess(outputs, inputs=inputs)

    def unpack_forward_output(self, fwd_output):
        tensors = []

        if hasattr(fwd_output, "last_hidden_state"):
            tensors.append(fwd_output.last_hidden_state.flatten())
        if (
            hasattr(fwd_output, "pooler_output")
            and fwd_output.pooler_output is not None
        ):
            tensors.append(fwd_output.pooler_output.flatten())

        if tensors:
            return torch.cat(tensors, dim=0)
        return fwd_output
