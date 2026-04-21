# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ChemMRL model loader implementation for molecular embedding generation.

Includes compatibility shims for the model's custom HuggingFace code
(written for transformers 4.x) to work with transformers 5.x.
"""
import torch
from typing import Optional, Tuple

import transformers.models.modernbert.modeling_modernbert as _modernbert_module

# --- Transformers 5.x backward-compatibility shims for ModernBert ---
# The Derify/ChemMRL custom model code targets the transformers 4.x internal API.

if not hasattr(_modernbert_module, "MODERNBERT_ATTENTION_FUNCTION"):
    _modernbert_module.MODERNBERT_ATTENTION_FUNCTION = (
        _modernbert_module.ALL_ATTENTION_FUNCTIONS
    )

if not hasattr(_modernbert_module.ModernBertPreTrainedModel, "_maybe_set_compile"):
    _modernbert_module.ModernBertPreTrainedModel._maybe_set_compile = lambda self: None

if not hasattr(_modernbert_module, "_unpad_modernbert_input"):

    def _unpad_modernbert_input(
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = int(seqlens_in_batch.max().item())
        cu_seqlens = torch.nn.functional.pad(
            torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
        )
        if inputs.dim() == 2:
            unpadded_inputs = inputs.flatten()[indices]
        else:
            batch, seqlen, *rest = inputs.shape
            shape = batch * seqlen
            unpadded_inputs = inputs.view(shape, *rest)[indices]
        unpadded_position_ids = (
            position_ids.flatten()[indices] if position_ids is not None else None
        )
        unpadded_labels = labels.flatten()[indices] if labels is not None else None
        return (
            unpadded_inputs,
            indices,
            cu_seqlens,
            max_seqlen_in_batch,
            unpadded_position_ids,
            unpadded_labels,
        )

    _modernbert_module._unpad_modernbert_input = _unpad_modernbert_input

if not hasattr(_modernbert_module, "_pad_modernbert_output"):

    def _pad_modernbert_output(
        inputs: torch.Tensor,
        indices: torch.Tensor,
        batch: int,
        seqlen: int,
    ) -> torch.Tensor:
        if inputs.dim() == 1:
            output = torch.zeros(
                batch * seqlen, dtype=inputs.dtype, device=inputs.device
            )
            output[indices] = inputs
            padded_inputs = output.view(batch, seqlen)
        else:
            _, *rest = inputs.shape
            output = torch.zeros(
                batch * seqlen, *rest, dtype=inputs.dtype, device=inputs.device
            )
            output[indices] = inputs
            padded_inputs = output.view(batch, seqlen, *rest)
        return padded_inputs

    _modernbert_module._pad_modernbert_output = _pad_modernbert_output

# Patch ModernBertEncoderLayer.forward to accept old-style (4.x) arguments
# and convert them to the new (5.x) API.
_original_encoder_layer_forward = _modernbert_module.ModernBertEncoderLayer.forward


def _compat_encoder_layer_forward(
    self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs
):
    position_ids = kwargs.pop("position_ids", None)
    sliding_window_mask = kwargs.pop("sliding_window_mask", None)
    kwargs.pop("cu_seqlens", None)
    kwargs.pop("max_seqlen", None)
    kwargs.pop("output_attentions", None)

    old_style = position_ids is not None

    if position_embeddings is None and position_ids is not None:
        rotary_emb = getattr(self, "_compat_rotary_emb", None)
        if rotary_emb is not None:
            layer_type = getattr(self, "attention_type", "full_attention")
            position_embeddings = rotary_emb(hidden_states, position_ids, layer_type)

    if sliding_window_mask is not None:
        layer_type = getattr(self, "attention_type", "full_attention")
        if layer_type == "sliding_attention":
            attention_mask = sliding_window_mask

    result = _original_encoder_layer_forward(
        self,
        hidden_states,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
        **kwargs,
    )

    if old_style and isinstance(result, torch.Tensor):
        return (result,)
    return result


_modernbert_module.ModernBertEncoderLayer.forward = _compat_encoder_layer_forward

from transformers import AutoModel, AutoTokenizer

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
    """Available ChemMRL model variants."""

    CHEM_MRL = "Derify/ChemMRL"


class ModelLoader(ForgeModel):
    """ChemMRL model loader for molecular embedding generation."""

    _VARIANTS = {
        ModelVariant.CHEM_MRL: ModelConfig(
            pretrained_model_name="Derify/ChemMRL",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CHEM_MRL

    sample_sentences = [
        "CCO",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="ChemMRL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)

        if not hasattr(model, "rotary_emb"):
            model.rotary_emb = _modernbert_module.ModernBertRotaryEmbedding(
                config=model.config
            )
        for layer in model.layers:
            layer._compat_rotary_emb = model.rotary_emb

        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_sentences,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

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
