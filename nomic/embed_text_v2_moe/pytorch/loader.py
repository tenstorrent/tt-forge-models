# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nomic Embed Text v2 MoE model loader implementation for sentence embedding generation.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
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


def _nomic_experts_forward(self, x, weights, top_weights, top_experts):
    """Device-friendly NomicExperts forward that avoids .tolist() / dynamic-shape ops.

    CPU: original per-expert loop (matches golden reference).
    Device (xla/cuda): batched-mm path with static shapes (no device-to-host transfer).
    """
    bsz, q_len, hidden_size = x.shape
    T = bsz * q_len
    K = top_experts.shape[-1]   # top_k  (= 2 for nomic-embed-text-v2-moe)
    E = self.moe_num_experts    # 8
    F_size = self.mlp.ffn_hidden_size  # 3072

    x_flat = x.view(T, hidden_size)  # [T, H]

    if x_flat.device.type == "cpu":
        # Original per-expert loop used for CPU golden reference
        out = torch.zeros_like(x_flat)
        expert_mask = F.one_hot(top_experts, num_classes=E).permute(2, 1, 0)
        for expert_idx in range(E):
            topk_idx, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.shape[0] == 0:
                continue
            token_list = token_idx.tolist()
            topk_list = topk_idx.tolist()
            expert_tokens = x_flat[token_list]
            expert_out = self.mlp(expert_tokens, expert_idx) * top_weights[token_list, topk_list, None]
            out.index_add_(0, token_idx, expert_out)
    else:
        # Device path: batched-MM with static shapes — no .tolist() / device-to-host transfer.
        # Enumerate all T*K (token, expert) slots explicitly.
        token_idx = torch.arange(T, device=x_flat.device).unsqueeze(1).expand(-1, K).reshape(-1)  # [S]
        expert_ids = top_experts.reshape(-1)     # [S]
        routing = top_weights.reshape(-1)        # [S]

        expert_ids_clamped = expert_ids.clamp(0, E - 1)

        # Gather hidden states for each slot
        selected_x = x_flat[token_idx]  # [S, H]

        # Select per-slot expert weights
        w1 = self.mlp.w1.view(E, F_size, hidden_size)  # [E, F, H]
        w2 = self.mlp.w2.view(E, F_size, hidden_size)  # [E, F, H]
        selected_w1 = w1[expert_ids_clamped]  # [S, F, H]
        selected_w2 = w2[expert_ids_clamped]  # [S, F, H]

        # Up projection: [S, 1, H] @ [S, H, F] → [S, F]
        x1 = torch.bmm(selected_x.unsqueeze(1), selected_w1.transpose(-1, -2)).squeeze(1)
        act_out = self.mlp.activation_fn(x1)  # [S, F]

        # Down projection: [S, 1, F] @ [S, F, H] → [S, H]
        expert_out = torch.bmm(act_out.unsqueeze(1), selected_w2).squeeze(1)  # [S, H]

        # Apply routing weights, then sum top-K contributions per token
        out = (expert_out * routing.unsqueeze(-1)).view(T, K, hidden_size).sum(dim=1)  # [T, H]

    return out.reshape(bsz, q_len, hidden_size) + self.bias


def _patch_nomic_experts(model):
    """Monkey-patch all NomicExperts modules with the device-friendly forward."""
    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name == "NomicExperts":
            module.forward = _nomic_experts_forward.__get__(module)


class ModelVariant(StrEnum):
    """Available Nomic Embed Text v2 MoE model variants for embedding generation."""

    NOMIC_EMBED_TEXT_V2_MOE = "nomic-embed-text-v2-moe"


class ModelLoader(ForgeModel):
    """Nomic Embed Text v2 MoE model loader implementation for sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.NOMIC_EMBED_TEXT_V2_MOE: ModelConfig(
            pretrained_model_name="nomic-ai/nomic-embed-text-v2-moe",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NOMIC_EMBED_TEXT_V2_MOE

    sample_sentences = [
        "search_document: TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten"
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Nomic-Embed-Text-v2-MoE",
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
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        _patch_nomic_experts(model)

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
