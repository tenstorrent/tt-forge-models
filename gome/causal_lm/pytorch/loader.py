# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gome model loader implementation for causal language modeling.
"""
import sys
import types
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

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
    """Available Gome model variants for causal language modeling."""

    GOME = "gome"


class ModelLoader(ForgeModel):
    """Gome model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.GOME: LLMModelConfig(
            pretrained_model_name="Prositron/gome",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.GOME

    # Shared configuration parameters
    sample_text = "The quick brown fox jumps over the lazy dog."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="Gome",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
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
        """Load and return the Gome model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Gome model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config

        self._patch_model(model)

        return model

    @staticmethod
    def _patch_model(model):
        """Patch forward methods that mix dtypes to keep all computation in the input's dtype.

        The model uses float32 casts internally for quantized computations, which causes
        dtype mismatch errors in the TT backend when the model is loaded in bfloat16.
        """
        for module in model.modules():
            class_name = module.__class__.__name__
            module_ns = sys.modules.get(module.__class__.__module__, None)
            ste_quantize = getattr(module_ns, "ste_quantize", None)

            if class_name == "W4A32Linear" and ste_quantize is not None:

                def make_w4_forward(ste_q):
                    def forward(self, x):
                        w = (self.weight * self.weight_scale).to(x.dtype)
                        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 7.0
                        w_q = ste_q(w, -8, 7, scale)
                        bias = self.bias.to(x.dtype) if self.bias is not None else None
                        return F.linear(x, w_q, bias)

                    return forward

                module.forward = types.MethodType(make_w4_forward(ste_quantize), module)

            elif class_name == "W8A32Linear" and ste_quantize is not None:

                def make_w8_forward(ste_q):
                    def forward(self, x):
                        w = (self.weight * self.weight_scale).to(x.dtype)
                        scale = (
                            w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
                        )
                        w_q = ste_q(w, -128, 127, scale)
                        bias = self.bias.to(x.dtype) if self.bias is not None else None
                        return F.linear(x, w_q, bias)

                    return forward

                module.forward = types.MethodType(make_w8_forward(ste_quantize), module)

            elif class_name == "Int8Embedding" and ste_quantize is not None:

                def make_int8_emb_forward(ste_q):
                    def forward(self, idx):
                        w = self.weight * self.weight_scale
                        scale = (
                            w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
                        )
                        w_q = ste_q(w, -128, 127, scale)
                        flat = idx.view(-1)
                        out = w_q[flat]
                        return out.view(*idx.shape, self.d)

                    return forward

                module.forward = types.MethodType(
                    make_int8_emb_forward(ste_quantize), module
                )

            elif class_name == "RMSNorm":

                def rmsnorm_forward(self, x):
                    n = x.float() * torch.rsqrt(
                        x.float().pow(2).mean(-1, keepdim=True) + self.eps
                    )
                    return n.to(x.dtype) * self.weight

                module.forward = types.MethodType(rmsnorm_forward, module)

            elif class_name == "ResonantCoordinateEmbeddings":

                def rce_forward(self, x):
                    coords = self.token_coord(x).sigmoid()
                    gx = coords[..., 0] * (self.grid_size - 1)
                    gy = coords[..., 1] * (self.grid_size - 1)
                    x0 = gx.floor().long().clamp(0, self.grid_size - 1)
                    x1 = (x0 + 1).clamp(0, self.grid_size - 1)
                    y0 = gy.floor().long().clamp(0, self.grid_size - 1)
                    y1 = (y0 + 1).clamp(0, self.grid_size - 1)
                    wx = (gx - x0.to(gx.dtype)).unsqueeze(-1)
                    wy = (gy - y0.to(gy.dtype)).unsqueeze(-1)
                    row = self.row_embed(x0) * (1 - wx) + self.row_embed(x1) * wx
                    col = self.col_embed(y0) * (1 - wy) + self.col_embed(y1) * wy
                    emb = torch.cat([row, col], dim=-1)
                    h = self.fc1(emb)
                    h = F.silu(h)
                    h = self.norm(h)
                    h = self.fc2(h)
                    return self.coord_gate * h

                module.forward = types.MethodType(rce_forward, module)

            elif class_name == "DeepMicroExpert":

                def dme_forward(self, x):
                    B, T, D = x.shape
                    x_flat = x.view(-1, D)
                    N = x_flat.shape[0]

                    logits = self.router(x_flat)
                    probs = torch.softmax(logits, -1)
                    w, idx = torch.topk(probs, self.top_k, dim=-1)
                    w = w / (w.sum(-1, keepdim=True) + 1e-8)

                    token_ids = (
                        torch.arange(N, device=x.device)
                        .unsqueeze(1)
                        .expand(-1, self.top_k)
                        .reshape(-1)
                    )
                    expert_ids = idx.reshape(-1)
                    weights = w.reshape(-1)

                    flat_input = x_flat[token_ids]
                    out = torch.zeros(N, D, device=x.device, dtype=x.dtype)

                    for e in range(self.num_experts):
                        mask = expert_ids == e
                        if mask.sum() == 0:
                            continue

                        inp = flat_input[mask]
                        wt = weights[mask].unsqueeze(-1)

                        h = F.silu(self.gate[e](inp)) * self.up[e](inp)

                        delta = torch.matmul(inp, self.adapter_B[e].to(inp.dtype))
                        delta = torch.matmul(delta, self.adapter_A[e].to(inp.dtype))
                        h = h + delta

                        h = self.down[e](h)
                        h = self.dropout(h)

                        out.index_add_(0, token_ids[mask], h * wt.to(h.dtype))

                    return out.view(B, T, D)

                module.forward = types.MethodType(dme_forward, module)

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Gome model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
