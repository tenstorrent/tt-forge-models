# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PyTorch loader for HunyuanImage-3.0 (tencent/HunyuanImage-3.0).

An 80B-total / 13B-active autoregressive multimodal MoE (64 experts, top-8).
"""

import contextlib
import sys
import types
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from torch.nn.utils import parametrize
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available HunyuanImage-3.0 variants."""

    HunyuanImage3MoE = "HunyuanImage-3.0"


def patch_moe_for_xla(model):
    """Neutralize CUDA calls that run before the eager MoE branch.

    ``moe_impl='eager'`` does not skip them. ``set_device`` raises because a
    CPU tensor's ``device.index`` is ``None``, and on XLA it reaches
    ``_cuda_setDevice``. ``nvtx.range`` raises on a non-CUDA build. The module
    imports ``nvtx`` once and only calls ``range``, so one null context covers
    MoE and attention.
    """
    moe_cls = next(
        (type(m) for m in model.modules() if type(m).__name__ == "HunyuanMoE"),
        None,
    )
    if moe_cls is None:
        raise RuntimeError(
            "No HunyuanMoE module found in the loaded model. The remote code "
            "layout changed; re-check the CUDA-only call workarounds."
        )
    if getattr(moe_cls.forward, "_tt_xla_patched", False):
        return model  # A second loader in the same process must not re-wrap.

    sys.modules[moe_cls.__module__].nvtx = SimpleNamespace(
        range=lambda *args, **kwargs: contextlib.nullcontext()
    )
    original_forward = moe_cls.forward

    def forward(self, hidden_states):
        saved = torch.cuda.set_device
        torch.cuda.set_device = lambda *args, **kwargs: None
        try:
            return original_forward(self, hidden_states)
        finally:
            torch.cuda.set_device = saved

    forward._tt_xla_patched = True
    moe_cls.forward = forward
    return model


def _split_mlp_forward(self, x):
    """Silu MLP on the unfused projections.

    The checkpoint applies the activation to the second half (``x1 * act(x2)``),
    so it lands on ``up_proj``. Swapping the two would not match training.
    """
    return self.down_proj(self.gate_proj(x) * self.act_fn(self.up_proj(x)))


def split_fused_gate_and_up(model):
    """Unfuse ``gate_and_up_proj`` so each half can column-shard.

    The weight is ``[gate | up]`` on dim 0. A column shard gives the first
    devices gate-only rows and the rest up-only rows, and ``chunk(2)`` cannot
    put them back, so every expert would reshard. Separate Linears drop the
    chunk and let ``down_proj`` row-shard against each half.

    Both halves stay views of the mmap'd checkpoint; copying them would
    materialise the expert stack in RAM. Every MLP in this checkpoint is silu
    with ``mlp_bias=False``. The Linears are built on meta so they allocate
    nothing of their own.
    """
    split = 0
    for module in model.modules():
        fused = getattr(module, "gate_and_up_proj", None)
        if fused is None:
            continue

        weight = fused.weight.detach()
        half = weight.shape[0] // 2
        for name, rows in (("gate_proj", weight[:half]), ("up_proj", weight[half:])):
            linear = nn.Linear(
                weight.shape[1], half, bias=False, device="meta", dtype=weight.dtype
            )
            linear.weight = nn.Parameter(rows, requires_grad=False)
            setattr(module, name, linear)

        del module.gate_and_up_proj
        module.forward = types.MethodType(_split_mlp_forward, module)
        split += 1

    if not split:
        raise RuntimeError(
            "No fused gate_and_up_proj found. The remote code layout changed; "
            "re-check the tensor-parallel shard spec."
        )
    return model


class ModelLoader(ForgeModel):
    """Load ``HunyuanImage3ForCausalMM`` from the Hub into ``HF_HOME``."""

    _VARIANTS = {
        ModelVariant.HunyuanImage3MoE: ModelConfig(
            pretrained_model_name="tencent/HunyuanImage-3.0"
        ),
    }
    DEFAULT_VARIANT = ModelVariant.HunyuanImage3MoE

    sample_text = "A brown and white dog is running on the grass"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="HunyuanImage3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_config(self):
        if self.model is not None:
            self.config = self.model.config
            return self.config

        checkpoint = self._variant_config.pretrained_model_name
        self.config = AutoConfig.from_pretrained(
            checkpoint,
            trust_remote_code=True,
        )
        return self.config

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the remote model on the host.

        ``moe_impl`` is a constructor arg: the MoE layer reads it in
        ``__init__``. The remote model has no tokenizer until one is attached,
        and ``prepare_model_inputs`` needs it.
        """
        checkpoint = self._variant_config.pretrained_model_name
        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "sdpa",
            "dtype": dtype_override or torch.bfloat16,
            "moe_impl": "eager",
        }
        model_kwargs.update(kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            **model_kwargs,
        )
        self.model.load_tokenizer(
            AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        )
        patch_moe_for_xla(self.model)
        split_fused_gate_and_up(self.model)
        self.model.eval()  # The split Linears are new and start in train mode.
        self.config = self.model.config
        return self.model

    def get_mesh_config(self, num_devices: int):
        """``(1, N)`` mesh, or ``(4, 8)`` at 32 devices. Axis 1 is the tensor split.

        The model axis has to divide the 8 KV heads — fused qkv splits on
        those boundaries — and ``moe_intermediate_size``.
        """
        mesh_shape = (4, 8) if num_devices == 32 else (1, num_devices)
        model_axis = mesh_shape[1]

        config = self.config if self.config is not None else self.load_config()

        assert (
            config.num_key_value_heads % model_axis == 0
        ), f"num_key_value_heads must be divisible by the model axis ({model_axis})"

        moe_intermediate = config.moe_intermediate_size
        if isinstance(moe_intermediate, (list, tuple)):
            moe_intermediate = moe_intermediate[0]
        assert (
            moe_intermediate % model_axis == 0
        ), f"moe_intermediate_size must be divisible by the model axis ({model_axis})"

        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Column-parallel on dim 0, row-parallel on dim 1.

        Fused qkv splits on KV-head boundaries, so each shard keeps its Q
        group plus K and V and ``o_proj``'s row shard lines up. Experts and
        the shared MLP are the unfused pair, closed by ``down_proj``. The
        router stays replicated so every device picks the same top-8. ``wte``
        stays replicated; ``lm_head`` column-shards the vocab.
        """

        if parametrize.is_parametrized(
            model.model.layers[0].self_attn.qkv_proj, "weight"
        ):
            raise RuntimeError(
                "Weights are parametrized (weight dtype overrides are active); "
                "the shard spec would not apply. Remove the "
                "mixed_precision_configs JSON for this model."
            )

        shard_specs = {}
        for layer in model.model.layers:
            attn = layer.self_attn
            shard_specs[attn.qkv_proj.weight] = ("model", "batch")
            shard_specs[attn.o_proj.weight] = ("batch", "model")

            for block in (*layer.mlp.experts, layer.mlp.shared_mlp):
                shard_specs[block.gate_proj.weight] = ("model", "batch")
                shard_specs[block.up_proj.weight] = ("model", "batch")
                shard_specs[block.down_proj.weight] = ("batch", "model")

        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
        prompt: Optional[str] = None,
        **kwargs,
    ):
        if self.model is None:
            self.load_model()

        prepared = self.model.prepare_model_inputs(
            prompt=[prompt or self.sample_text] * batch_size,
            mode="gen_text",
            max_new_tokens=1,
            **kwargs,
        )
        attention_mask = self.model._prepare_attention_mask_for_generation(
            prepared["input_ids"],
            self.model.generation_config,
            prepared,
        )
        return {
            "input_ids": prepared["input_ids"],
            "attention_mask": attention_mask,
            "position_ids": prepared["position_ids"],
            "custom_pos_emb": prepared["custom_pos_emb"],
            "use_cache": False,
            "mode": "gen_text",
            "return_dict": False,
        }
