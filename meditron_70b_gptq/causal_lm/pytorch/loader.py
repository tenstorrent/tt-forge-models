# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TheBloke meditron-70B-GPTQ model loader implementation for causal language modeling.
"""

from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available meditron-70B-GPTQ model variants."""

    MEDITRON_70B_GPTQ = "70B_GPTQ"


class ModelLoader(ForgeModel):
    """TheBloke meditron-70B-GPTQ model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MEDITRON_70B_GPTQ: LLMModelConfig(
            pretrained_model_name="TheBloke/meditron-70B-GPTQ",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MEDITRON_70B_GPTQ

    sample_text = "What are the common symptoms of type 2 diabetes?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="meditron-70B-GPTQ",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"device_map": "cpu"}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        # gptqmodel 6.x removed BACKEND.EXLLAMA_V1; add compat shim so that
        # optimum 2.1.0's post_init_model comparison always evaluates False.
        try:
            from gptqmodel import BACKEND

            if not hasattr(BACKEND, "EXLLAMA_V1"):
                BACKEND.EXLLAMA_V1 = "exllama_v1_compat"
        except ImportError:
            pass

        # TorchFusedQuantLinear's _weight_int4pack_mm_for_cpu only supports
        # group_size in {32,64,128,256}, but this model uses group_size=-1
        # (no grouping, becomes in_features=8192 at runtime).  Temporarily
        # remove -1 from SUPPORTS_GROUP_SIZE so auto-selection skips it and
        # falls back to TorchQuantLinear which supports all group sizes.
        _fused_orig_gs = None
        try:
            from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedQuantLinear

            _fused_orig_gs = TorchFusedQuantLinear.SUPPORTS_GROUP_SIZE
            TorchFusedQuantLinear.SUPPORTS_GROUP_SIZE = [
                g for g in _fused_orig_gs if g != -1
            ]
        except ImportError:
            pass

        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()
        finally:
            if _fused_orig_gs is not None:
                TorchFusedQuantLinear.SUPPORTS_GROUP_SIZE = _fused_orig_gs

        self.config = model.config
        self.model = model
        return model

    def get_mesh_config(self, num_devices: int):
        return (1, num_devices), ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            attn = layer.self_attn
            mlp = layer.mlp
            for proj, col_parallel in [
                (attn.q_proj, True),
                (attn.k_proj, True),
                (attn.v_proj, True),
                (attn.o_proj, False),
                (mlp.gate_proj, True),
                (mlp.up_proj, True),
                (mlp.down_proj, False),
            ]:
                if hasattr(proj, "qweight"):
                    # GPTQ: qweight is [in//pack, out], scales/qzeros are [groups, out]
                    # column-parallel shards along out (dim 1); row-parallel shards along in (dim 0)
                    qw_spec = ("batch", "model") if col_parallel else ("model", "batch")
                    scale_spec = ("batch", "model") if col_parallel else (None, None)
                    shard_specs[proj.qweight] = qw_spec
                    if hasattr(proj, "scales") and proj.scales is not None:
                        shard_specs[proj.scales] = scale_spec
                    if hasattr(proj, "qzeros") and proj.qzeros is not None:
                        shard_specs[proj.qzeros] = scale_spec
                elif hasattr(proj, "weight") and isinstance(proj.weight, torch.Tensor):
                    col_spec = ("model", "batch")
                    row_spec = ("batch", "model")
                    shard_specs[proj.weight] = col_spec if col_parallel else row_spec
        if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            if isinstance(model.lm_head.weight, torch.Tensor):
                shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
