# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Dolphin 2.5 Mixtral 8x7b GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _patch_mixtral_gguf_support():
    """Patch transformers to support Mixtral GGUF loading.

    Mixtral GGUFs use 'llama' as general.architecture but store MoE weights as
    individual per-expert tensors: blk.N.ffn_gate.K.weight / ffn_up.K / ffn_down.K
    (one tensor per expert K, NOT the stacked _exps format).

    Transformers 5.x MixtralForCausalLM expects stacked expert tensors:
      mlp.experts.gate_up_proj  [n_experts, 2*d_ff, d_model]
      mlp.experts.down_proj     [n_experts, d_model, d_ff]

    Fix:
    1. Register 'mixtral' in GGUF_TO_FAST_CONVERTERS (tokenizer path uses
       the same SentencePiece format as Llama).
    2. Patch load_gguf_checkpoint to detect Mixtral GGUFs and set model_type
       to 'mixtral' so AutoConfig/AutoModel create the right class, and swap
       TENSOR_PROCESSORS["llama"] with MixtralTensorProcessor so the custom
       process() method accumulates per-expert slices correctly.
    3. Patch get_gguf_hf_weights_map to remove wrong blk.N.ffn_down_exps
       entries (from qwen2moe arch map) and add correct per-expert GGUF keys
       blk.N.ffn_{gate,up,down}.K.weight → mlp.experts.{gate_up_proj,down_proj}.
    """
    import re as _re
    import numpy as np
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        TensorProcessor,
        GGUFTensor,
        LlamaTensorProcessor,
        TENSOR_PROCESSORS,
        _gguf_parse_value,
    )

    if getattr(_gguf_utils, "_mixtral_gguf_patched", False):
        return
    _gguf_utils._mixtral_gguf_patched = True

    # Mixtral uses the same SentencePiece tokenizer as Llama; register the
    # converter so convert_gguf_tokenizer("mixtral", ...) doesn't KeyError.
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFLlamaConverter

    GGUF_TO_FAST_CONVERTERS.setdefault("mixtral", GGUFLlamaConverter)

    class MixtralTensorProcessor(LlamaTensorProcessor):
        """Accumulates per-expert GGUF tensors into stacked MixtralForCausalLM expert buffers.

        This GGUF stores one tensor per expert:
          blk.N.ffn_gate.K.weight  →  gate slice of mlp.experts.gate_up_proj[K]
          blk.N.ffn_up.K.weight    →  up slice of mlp.experts.gate_up_proj[K]
          blk.N.ffn_down.K.weight  →  mlp.experts.down_proj[K]

        Inherits Q/K attention weight permutation from LlamaTensorProcessor.
        """
        _GATE_P = _re.compile(r"blk\.(?P<bid>\d+)\.ffn_gate\.(?P<eid>\d+)\.weight$")
        _UP_P = _re.compile(r"blk\.(?P<bid>\d+)\.ffn_up\.(?P<eid>\d+)\.weight$")
        _DOWN_P = _re.compile(r"blk\.(?P<bid>\d+)\.ffn_down\.(?P<eid>\d+)\.weight$")

        def __init__(self, config=None):
            super().__init__(config=config)
            self._n_experts = (config or {}).get("num_local_experts", 8)

        def process(self, weights, name: str, **kwargs):
            tensor_key_mapping = kwargs.get("tensor_key_mapping")
            parsed_parameters = kwargs.get("parsed_parameters")

            if tensor_key_mapping and parsed_parameters:
                if m := _re.fullmatch(self._GATE_P, name):
                    hf_name = tensor_key_mapping.get(name)
                    if hf_name:
                        self._accum_gate_up(weights, parsed_parameters, hf_name, int(m["eid"]), gate=True)
                        return GGUFTensor(weights, None, {})

                if m := _re.fullmatch(self._UP_P, name):
                    hf_name = tensor_key_mapping.get(name)
                    if hf_name:
                        self._accum_gate_up(weights, parsed_parameters, hf_name, int(m["eid"]), gate=False)
                        return GGUFTensor(weights, None, {})

                if m := _re.fullmatch(self._DOWN_P, name):
                    hf_name = tensor_key_mapping.get(name)
                    if hf_name:
                        self._accum_down(weights, parsed_parameters, hf_name, int(m["eid"]))
                        return GGUFTensor(weights, None, {})

            return super().process(weights, name, **kwargs)

        def _accum_gate_up(self, weights, parsed_parameters, hf_name, eid, gate: bool):
            tw = torch.from_numpy(np.copy(weights))  # [d_ff, d_model]
            d_ff = weights.shape[0]
            if hf_name not in parsed_parameters["tensors"]:
                parsed_parameters["tensors"][hf_name] = torch.zeros(
                    [self._n_experts, d_ff * 2, weights.shape[1]], dtype=tw.dtype
                )
            offset = 0 if gate else d_ff
            parsed_parameters["tensors"][hf_name][eid, offset:offset + d_ff, :].copy_(tw)

        def _accum_down(self, weights, parsed_parameters, hf_name, eid):
            tw = torch.from_numpy(np.copy(weights))  # [d_model, d_ff]
            if hf_name not in parsed_parameters["tensors"]:
                parsed_parameters["tensors"][hf_name] = torch.zeros(
                    [self._n_experts, weights.shape[0], weights.shape[1]], dtype=tw.dtype
                )
            parsed_parameters["tensors"][hf_name][eid].copy_(tw)

    # Patch get_gguf_hf_weights_map: when building the tensor key map for a
    # MixtralForCausalLM, use qwen2moe arch for non-expert lookups, then
    # remove the wrong stacked-format _exps entries and add per-expert mappings.
    _orig_get_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_get_map(hf_model, processor, model_type=None, num_layers=None, qual_name=""):
        mt = model_type
        if mt is None:
            mt = getattr(getattr(hf_model, "config", None), "model_type", None)
        if mt != "mixtral":
            return _orig_get_map(hf_model, processor, model_type=model_type, num_layers=num_layers, qual_name=qual_name)

        # Build map using qwen2moe arch (has block_sparse_moe.gate, attn_*, etc.)
        mixtral_proc = MixtralTensorProcessor(config=getattr(processor, "config", {}))
        result_map = _orig_get_map(hf_model, mixtral_proc, model_type="qwen2moe", num_layers=num_layers, qual_name=qual_name)

        # qwen2moe map resolves mlp.experts.down_proj → blk.N.ffn_down_exps (wrong
        # for this GGUF which uses per-expert blk.N.ffn_{gate,up,down}.K.weight).
        for k in [k for k in result_map if "_exps" in k]:
            del result_map[k]

        # Add per-expert mappings.  gate_up_proj / down_proj have no .weight
        # suffix in the state dict, so keys also have no .weight suffix.
        n_layers = getattr(getattr(hf_model, "config", None), "num_hidden_layers", None)
        n_experts = getattr(getattr(hf_model, "config", None), "num_local_experts", mixtral_proc._n_experts)
        if n_layers is not None:
            for layer in range(n_layers):
                pfx = qual_name
                gate_up = f"{pfx}model.layers.{layer}.mlp.experts.gate_up_proj"
                down = f"{pfx}model.layers.{layer}.mlp.experts.down_proj"
                for eid in range(n_experts):
                    result_map[f"blk.{layer}.ffn_gate.{eid}.weight"] = gate_up
                    result_map[f"blk.{layer}.ffn_up.{eid}.weight"] = gate_up
                    result_map[f"blk.{layer}.ffn_down.{eid}.weight"] = down

        return result_map

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_map

    # Patch load_gguf_checkpoint to detect Mixtral GGUFs, swap in the
    # MixtralTensorProcessor (so process() accumulates expert slices), and
    # fix model_type / expert config fields in the returned result dict.
    _orig_load = _gguf_utils.load_gguf_checkpoint

    def _patched_load(gguf_path, return_tensors=False, **kwargs):
        from gguf import GGUFReader

        reader = GGUFReader(gguf_path)
        arch_field = reader.fields.get("general.architecture")
        expert_field = reader.fields.get("llama.expert_count")
        expert_used_field = reader.fields.get("llama.expert_used_count")

        is_mixtral = False
        expert_count = None
        expert_used_count = None
        if arch_field is not None and expert_field is not None and len(expert_field.data) > 0:
            arch_val = _gguf_parse_value(arch_field.parts[arch_field.data[0]], arch_field.types)
            if str(arch_val) == "llama":
                is_mixtral = True
                expert_count = int(
                    _gguf_parse_value(expert_field.parts[expert_field.data[0]], expert_field.types)
                )
                if expert_used_field is not None and len(expert_used_field.data) > 0:
                    expert_used_count = int(
                        _gguf_parse_value(
                            expert_used_field.parts[expert_used_field.data[0]],
                            expert_used_field.types,
                        )
                    )

        # Swap in our processor so process() accumulates per-expert slices.
        orig_llama_proc = TENSOR_PROCESSORS.get("llama")
        if is_mixtral:
            TENSOR_PROCESSORS["llama"] = MixtralTensorProcessor

        try:
            result = _orig_load(gguf_path, return_tensors=return_tensors, **kwargs)
        finally:
            if is_mixtral:
                if orig_llama_proc is not None:
                    TENSOR_PROCESSORS["llama"] = orig_llama_proc
                else:
                    TENSOR_PROCESSORS.pop("llama", None)

        if is_mixtral:
            result["config"]["model_type"] = "mixtral"
            if expert_count is not None:
                result["config"].setdefault("num_local_experts", expert_count)
            if expert_used_count is not None:
                result["config"].setdefault("num_experts_per_tok", expert_used_count)

        return result

    _gguf_utils.load_gguf_checkpoint = _patched_load

    import transformers.modeling_utils as _mu
    import transformers.configuration_utils as _cu

    for mod in (_mu, _cu):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = _patched_load


_patch_mixtral_gguf_support()

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
    """Available Dolphin 2.5 Mixtral 8x7b GGUF model variants for causal language modeling."""

    DOLPHIN_2_5_MIXTRAL_8X7B_GGUF = "DOLPHIN_2_5_MIXTRAL_8X7B_GGUF"


class ModelLoader(ForgeModel):
    """Dolphin 2.5 Mixtral 8x7b GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DOLPHIN_2_5_MIXTRAL_8X7B_GGUF: LLMModelConfig(
            pretrained_model_name="TheBloke/dolphin-2.5-mixtral-8x7b-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DOLPHIN_2_5_MIXTRAL_8X7B_GGUF

    _GGUF_FILES = {
        ModelVariant.DOLPHIN_2_5_MIXTRAL_8X7B_GGUF: "dolphin-2.5-mixtral-8x7b.Q4_K_M.gguf",
    }

    sample_text = "What is your favorite city?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.gguf_file = self._GGUF_FILES[self._variant]
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Dolphin 2.5 Mixtral 8x7b GGUF",
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
        tokenizer_kwargs["gguf_file"] = self.gguf_file

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

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.gguf_file
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts = [text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
