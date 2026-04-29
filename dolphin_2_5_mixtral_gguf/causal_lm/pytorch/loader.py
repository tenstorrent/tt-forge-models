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

    Mixtral GGUFs use 'llama' as general.architecture but contain MoE expert
    tensors (ffn_gate_exps, ffn_up_exps, ffn_down_exps).  Transformers' GGUF
    loader uses the Llama tensor processor which doesn't know how to map those
    tensors, leaving all MLP weights randomly initialised (PCC ≈ 0.13).

    Fix: detect llama GGUF files that have expert_count > 0 and transparently
    substitute Qwen2MoeTensorProcessor (which handles the same expert tensor
    format) and remap model_type to 'mixtral' so the right config class is
    instantiated.
    """
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        Qwen2MoeTensorProcessor,
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

    # Patch get_gguf_hf_weights_map so that model_type='mixtral' resolves to
    # the qwen2moe tensor-name map (same GGUF tensor naming convention).
    _orig_get_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_get_map(hf_model, processor, model_type=None, num_layers=None, qual_name=""):
        mt = model_type
        if mt is None:
            mt = getattr(getattr(hf_model, "config", None), "model_type", None)
        if mt == "mixtral":
            processor = Qwen2MoeTensorProcessor(config=getattr(processor, "config", {}))
            return _orig_get_map(
                hf_model, processor, model_type="qwen2moe",
                num_layers=num_layers, qual_name=qual_name,
            )
        return _orig_get_map(hf_model, processor, model_type=model_type,
                             num_layers=num_layers, qual_name=qual_name)

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_map

    # Patch load_gguf_checkpoint to detect Mixtral GGUFs and use the MoE
    # tensor processor, then fix model_type / expert config fields in result.
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

        if is_mixtral:
            orig_llama_proc = TENSOR_PROCESSORS.get("llama")
            TENSOR_PROCESSORS["llama"] = Qwen2MoeTensorProcessor

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
