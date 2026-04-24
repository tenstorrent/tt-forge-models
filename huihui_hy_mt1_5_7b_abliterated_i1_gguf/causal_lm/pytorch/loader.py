# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui HY-MT1.5 7B abliterated i1 GGUF model loader implementation for causal language modeling.
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


def _patch_gguf_hunyuan_dense():
    """Register hunyuan-dense GGUF architecture with transformers.

    transformers 5.2.0 lacks a mapping for the hunyuan-dense GGUF architecture.
    Hunyuan Dense uses the same GGUF config parameter layout as llama, so we
    reuse that mapping. We also remap model_type from hunyuan-dense to
    hunyuan_v1_dense so AutoConfig can resolve the correct config class.
    """
    import transformers.configuration_utils as config_utils_mod
    import transformers.modeling_gguf_pytorch_utils as gguf_utils_mod

    if "hunyuan-dense" in gguf_utils_mod.GGUF_SUPPORTED_ARCHITECTURES:
        return

    llama_map = gguf_utils_mod.GGUF_TO_TRANSFORMERS_MAPPING["config"].get("llama", {})
    gguf_utils_mod.GGUF_TO_TRANSFORMERS_MAPPING["config"][
        "hunyuan-dense"
    ] = llama_map.copy()
    gguf_utils_mod.GGUF_SUPPORTED_ARCHITECTURES.append("hunyuan-dense")

    _original_load = gguf_utils_mod.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(
        gguf_checkpoint_path, return_tensors=False, model_to_load=None, **kwargs
    ):
        result = _original_load(
            gguf_checkpoint_path,
            return_tensors=return_tensors,
            model_to_load=model_to_load,
            **kwargs,
        )
        for section in ("config", "tokenizer_config"):
            if result.get(section, {}).get("model_type") == "hunyuan-dense":
                result[section]["model_type"] = "hunyuan_v1_dense"
        return result

    gguf_utils_mod.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    config_utils_mod.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    import transformers.models.auto.tokenization_auto as tokenization_auto_mod

    tokenization_auto_mod.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "hunyuan-dense" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["hunyuan-dense"] = GGUF_TO_FAST_CONVERTERS["llama"]


class ModelVariant(StrEnum):
    """Available Huihui HY-MT1.5 7B abliterated i1 GGUF model variants for causal language modeling."""

    HUIHUI_HY_MT1_5_7B_ABLITERATED_I1_GGUF = "HUIHUI_HY_MT1_5_7B_ABLITERATED_I1_GGUF"


class ModelLoader(ForgeModel):
    """Huihui HY-MT1.5 7B abliterated i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_HY_MT1_5_7B_ABLITERATED_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-HY-MT1.5-7B-abliterated-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_HY_MT1_5_7B_ABLITERATED_I1_GGUF

    GGUF_FILE = "Huihui-HY-MT1.5-7B-abliterated.i1-Q4_K_M.gguf"

    sample_text = (
        "Translate the following segment into Chinese: The weather is nice today."
    )

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
            model="Huihui HY-MT1.5 7B abliterated i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _patch_gguf_hunyuan_dense()
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        _patch_gguf_hunyuan_dense()
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
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

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        _patch_gguf_hunyuan_dense()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
