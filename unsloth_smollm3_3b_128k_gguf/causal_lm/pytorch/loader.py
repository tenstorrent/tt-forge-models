# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unsloth SmolLM3 3B 128K GGUF model loader implementation for causal language modeling.
"""
import torch
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


def _patch_transformers_smollm3_gguf():
    """Monkey-patch transformers to add smollm3 GGUF architecture support.

    SmolLM3 uses the 'smollm3' architecture identifier in GGUF metadata but is
    Llama-based (SmolLM3ForCausalLM inherits LlamaForCausalLM). We bridge the gap
    by registering smollm3 with llama's config mapping and tokenizer converter.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )

    if "smollm3" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("smollm3")
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["smollm3"] = GGUF_TO_TRANSFORMERS_MAPPING[
        "config"
    ]["llama"]

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFLlamaConverter,
    )

    class GGUFSmolLM3Converter(GGUFLlamaConverter):
        def tokenizer(self, proto):
            # GGUFLlamaConverter.tokenizer has a bug where it accesses proto.bos_token_id
            # when checking eos_token_id (line 413 of ggml.py). SmolLM3 GGUF omits
            # bos_token_id; fall back to eos_token_id so the index lookup succeeds.
            if not hasattr(proto, "bos_token_id"):
                proto.bos_token_id = getattr(proto, "eos_token_id", None)
            return super().tokenizer(proto)

    if "smollm3" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["smollm3"] = GGUFSmolLM3Converter


_patch_transformers_smollm3_gguf()


class ModelVariant(StrEnum):
    """Available Unsloth SmolLM3 3B 128K GGUF model variants for causal language modeling."""

    UNSLOTH_SMOLLM3_3B_128K_GGUF = "3B_128K_GGUF"


class ModelLoader(ForgeModel):
    """Unsloth SmolLM3 3B 128K GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.UNSLOTH_SMOLLM3_3B_128K_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/SmolLM3-3B-128K-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.UNSLOTH_SMOLLM3_3B_128K_GGUF

    GGUF_FILE = "SmolLM3-3B-128K-Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language models."

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
            model="Unsloth SmolLM3 3B 128K GGUF",
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
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
