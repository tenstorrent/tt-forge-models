# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mira-7B i1 GGUF model loader implementation for causal language modeling.

Based on Qwen2 architecture, quantized from emrahemrez/mira-7b.
"""
import numpy as np
import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
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


def _load_gguf_weights(model, gguf_path, dtype=None):
    """Load GGUF weights into model using gguf-py directly.

    Bypasses transformers' load_gguf_checkpoint to avoid compatibility issues
    with other loaders that monkey-patch that function without forwarding the
    model_to_load argument required by transformers >= 5.x.
    """
    from gguf import GGUFReader, dequantize
    from transformers.modeling_gguf_pytorch_utils import TensorProcessor

    reader = GGUFReader(gguf_path)
    processor = TensorProcessor()
    tensor_key_mapping = _gguf_utils.get_gguf_hf_weights_map(model, processor)

    target_dtype = dtype if dtype is not None else torch.bfloat16
    tensors_to_load = {}

    for tensor in reader.tensors:
        if tensor.name not in tensor_key_mapping:
            continue
        weights = dequantize(tensor.data, tensor.tensor_type)
        hf_name = tensor_key_mapping[tensor.name]
        tensors_to_load[hf_name] = torch.from_numpy(np.copy(weights)).to(target_dtype)

    model.load_state_dict(tensors_to_load, strict=False)


class ModelVariant(StrEnum):
    """Available Mira-7B i1 GGUF model variants for causal language modeling."""

    MIRA_7B_I1_GGUF = "mira_7b_i1_GGUF"


class ModelLoader(ForgeModel):
    """Mira-7B i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MIRA_7B_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/mira-7b-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MIRA_7B_I1_GGUF

    GGUF_FILE = "mira-7b.i1-Q4_K_M.gguf"

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
            model="Mira-7B i1 GGUF",
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
        from huggingface_hub import hf_hub_download

        pretrained_model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        gguf_path = hf_hub_download(
            repo_id=pretrained_model_name, filename=self.GGUF_FILE
        )

        config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=self.GGUF_FILE
        )

        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        model = AutoModelForCausalLM.from_config(config).to(dtype)

        _load_gguf_weights(model, gguf_path, dtype=dtype)

        model.eval()
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
            shard_specs[layer.self_attn.q_proj.bias] = ("model",)
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.bias] = ("model",)
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.bias] = ("model",)
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
