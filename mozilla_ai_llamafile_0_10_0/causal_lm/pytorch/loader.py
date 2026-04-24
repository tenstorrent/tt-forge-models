# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mozilla-ai/llamafile_0.10.0 model loader implementation for causal language modeling.
"""
import os
import zipfile
import torch
from huggingface_hub import hf_hub_download
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
    """Available mozilla-ai/llamafile_0.10.0 model variants for causal language modeling."""

    QWEN3_5_0_8B_Q8_0_LLAMAFILE = "Qwen3.5-0.8B-Q8_0-llamafile"


class ModelLoader(ForgeModel):
    """mozilla-ai/llamafile_0.10.0 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_5_0_8B_Q8_0_LLAMAFILE: LLMModelConfig(
            pretrained_model_name="mozilla-ai/llamafile_0.10.0",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_5_0_8B_Q8_0_LLAMAFILE

    _GGUF_FILES = {
        ModelVariant.QWEN3_5_0_8B_Q8_0_LLAMAFILE: "Qwen3.5-0.8B-Q8_0.llamafile",
    }

    sample_text = "Give me a short introduction to large language models."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers
        self.gguf_file = self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="mozilla-ai llamafile_0.10.0",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_gguf_path(self):
        """Extract GGUF from llamafile zip and return (local_dir, gguf_filename)."""
        llamafile_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=self.gguf_file,
        )
        gguf_name = os.path.splitext(os.path.basename(self.gguf_file))[0] + ".gguf"
        extract_dir = os.path.dirname(llamafile_path)
        gguf_path = os.path.join(extract_dir, gguf_name)
        if not os.path.exists(gguf_path):
            with zipfile.ZipFile(llamafile_path) as z:
                z.extract(gguf_name, path=extract_dir)
        return extract_dir, gguf_name

    def _load_tokenizer(self, dtype_override=None):
        local_dir, gguf_name = self._get_gguf_path()
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = gguf_name

        self.tokenizer = AutoTokenizer.from_pretrained(local_dir, **tokenizer_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        local_dir, gguf_name = self._get_gguf_path()

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = gguf_name

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(local_dir, gguf_file=gguf_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(local_dir, **model_kwargs).eval()

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
        local_dir, gguf_name = self._get_gguf_path()
        self.config = AutoConfig.from_pretrained(local_dir, gguf_file=gguf_name)
        return self.config
