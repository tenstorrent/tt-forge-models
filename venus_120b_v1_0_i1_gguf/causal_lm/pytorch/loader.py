# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Venus 120B v1.0 i1 GGUF model loader implementation for causal language modeling.
"""
import os
import shutil

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


def _concat_gguf_parts(repo_id, base_name, num_parts, cache_dir=None):
    """Download split GGUF parts and concatenate into a single file.

    Returns the directory and filename of the concatenated GGUF.
    """
    if cache_dir is None:
        cache_dir = os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "gguf_merged",
            repo_id.replace("/", "__"),
        )
    os.makedirs(cache_dir, exist_ok=True)
    merged_path = os.path.join(cache_dir, base_name)

    if os.path.isfile(merged_path):
        return cache_dir, base_name

    part_paths = []
    for i in range(1, num_parts + 1):
        part_name = f"{base_name}.part{i}of{num_parts}"
        local = hf_hub_download(repo_id=repo_id, filename=part_name)
        part_paths.append(local)

    tmp_path = merged_path + ".tmp"
    with open(tmp_path, "wb") as out:
        for p in part_paths:
            with open(p, "rb") as inp:
                shutil.copyfileobj(inp, out)
    os.replace(tmp_path, merged_path)
    return cache_dir, base_name


class ModelVariant(StrEnum):
    """Available Venus 120B v1.0 i1 GGUF model variants for causal language modeling."""

    VENUS_120B_V1_0_I1_Q4_K_M_GGUF = "VENUS_120B_V1_0_I1_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Venus 120B v1.0 i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.VENUS_120B_V1_0_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Venus-120b-v1.0-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VENUS_120B_V1_0_I1_Q4_K_M_GGUF

    GGUF_BASE_NAME = "Venus-120b-v1.0.i1-Q4_K_M.gguf"
    GGUF_NUM_PARTS = 2

    sample_text = "What is your favorite city?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers
        self._gguf_dir = None
        self._gguf_file = None

    def _ensure_gguf(self):
        if self._gguf_dir is not None:
            return
        self._gguf_dir, self._gguf_file = _concat_gguf_parts(
            self._variant_config.pretrained_model_name,
            self.GGUF_BASE_NAME,
            self.GGUF_NUM_PARTS,
        )

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Venus 120B v1.0 i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self._ensure_gguf()
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self._gguf_file

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._gguf_dir, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        self._ensure_gguf()

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self._gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                self._gguf_dir, gguf_file=self._gguf_file
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            self._gguf_dir, **model_kwargs
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
        return shard_specs

    def load_config(self):
        self._ensure_gguf()
        self.config = AutoConfig.from_pretrained(
            self._gguf_dir, gguf_file=self._gguf_file
        )
        return self.config
