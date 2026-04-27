# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mozilla-AI Meta-Llama-3.1-70B-Instruct-llamafile model loader implementation for causal language modeling.
"""
import contextlib
import struct
import zipfile
from typing import Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import modeling_gguf_pytorch_utils as _gguf_utils

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


def _get_llamafile_gguf_offset(path):
    """Returns the byte offset of the stored GGUF within a llamafile ZIP, or None."""
    try:
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.filename.endswith(".gguf") and info.compress_type == 0:
                    with open(path, "rb") as f:
                        f.seek(info.header_offset + 26)
                        fname_len, extra_len = struct.unpack("<HH", f.read(4))
                    return info.header_offset + 30 + fname_len + extra_len
    except Exception:
        pass
    return None


@contextlib.contextmanager
def _llamafile_memmap_patch(path):
    """Patches np.memmap so GGUFReader can read the GGUF embedded in a llamafile."""
    data_offset = _get_llamafile_gguf_offset(path)
    if data_offset is None:
        yield
        return

    _orig_memmap = np.memmap
    target = str(path)

    class _OffsetMemmap(np.memmap):
        def __new__(
            cls, filename, dtype="uint8", mode="r+", offset=0, shape=None, order="C"
        ):
            if str(filename) == target:
                return _orig_memmap.__new__(
                    cls,
                    filename,
                    dtype=dtype,
                    mode=mode,
                    offset=data_offset + offset,
                    shape=shape,
                    order=order,
                )
            return _orig_memmap.__new__(
                cls,
                filename,
                dtype=dtype,
                mode=mode,
                offset=offset,
                shape=shape,
                order=order,
            )

    np.memmap = _OffsetMemmap
    try:
        yield
    finally:
        np.memmap = _orig_memmap


_orig_load_gguf_checkpoint = _gguf_utils.load_gguf_checkpoint


def _patched_load_gguf_checkpoint(
    gguf_checkpoint_path, return_tensors=False, model_to_load=None
):
    with _llamafile_memmap_patch(gguf_checkpoint_path):
        return _orig_load_gguf_checkpoint(
            gguf_checkpoint_path, return_tensors=return_tensors, model_to_load=model_to_load
        )


_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available Mozilla-AI Meta-Llama-3.1-70B-Instruct-llamafile model variants for causal language modeling."""

    META_LLAMA_3_1_70B_INSTRUCT_Q4_K_M_LLAMAFILE = (
        "Meta-Llama-3.1-70B-Instruct.Q4_K_M.llamafile"
    )


class ModelLoader(ForgeModel):
    """Mozilla-AI Meta-Llama-3.1-70B-Instruct-llamafile model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.META_LLAMA_3_1_70B_INSTRUCT_Q4_K_M_LLAMAFILE: LLMModelConfig(
            pretrained_model_name="mozilla-ai/Meta-Llama-3.1-70B-Instruct-llamafile",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.META_LLAMA_3_1_70B_INSTRUCT_Q4_K_M_LLAMAFILE

    GGUF_FILE = "Meta-Llama-3.1-70B-Instruct.Q4_K_M.llamafile"

    sample_text = "What is the capital of France?"

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
            model="Meta-Llama-3.1-70B-Instruct-llamafile",
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

        prompts = [self.sample_text]

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
