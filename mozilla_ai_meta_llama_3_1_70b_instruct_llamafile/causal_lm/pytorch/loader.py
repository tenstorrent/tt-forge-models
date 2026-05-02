# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mozilla-AI Meta-Llama-3.1-70B-Instruct-llamafile model loader implementation for causal language modeling.
"""
import contextlib
import os
import zipfile
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


def _find_gguf_offset_in_llamafile(llamafile_path):
    """Return the byte offset of the uncompressed GGUF entry inside a llamafile ZIP.

    A llamafile is an APE polyglot (ZIP + binary). The GGUF weights are stored as
    an uncompressed ZIP entry. GGUFReader uses np.memmap starting at offset 0, so
    we intercept the memmap call and add the correct byte offset.
    Returns None if the path is not a llamafile or contains no uncompressed GGUF.
    """
    try:
        with zipfile.ZipFile(llamafile_path) as z:
            for info in z.infolist():
                if info.filename.endswith(".gguf") and info.compress_type == 0:
                    # Read fname_len and extra_len from the local file header
                    with open(llamafile_path, "rb") as f:
                        f.seek(info.header_offset + 26)
                        fname_len = int.from_bytes(f.read(2), "little")
                        extra_len = int.from_bytes(f.read(2), "little")
                    return info.header_offset + 30 + fname_len + extra_len
    except (zipfile.BadZipFile, OSError):
        pass
    return None


@contextlib.contextmanager
def _gguf_from_llamafile_ctx(llamafile_path):
    """Temporarily patch gguf.gguf_reader.np.memmap to read GGUF from within a llamafile."""
    data_offset = _find_gguf_offset_in_llamafile(llamafile_path)
    if data_offset is None:
        yield
        return

    import gguf.gguf_reader as _gguf_reader_mod

    _orig_np = _gguf_reader_mod.np
    abs_path = os.path.abspath(str(llamafile_path))

    class _NpWithOffset:
        def memmap(self, path, **kwargs):
            if os.path.abspath(str(path)) == abs_path and "offset" not in kwargs:
                kwargs["offset"] = data_offset
            return _orig_np.memmap(path, **kwargs)

        def __getattr__(self, name):
            return getattr(_orig_np, name)

    _gguf_reader_mod.np = _NpWithOffset()
    try:
        yield
    finally:
        _gguf_reader_mod.np = _orig_np


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

    def _get_llamafile_path(self):
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=self.GGUF_FILE,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        with _gguf_from_llamafile_ctx(self._get_llamafile_path()):
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

        llamafile_path = self._get_llamafile_path()

        if self.num_layers is not None:
            with _gguf_from_llamafile_ctx(llamafile_path):
                config = AutoConfig.from_pretrained(
                    pretrained_model_name, gguf_file=self.GGUF_FILE
                )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        with _gguf_from_llamafile_ctx(llamafile_path):
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
        with _gguf_from_llamafile_ctx(self._get_llamafile_path()):
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
            )
        return self.config
