# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mozilla-AI Meta-Llama-3.1-70B-Instruct-llamafile model loader implementation for causal language modeling.
"""
import struct
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


class ModelVariant(StrEnum):
    """Available Mozilla-AI Meta-Llama-3.1-70B-Instruct-llamafile model variants for causal language modeling."""

    META_LLAMA_3_1_70B_INSTRUCT_Q4_K_M_LLAMAFILE = (
        "Meta-Llama-3.1-70B-Instruct.Q4_K_M.llamafile"
    )


def _get_gguf_offset_in_llamafile(path: str) -> int:
    """Return the byte offset of the embedded GGUF data inside a llamafile.

    A .llamafile is a cosmopolitan APE (Actually Portable Executable) that
    also serves as a valid ZIP archive.  The GGUF model weights are stored
    as a stored (uncompressed) ZIP entry inside that archive.

    Returns 0 if the file is a plain GGUF (not a llamafile).
    """
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for info in zf.infolist():
                if not info.filename.endswith(".gguf"):
                    continue
                if info.compress_type != zipfile.ZIP_STORED:
                    # Compressed GGUF inside a ZIP is not supported by GGUFReader.
                    raise ValueError(
                        f"llamafile entry {info.filename!r} is compressed "
                        f"(compress_type={info.compress_type}); cannot mmap directly."
                    )
                # The actual GGUF bytes start after the local file header.
                # Local file header layout (APPNOTE 4.3.7):
                #   signature         4 bytes  (PK\x03\x04)
                #   version needed    2 bytes
                #   general purpose   2 bytes
                #   compression       2 bytes
                #   last mod time     2 bytes
                #   last mod date     2 bytes
                #   crc-32            4 bytes
                #   compressed size   4 bytes
                #   uncompressed size 4 bytes
                #   file name length  2 bytes  (offset 26)
                #   extra field len   2 bytes  (offset 28)
                #   --- total: 30 bytes ---
                with open(path, "rb") as fh:
                    fh.seek(info.header_offset + 26)
                    fname_len, extra_len = struct.unpack("<HH", fh.read(4))
                return info.header_offset + 30 + fname_len + extra_len
    except (zipfile.BadZipFile, KeyError):
        pass
    return 0


def _patch_gguf_reader_for_llamafile():
    """Patch GGUFReader to handle the llamafile format.

    A .llamafile is a cosmopolitan APE that embeds the GGUF as a stored ZIP
    entry at a large byte offset.  The upstream GGUFReader calls
    ``np.memmap(path)`` starting at offset 0, which reads the MZ/APE
    executable header and raises ``ValueError: GGUF magic invalid``.

    This patch detects the llamafile format, finds the embedded GGUF entry
    via Python's zipfile module, and temporarily replaces the np reference
    inside gguf_reader so that GGUFReader's memmap starts at the GGUF data.
    """
    try:
        import gguf.gguf_reader as _gguf_reader
        import numpy as _np
    except ImportError:
        return

    _original_init = _gguf_reader.GGUFReader.__init__

    def _llamafile_aware_init(self, path, mode="r"):
        gguf_offset = _get_gguf_offset_in_llamafile(str(path))
        if gguf_offset == 0:
            _original_init(self, path, mode)
            return

        # Temporarily swap gguf_reader's numpy reference to inject the
        # byte offset into every memmap call made by the original __init__.
        _orig_np = _gguf_reader.np
        _frozen_offset = gguf_offset

        class _NpWithOffsetMemmap:
            """Shim that forwards all numpy attribute access but intercepts memmap."""

            def __getattr__(self, name):
                return getattr(_orig_np, name)

            @staticmethod
            def memmap(filename, dtype=_np.uint8, mode="r", **kwargs):
                kwargs.setdefault("offset", _frozen_offset)
                return _orig_np.memmap(filename, dtype=dtype, mode=mode, **kwargs)

        _gguf_reader.np = _NpWithOffsetMemmap()
        try:
            _original_init(self, path, mode)
        finally:
            _gguf_reader.np = _orig_np

    _gguf_reader.GGUFReader.__init__ = _llamafile_aware_init


_patch_gguf_reader_for_llamafile()


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
