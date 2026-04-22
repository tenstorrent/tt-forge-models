# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mozilla-AI gemma-2-9b-it-llamafile model loader implementation for causal language modeling.
"""
import struct
import zipfile
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional

# Llamafile is a polyglot ZIP+shell-script executable. The actual GGUF data is
# stored uncompressed inside the ZIP. Patch GGUFReader to seek past the shell
# script header so transformers can load weights directly from the llamafile.
try:
    from gguf import GGUFReader as _GGUFReader

    def _get_llamafile_gguf_offset(path):
        """Return byte offset of the uncompressed GGUF entry inside a llamafile ZIP."""
        try:
            with zipfile.ZipFile(path) as zf:
                for name in zf.namelist():
                    if name.endswith(".gguf"):
                        info = zf.getinfo(name)
                        if info.compress_type == 0:
                            with open(path, "rb") as f:
                                f.seek(info.header_offset + 26)
                                fname_len = struct.unpack("<H", f.read(2))[0]
                                extra_len = struct.unpack("<H", f.read(2))[0]
                            return info.header_offset + 30 + fname_len + extra_len
        except Exception:
            pass
        return 0

    _orig_gguf_reader_init = _GGUFReader.__init__

    def _patched_gguf_reader_init(self, path, mode="r"):
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic != b"GGUF":
            offset = _get_llamafile_gguf_offset(path)
            if offset:
                _orig_np_memmap = np.memmap

                def _memmap_with_offset(filename, *args, **kwargs):
                    if str(filename) == str(path):
                        kwargs["offset"] = offset
                    return _orig_np_memmap(filename, *args, **kwargs)

                np.memmap = _memmap_with_offset
                try:
                    _orig_gguf_reader_init(self, path, mode)
                finally:
                    np.memmap = _orig_np_memmap
                return
        _orig_gguf_reader_init(self, path, mode)

    _GGUFReader.__init__ = _patched_gguf_reader_init
except ImportError:
    pass  # gguf not yet installed; RequirementsManager will install it before use

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
    """Available Mozilla-AI gemma-2-9b-it-llamafile model variants for causal language modeling."""

    GEMMA_2_9B_IT_LLAMAFILE = "gemma-2-9b-it-llamafile"


class ModelLoader(ForgeModel):
    """Mozilla-AI gemma-2-9b-it-llamafile model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_2_9B_IT_LLAMAFILE: LLMModelConfig(
            pretrained_model_name="mozilla-ai/gemma-2-9b-it-llamafile",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_2_9B_IT_LLAMAFILE

    GGUF_FILE = "gemma-2-9b-it.Q4_K_M.llamafile"

    sample_text = "What is the capital of France?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="gemma-2-9b-it-llamafile",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self.GGUF_FILE,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the gemma-2-9b-it-llamafile model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            **model_kwargs,
        )

        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        prompts = [text] * batch_size

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        return inputs
