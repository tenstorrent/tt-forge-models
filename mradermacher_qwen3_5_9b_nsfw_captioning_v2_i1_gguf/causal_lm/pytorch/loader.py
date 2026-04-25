# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.5 9B NSFW Captioning v2 i1 GGUF model loader implementation for causal language modeling.

Qwen 3.5 9B is a Mamba-attention hybrid: every 4th layer (3, 7, 11, ...) is a
full-attention layer; the remaining layers use linear (SSM-style) attention.
Transformers does not yet support loading this architecture from GGUF, so we
instantiate Qwen3_5ForCausalLM directly with the config derived from GGUF
metadata and use random weights (appropriate for compile-only environments).
"""
import os
from typing import Optional

import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import AutoTokenizer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

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


def _patch_gguf_is_available():
    """Fix is_gguf_available() for gguf installed after transformers was imported.

    PACKAGE_DISTRIBUTION_MAPPING is frozen at transformers import time, so gguf
    installed via requirements.txt during the test run is absent from the map.
    This causes _is_package_available to fall back to getattr(gguf, '__version__',
    'N/A'), and since gguf has no __version__, version.parse('N/A') raises
    InvalidVersion. We bypass the cached mapping by calling
    importlib.metadata.version() directly.
    """
    import importlib.metadata
    import importlib.util

    from packaging import version as pkg_version

    def _fixed_is_gguf_available(min_version=None):
        if importlib.util.find_spec("gguf") is None:
            return False
        try:
            gguf_ver = importlib.metadata.version("gguf")
            if min_version is None:
                return True
            try:
                return pkg_version.parse(gguf_ver) >= pkg_version.parse(min_version)
            except pkg_version.InvalidVersion:
                return True
        except Exception:
            return True

    _gguf_utils.is_gguf_available = _fixed_is_gguf_available


_patch_gguf_is_available()


class ModelVariant(StrEnum):
    """Available Qwen 3.5 9B NSFW Captioning v2 i1 GGUF model variants for causal language modeling."""

    QWEN_3_5_9B_NSFW_CAPTIONING_V2_I1_GGUF = "9B_i1_GGUF"


class ModelLoader(ForgeModel):
    """Qwen 3.5 9B NSFW Captioning v2 i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_9B_NSFW_CAPTIONING_V2_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/qwen3.5-9b-nsfw-captioning-v2-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_9B_NSFW_CAPTIONING_V2_I1_GGUF

    GGUF_FILE = "qwen3.5-9b-nsfw-captioning-v2.i1-Q4_K_M.gguf"

    # Qwen 3.5 9B architecture constants derived from GGUF metadata.
    VOCAB_SIZE = 248320

    sample_text = "Give me a short introduction to large language model."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @staticmethod
    def _use_random_weights():
        return os.environ.get("TT_RANDOM_WEIGHTS") or os.environ.get(
            "TT_COMPILE_ONLY_SYSTEM_DESC"
        )

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen 3.5 9B NSFW Captioning v2 i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _qwen35_config(self):
        """Build Qwen3_5TextConfig matching the GGUF metadata for this model.

        Qwen 3.5 9B uses 32 layers with full_attention every 4th layer
        (indices 3, 7, 11, ..., 31); remaining layers use linear_attention.
        All other values match Qwen3_5TextConfig defaults which were designed
        for this model family.
        """
        num_layers = self.num_layers if self.num_layers is not None else 32
        layer_types = [
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(num_layers)
        ]
        return Qwen3_5TextConfig(
            vocab_size=self.VOCAB_SIZE,
            num_hidden_layers=num_layers,
            layer_types=layer_types,
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
        config = self._qwen35_config()
        self.config = config

        target_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(target_dtype)
        try:
            model = Qwen3_5ForCausalLM(config)
        finally:
            torch.set_default_dtype(orig_dtype)

        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        max_length = self._variant_config.max_length

        if self._use_random_weights():
            input_ids = torch.randint(0, self.VOCAB_SIZE, (batch_size, max_length))
            attention_mask = torch.ones(batch_size, max_length, dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

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
            if hasattr(layer, "self_attn"):
                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            if hasattr(layer, "mlp"):
                shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = self._qwen35_config()
        return self.config
