# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EAGLE3 Speculator model loader implementation for speculative decoding.
"""

import json
import os
import shutil
import tempfile

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Optional

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel

VERIFIER_TOKENIZER_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"


def _clear_speculators_eagle3_registry():
    try:
        from speculators import SpeculatorModelConfig, SpeculatorModel

        for cls in (SpeculatorModelConfig, SpeculatorModel):
            for attr in dir(cls):
                if "registry" in attr.lower():
                    reg = getattr(cls, attr)
                    if isinstance(reg, dict) and "eagle3" in reg:
                        del reg["eagle3"]
    except ImportError:
        pass


def _prepare_local_model_dir(pretrained_model_name):
    config_path = hf_hub_download(pretrained_model_name, "config.json")
    eagle3_path = hf_hub_download(pretrained_model_name, "eagle3.py")
    weights_path = hf_hub_download(pretrained_model_name, "model.safetensors")

    with open(config_path) as f:
        config = json.load(f)
    config["auto_map"] = {
        "AutoConfig": "eagle3.Eagle3SpeculatorConfig",
        "AutoModel": "eagle3.Eagle3Speculator",
    }
    if "speculators_config" in config and "verifier" in config["speculators_config"]:
        config["speculators_config"]["verifier"]["name_or_path"] = None

    with open(eagle3_path) as f:
        eagle3_code = f.read()
    eagle3_code = eagle3_code.replace(
        "def tie_weights(self):",
        "def tie_weights(self, **kwargs):",
    )

    tmp_dir = tempfile.mkdtemp()
    with open(os.path.join(tmp_dir, "eagle3.py"), "w") as f:
        f.write(eagle3_code)
    with open(os.path.join(tmp_dir, "config.json"), "w") as f:
        json.dump(config, f)
    os.symlink(weights_path, os.path.join(tmp_dir, "model.safetensors"))

    return tmp_dir


class ModelVariant(StrEnum):
    """Available EAGLE3 Speculator model variants."""

    LLAMA_3_1_8B_INSTRUCT_EAGLE3 = "3.1_8B_Instruct_EAGLE3"


class ModelLoader(ForgeModel):
    """EAGLE3 Speculator model loader for speculative decoding tasks."""

    _VARIANTS = {
        ModelVariant.LLAMA_3_1_8B_INSTRUCT_EAGLE3: LLMModelConfig(
            pretrained_model_name="RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_1_8B_INSTRUCT_EAGLE3

    sample_text = "Hey how are you doing today?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Llama_EAGLE3_Speculator",
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            VERIFIER_TOKENIZER_NAME,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        _clear_speculators_eagle3_registry()

        local_dir = _prepare_local_model_dir(pretrained_model_name)
        try:
            auto_config = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)

            model_kwargs = {"trust_remote_code": True}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs

            model = AutoModel.from_pretrained(
                local_dir,
                config=auto_config,
                verifier_attachment_mode="detached",
                **model_kwargs,
            )
        finally:
            shutil.rmtree(local_dir)

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        seq_len = inputs["input_ids"].shape[1]
        hidden_size = 4096
        hidden_dtype = dtype_override if dtype_override is not None else torch.float32
        inputs["hidden_states"] = torch.randn(
            batch_size, seq_len, 3 * hidden_size, dtype=hidden_dtype
        )

        if dtype_override is not None:
            for key in inputs:
                if inputs[key].is_floating_point():
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs

    def load_config(self):
        if self.config is not None:
            return self.config

        _clear_speculators_eagle3_registry()

        local_dir = _prepare_local_model_dir(self._variant_config.pretrained_model_name)
        try:
            self.config = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
        finally:
            shutil.rmtree(local_dir)

        return self.config
