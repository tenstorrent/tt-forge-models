# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
xihc-ucb/Qwen2.5-7B-train-Quasar-1214 model loader implementation for causal language modeling.
"""
import glob
import os
import sys

# Inject stub quasar package before transformers loads the model's custom code.
# The model requires a private FP8 quantization library (quasar) not on PyPI;
# the stub provides compatible interfaces for instantiation and compilation.
_stub_path = os.path.join(os.path.dirname(__file__), "quasar_stub")
if _stub_path not in sys.path:
    sys.path.insert(0, _stub_path)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _patch_fp8_qwen2_config():
    """Patch cached FP8Qwen2 custom code for transformers >= 5.x compatibility.

    Fixes two issues:
    1. FP8Qwen2Config.from_dict: super().from_dict() returns (config, unused_kwargs)
       when called with return_unused_kwargs=True.
    2. modeling_fp8_qwen2.py imports check_model_inputs which no longer exists.
    """
    import importlib
    import shutil

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    modules_dir = os.path.join(hf_home, "modules", "transformers_modules")

    # Patch configuration_fp8_qwen2.py
    for config_path in glob.glob(
        os.path.join(modules_dir, "*", "*", "*", "configuration_fp8_qwen2.py")
    ):
        with open(config_path) as f:
            content = f.read()
        if "return_unused_kwargs" not in content:
            content = content.replace(
                "    @classmethod\n    def from_dict(cls, config_dict, **kwargs):\n"
                "        config = super().from_dict(config_dict, **kwargs)\n"
                "        \n"
                "        fp8_config",
                "    @classmethod\n    def from_dict(cls, config_dict, **kwargs):\n"
                '        return_unused_kwargs = kwargs.get("return_unused_kwargs", False)\n'
                "        result = super().from_dict(config_dict, **kwargs)\n"
                "        config, unused_kwargs = result if return_unused_kwargs else (result, {})\n"
                "        \n"
                "        fp8_config",
            ).replace(
                "        config.fp8_config = FP8Config(**fp8_config)\n        return config",
                "        config.fp8_config = FP8Config(**fp8_config)\n"
                "        if return_unused_kwargs:\n"
                "            return config, unused_kwargs\n"
                "        return config",
            )
            with open(config_path, "w") as f:
                f.write(content)
            pycache = os.path.join(os.path.dirname(config_path), "__pycache__")
            if os.path.isdir(pycache):
                shutil.rmtree(pycache)

    # Patch modeling_fp8_qwen2.py: check_model_inputs removed in transformers 5.x
    for model_path in glob.glob(
        os.path.join(modules_dir, "*", "*", "*", "modeling_fp8_qwen2.py")
    ):
        with open(model_path) as f:
            content = f.read()
        if "check_model_inputs" in content:
            content = content.replace(
                "from transformers.utils.generic import check_model_inputs",
                "try:\n"
                "    from transformers.utils.generic import check_model_inputs\n"
                "except ImportError:\n"
                "    check_model_inputs = lambda *a, **k: None",
            )
            with open(model_path, "w") as f:
                f.write(content)
            pycache = os.path.join(os.path.dirname(model_path), "__pycache__")
            if os.path.isdir(pycache):
                shutil.rmtree(pycache)

    importlib.invalidate_caches()
    for mod_name in list(sys.modules.keys()):
        if "configuration_fp8_qwen2" in mod_name or "modeling_fp8_qwen2" in mod_name:
            del sys.modules[mod_name]


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
    """Available xihc-ucb/Qwen2.5-7B-train-Quasar-1214 model variants for causal language modeling."""

    XIHC_UCB_QWEN2_5_7B_TRAIN_QUASAR_1214 = "xihc_ucb_qwen2_5_7b_train_quasar_1214"


class ModelLoader(ForgeModel):
    """xihc-ucb/Qwen2.5-7B-train-Quasar-1214 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.XIHC_UCB_QWEN2_5_7B_TRAIN_QUASAR_1214: LLMModelConfig(
            pretrained_model_name="xihc-ucb/Qwen2.5-7B-train-Quasar-1214",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XIHC_UCB_QWEN2_5_7B_TRAIN_QUASAR_1214

    sample_text = "Give me a short introduction to large language model."

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
            model="xihc-ucb/Qwen2.5-7B-train-Quasar-1214",
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
            self._variant_config.pretrained_model_name,
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

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        # Retry loop: each iteration patches a compatibility issue and retries.
        # Attempt 1 may fail with AttributeError (from_dict tuple return).
        # Attempt 2 may fail with ImportError (check_model_inputs missing).
        # Both are fixed by _patch_fp8_qwen2_config() which patches cached files.
        model = None
        for attempt in range(3):
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name, trust_remote_code=True, **model_kwargs
                ).eval()
                break
            except (AttributeError, ImportError) as e:
                if "fp8_config" not in str(e) and "check_model_inputs" not in str(e):
                    raise
                if attempt >= 2:
                    raise
                _patch_fp8_qwen2_config()
        assert model is not None

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

    def load_config(self):
        for attempt in range(3):
            try:
                self.config = AutoConfig.from_pretrained(
                    self._variant_config.pretrained_model_name, trust_remote_code=True
                )
                break
            except (AttributeError, ImportError) as e:
                if "fp8_config" not in str(e) and "check_model_inputs" not in str(e):
                    raise
                if attempt >= 2:
                    raise
                _patch_fp8_qwen2_config()
        return self.config
