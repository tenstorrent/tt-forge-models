# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-OSS 20B Counsel MindBuddi GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional
from contextlib import contextmanager

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS


def _find_true_load_gguf_checkpoint():
    """Return the unpatched load_gguf_checkpoint from transformers.

    Several loaders patch load_gguf_checkpoint at import time without the
    model_to_load kwarg required by transformers 5.x.  Early-imported patching
    loaders (e.g. bartowski_coniccat) save the True Original as
    _orig_load_gguf_checkpoint.  We search sys.modules for that reference.
    """
    import sys
    for mod in list(sys.modules.values()):
        fn = getattr(mod, "_orig_load_gguf_checkpoint", None)
        if (
            fn is not None
            and callable(fn)
            and getattr(fn, "__module__", "") == "transformers.modeling_gguf_pytorch_utils"
            and getattr(fn, "__qualname__", "") == "load_gguf_checkpoint"
        ):
            return fn
    return _gguf_utils.load_gguf_checkpoint


def _patch_gpt_oss_support():
    """Register gpt-oss architecture as an alias for qwen3_moe.

    GPT-OSS uses the same model architecture as Qwen3 MoE but the GGUF file
    declares architecture as 'gpt-oss' which transformers does not recognise.
    """
    if "gpt-oss" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("gpt-oss")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3_moe" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            mapping = dict(
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3_moe"]
            )
            mapping["expert_feed_forward_length"] = "moe_intermediate_size"
            mapping["attention.sliding_window"] = "sliding_window"
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["gpt-oss"] = mapping
    if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["gpt-oss"] = GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
    if hasattr(_gguf_utils, "GGUF_CONFIG_DEFAULTS_MAPPING"):
        if "qwen3_moe" in _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING:
            _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING[
                "gpt-oss"
            ] = _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING["qwen3_moe"]


_patch_gpt_oss_support()


@contextmanager
def _gguf_for_gpt_oss():
    """Install a correct load_gguf_checkpoint on all binding sites for from_pretrained.

    The context manager:
    - Uses the True Original (found via sys.modules) to accept model_to_load
    - Registers gpt-oss architecture so the GGUF reader doesn't reject it
    - Remaps model_type "gpt-oss" -> "qwen3_moe" so AutoConfig resolves correctly
    Restores previous bindings on exit.
    """

    def _patched(gguf_path, return_tensors=False, model_to_load=None):
        _patch_gpt_oss_support()
        true_fn = _find_true_load_gguf_checkpoint()
        result = true_fn(
            gguf_path, return_tensors=return_tensors, model_to_load=model_to_load
        )
        if result.get("config", {}).get("model_type") == "gpt-oss":
            result["config"]["model_type"] = "qwen3_moe"
        return result

    _sites = [
        (_gguf_utils, "load_gguf_checkpoint"),
        (_config_utils, "load_gguf_checkpoint"),
        (_auto_tokenizer, "load_gguf_checkpoint"),
        (_tok_utils, "load_gguf_checkpoint"),
    ]
    _saved = [(m, n, getattr(m, n, None)) for m, n in _sites]
    try:
        for m, n, _ in _saved:
            setattr(m, n, _patched)
        yield
    finally:
        for m, n, orig in _saved:
            if orig is not None:
                setattr(m, n, orig)


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
    """Available GPT-OSS 20B Counsel MindBuddi GGUF model variants for causal language modeling."""

    GPT_OSS_20B_COUNSEL_MINDBUDDI_GGUF = "20B_Counsel_MindBuddi_GGUF"


class ModelLoader(ForgeModel):
    """GPT-OSS 20B Counsel MindBuddi GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GPT_OSS_20B_COUNSEL_MINDBUDDI_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/gpt-oss-20b-counsel-MindBuddi-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GPT_OSS_20B_COUNSEL_MINDBUDDI_GGUF

    GGUF_FILE = "gpt-oss-20b-counsel-MindBuddi.Q4_K_M.gguf"

    sample_text = "What is your favorite city?"

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
            model="GPT-OSS 20B Counsel MindBuddi GGUF",
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
            with _gguf_for_gpt_oss():
                config = AutoConfig.from_pretrained(
                    pretrained_model_name, gguf_file=self.GGUF_FILE
                )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        with _gguf_for_gpt_oss():
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

        if self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": self.sample_text}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts = [text]
        else:
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
        return shard_specs

    def load_config(self):
        with _gguf_for_gpt_oss():
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
            )
        return self.config
