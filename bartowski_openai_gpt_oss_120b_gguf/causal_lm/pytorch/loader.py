# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
bartowski openai gpt-oss 120B GGUF model loader implementation for causal language modeling.
"""
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


def _patch_gpt_oss_support():
    """Register gpt-oss architecture as an alias for qwen3_moe.

    GPT-OSS 120B uses the same model architecture as Qwen3 MoE but the GGUF
    file declares architecture as 'gpt-oss' which transformers does not
    recognise.
    """
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils
    from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

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


def _get_real_load_gguf_checkpoint(fn):
    """Walk the patch chain to find the real transformers load_gguf_checkpoint."""
    import inspect

    seen = set()
    current = fn
    while True:
        fn_id = id(current)
        if fn_id in seen or not callable(current) or not hasattr(current, "__code__"):
            return current
        seen.add(fn_id)
        if (
            getattr(current, "__module__", "")
            == "transformers.modeling_gguf_pytorch_utils"
        ):
            return current
        try:
            if "model_to_load" in inspect.signature(current).parameters:
                return current
        except (ValueError, TypeError):
            pass
        freevars = current.__code__.co_freevars
        cells = current.__closure__ or ()
        next_fn = None
        for i, varname in enumerate(freevars):
            if i >= len(cells):
                break
            if "load_gguf_checkpoint" in varname or "orig_load" in varname:
                try:
                    v = cells[i].cell_contents
                    if callable(v) and id(v) not in seen:
                        next_fn = v
                        break
                except ValueError:
                    pass
        if next_fn is None:
            v = getattr(current, "__globals__", {}).get("_orig_load_gguf_checkpoint")
            if v is not None and callable(v) and id(v) not in seen:
                next_fn = v
        if next_fn is None:
            return current
        current = next_fn


def _install_gpt_oss_patch():
    """Install model_to_load-aware patch for gpt-oss GGUF loading."""
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils

    _patch_gpt_oss_support()

    _chain_fn = _gguf_utils.load_gguf_checkpoint
    _real_fn = _get_real_load_gguf_checkpoint(_chain_fn)

    def patched_load_gguf_checkpoint(
        gguf_checkpoint_path, return_tensors=False, model_to_load=None, **kwargs
    ):
        _patch_gpt_oss_support()
        if return_tensors:
            # Bypass old-API chain; call real transformers function directly
            return _real_fn(
                gguf_checkpoint_path,
                return_tensors=True,
                model_to_load=model_to_load,
                **kwargs,
            )
        result = _chain_fn(gguf_checkpoint_path, return_tensors=False)
        if result.get("config", {}).get("model_type") == "gpt-oss":
            result["config"]["model_type"] = "qwen3_moe"
        return result

    _gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    return patched_load_gguf_checkpoint


_GPT_OSS_GGUF_LOAD_FN = _install_gpt_oss_patch()


class ModelVariant(StrEnum):
    """Available bartowski openai gpt-oss 120B GGUF model variants for causal language modeling."""

    BARTOWSKI_OPENAI_GPT_OSS_120B_MXFP4_MOE_GGUF = (
        "BARTOWSKI_OPENAI_GPT_OSS_120B_MXFP4_MOE_GGUF"
    )


class ModelLoader(ForgeModel):
    """bartowski openai gpt-oss 120B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.BARTOWSKI_OPENAI_GPT_OSS_120B_MXFP4_MOE_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/openai_gpt-oss-120b-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BARTOWSKI_OPENAI_GPT_OSS_120B_MXFP4_MOE_GGUF

    GGUF_FILE = "openai_gpt-oss-120b-MXFP4_MOE/openai_gpt-oss-120b-MXFP4_MOE-00001-of-00002.gguf"

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
            model="bartowski openai gpt-oss 120B GGUF",
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

        import transformers.modeling_gguf_pytorch_utils as _gguf_mod

        _saved_gguf_fn = _gguf_mod.load_gguf_checkpoint
        if _GPT_OSS_GGUF_LOAD_FN is not None:
            _gguf_mod.load_gguf_checkpoint = _GPT_OSS_GGUF_LOAD_FN
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()
        finally:
            _gguf_mod.load_gguf_checkpoint = _saved_gguf_fn

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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
