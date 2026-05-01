# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-OSS 20B Counsel MindBuddi GGUF model loader implementation for causal language modeling.
"""
import contextlib
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


def _find_real_load_gguf_checkpoint(fn):
    """Walk a chain of monkey-patched wrappers to find the real transformers function.

    Many GGUF loaders patch load_gguf_checkpoint with fixed-arg wrappers that
    do not accept model_to_load. Walk through closure variables and module-global
    references until we reach the function whose __name__ is 'load_gguf_checkpoint'
    in the transformers module.
    """
    seen = set()
    while callable(fn):
        fn_id = id(fn)
        if fn_id in seen:
            return fn
        seen.add(fn_id)

        if (getattr(fn, '__name__', '') == 'load_gguf_checkpoint'
                and 'transformers' in getattr(fn, '__module__', '')):
            return fn

        moved = False
        # Closure pattern: good wrappers capture orig as a local variable
        if fn.__closure__:
            for cell in fn.__closure__:
                try:
                    v = cell.cell_contents
                    if callable(v) and id(v) not in seen:
                        fn = v
                        moved = True
                        break
                except ValueError:
                    pass

        # Module-global pattern: broken wrappers store the original as
        # _orig_load_gguf_checkpoint in their loader module's globals
        if not moved:
            globs = getattr(fn, '__globals__', {})
            orig = globs.get('_orig_load_gguf_checkpoint')
            if callable(orig) and id(orig) not in seen:
                fn = orig
                moved = True

        if not moved:
            return fn

    return fn


def _register_gpt_oss_support():
    """Register gpt-oss architecture as a qwen3_moe alias (idempotent)."""
    from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES
    if "gpt-oss" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    GGUF_SUPPORTED_ARCHITECTURES.append("gpt-oss")
    for section in gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3_moe" in gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            mapping = dict(gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3_moe"])
            mapping["expert_feed_forward_length"] = "moe_intermediate_size"
            mapping["attention.sliding_window"] = "sliding_window"
            gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["gpt-oss"] = mapping
    if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["gpt-oss"] = GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
    if hasattr(gguf_utils, "GGUF_CONFIG_DEFAULTS_MAPPING"):
        if "qwen3_moe" in gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING:
            gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING["gpt-oss"] = gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING["qwen3_moe"]


_register_gpt_oss_support()


@contextlib.contextmanager
def _gpt_oss_gguf_load_context():
    """Temporarily install a correct load_gguf_checkpoint at call time.

    Other GGUF loaders imported before or after this one may install fixed-arg
    patches that lack the model_to_load kwarg required by transformers 5.x.
    This context manager walks the patch chain to find the real transformers
    function, installs a correct wrapper for the duration of the call, and
    restores whatever was there before.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    import transformers.configuration_utils as _config_utils
    import transformers.models.auto.tokenization_auto as _tok_auto

    _register_gpt_oss_support()
    true_orig = _find_real_load_gguf_checkpoint(gguf_utils.load_gguf_checkpoint)

    def _patched_load(*args, **kwargs):
        result = true_orig(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "gpt-oss":
            result["config"]["model_type"] = "qwen3_moe"
        return result

    _targets = [gguf_utils, _config_utils, _tok_auto]
    saved = {mod: mod.load_gguf_checkpoint for mod in _targets
             if hasattr(mod, 'load_gguf_checkpoint')}
    for mod in saved:
        mod.load_gguf_checkpoint = _patched_load
    try:
        yield
    finally:
        for mod, fn in saved.items():
            mod.load_gguf_checkpoint = fn


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

        with _gpt_oss_gguf_load_context():
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
            with _gpt_oss_gguf_load_context():
                config = AutoConfig.from_pretrained(
                    pretrained_model_name, gguf_file=self.GGUF_FILE
                )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        with _gpt_oss_gguf_load_context():
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()

        model.config._experts_implementation = "batched_mm"

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        if self.tokenizer.chat_template is not None:
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
        else:
            text = self.sample_text
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
            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                shard_specs[mlp.experts.gate_up_proj] = (None, "model", "batch")
                shard_specs[mlp.experts.down_proj] = (None, "batch", "model")
            if hasattr(mlp, "shared_expert"):
                shard_specs[mlp.shared_expert.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_expert.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_expert.down_proj.weight] = ("batch", "model")
            if hasattr(layer, "self_attn"):
                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        return shard_specs

    def load_config(self):
        with _gpt_oss_gguf_load_context():
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
            )
        return self.config
