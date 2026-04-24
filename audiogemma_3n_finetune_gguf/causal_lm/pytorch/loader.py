# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Audiogemma 3N Finetune GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

from ....base import ForgeModel


def _patch_transformers_gemma3n_gguf():
    """Register gemma3n as a supported GGUF architecture.

    Gemma 3N uses the gemma3n GGUF architecture name. transformers 5.2.0 has the
    model class (gemma3n_text) but not the GGUF loading support. This patch adds
    the config mapping and post-processes rope_parameters and layer_types from the
    GGUF metadata fields.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "gemma3n" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("gemma3n")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["gemma3n"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "_rope_theta_full",
        "rope.freq_base_swa": "_rope_theta_swa",
        "attention.key_length": "head_dim",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.sliding_window": "sliding_window",
        "attention.sliding_window_pattern": "_sliding_window_pattern",
        "final_logit_softcapping": "final_logit_softcapping",
        "vocab_size": "vocab_size",
        "altup.active_idx": "altup_active_idx",
        "altup.num_inputs": "altup_num_inputs",
        "embedding_length_per_layer_input": "hidden_size_per_layer_input",
        "attention.shared_kv_layers": "num_kv_shared_layers",
        "attention.value_length": None,
        "activation_sparsity_scale": None,
    }

    if "gemma3" in TENSOR_PROCESSORS:
        TENSOR_PROCESSORS["gemma3n"] = TENSOR_PROCESSORS["gemma3"]

    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "gemma3_text" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["gemma3n"] = GGUF_TO_FAST_CONVERTERS["gemma3_text"]
        GGUF_TO_FAST_CONVERTERS["gemma3n_text"] = GGUF_TO_FAST_CONVERTERS["gemma3_text"]

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})

        if config.get("model_type") == "gemma3n":
            config["model_type"] = "gemma3n_text"

            rope_theta_full = config.pop("_rope_theta_full", 1000000.0)
            rope_theta_swa = config.pop("_rope_theta_swa", 10000.0)
            config["rope_parameters"] = {
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": rope_theta_swa,
                },
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": rope_theta_full,
                },
            }

            sliding_window_pattern = config.pop("_sliding_window_pattern", None)
            if sliding_window_pattern is not None:
                if not isinstance(sliding_window_pattern, list):
                    sliding_window_pattern = [sliding_window_pattern]
                config["layer_types"] = [
                    "sliding_attention" if is_sliding else "full_attention"
                    for is_sliding in sliding_window_pattern
                ]

        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # gemma3n_text is the HF model_type but gguf-py knows it as gemma3n
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type == "gemma3n_text":
            model_type = "gemma3n"
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


_patch_transformers_gemma3n_gguf()

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
    """Available Audiogemma 3N Finetune GGUF model variants for causal language modeling."""

    AUDIOGEMMA_3N_FINETUNE_GGUF = "FINETUNE_GGUF"


class ModelLoader(ForgeModel):
    """Audiogemma 3N Finetune GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.AUDIOGEMMA_3N_FINETUNE_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Audiogemma-3N-finetune-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AUDIOGEMMA_3N_FINETUNE_GGUF

    GGUF_FILE = "Audiogemma-3N-finetune.Q4_K_M.gguf"

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
            model="Audiogemma 3N Finetune GGUF",
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
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
        ).eval()

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
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
