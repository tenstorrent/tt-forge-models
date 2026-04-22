# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ministral-3-3B-Reasoning-2512 GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _patch_transformers_mistral3_gguf():
    """Monkey-patch transformers to add mistral3 GGUF architecture support.

    Transformers 5.x has Ministral3ForCausalLM (model_type='ministral3') but
    the GGUF file encodes the architecture as 'mistral3' (without the 'al').
    This patch bridges the gap so the standard from_pretrained+gguf_file path works.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )

    if "mistral3" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("mistral3")
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["mistral3"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": "head_dim",
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    }

    # Register the BPE tokenizer converter for the remapped model_type.
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "ministral3" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["ministral3"] = GGUFQwen2Converter

    _orig_load = gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(
        gguf_checkpoint_path, return_tensors=False, model_to_load=None
    ):
        result = _orig_load(
            gguf_checkpoint_path,
            return_tensors=return_tensors,
            model_to_load=model_to_load,
        )
        if result.get("config", {}).get("model_type") == "mistral3":
            result["config"]["model_type"] = "ministral3"
        return result

    gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    _orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor=None, model_type=None, num_layers=None, qual_name=""
    ):
        mt = model_type if model_type is not None else hf_model.config.model_type
        if mt == "ministral3":
            model_type = "mistral3"
        return _orig_get_map(
            hf_model,
            processor=processor,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


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
    """Available Ministral-3-3B-Reasoning-2512 GGUF model variants for causal language modeling."""

    MINISTRAL_3_3B_REASONING_2512_GGUF = "Ministral-3-3B-Reasoning-2512-GGUF"
    JENEY_MINISTRAL_3_3B_REASONING_2512_GGUF = (
        "Jeney_Ministral-3-3B-Reasoning-2512-GGUF"
    )


class ModelLoader(ForgeModel):
    """Ministral-3-3B-Reasoning-2512 GGUF model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.MINISTRAL_3_3B_REASONING_2512_GGUF: LLMModelConfig(
            pretrained_model_name="mistralai/Ministral-3-3B-Reasoning-2512-GGUF",
            max_length=128,
        ),
        ModelVariant.JENEY_MINISTRAL_3_3B_REASONING_2512_GGUF: LLMModelConfig(
            pretrained_model_name="Jeney/Ministral-3-3B-Reasoning-2512-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINISTRAL_3_3B_REASONING_2512_GGUF

    GGUF_FILE = "Ministral-3-3B-Reasoning-2512-Q4_K_M.gguf"

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
            model="Ministral-3-3B-Reasoning-2512 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _patch_transformers_mistral3_gguf()
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
        _patch_transformers_mistral3_gguf()
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
        _patch_transformers_mistral3_gguf()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
