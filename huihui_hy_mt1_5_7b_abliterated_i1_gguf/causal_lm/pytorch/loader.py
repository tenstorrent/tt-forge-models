# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui HY-MT1.5 7B abliterated i1 GGUF model loader implementation for causal language modeling.
"""
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _patch_transformers_hunyuan_dense_gguf():
    """Monkey-patch transformers to add hunyuan-dense GGUF architecture support.

    Transformers 5.x has HunYuanDenseV1ForCausalLM (model_type='hunyuan_v1_dense') but
    lacks GGUF loading support for the 'hunyuan-dense' architecture.  The gguf library
    already knows about hunyuan-dense tensor names, so we only need to bridge the config
    and tensor-mapping layers.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        get_gguf_hf_weights_map,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "hunyuan-dense" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("hunyuan-dense")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["hunyuan-dense"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": "head_dim",
        "rope.freq_base": "rope_theta",
        "vocab_size": "vocab_size",
    }

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFGPTConverter,
    )

    if "hunyuan_v1_dense" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["hunyuan_v1_dense"] = GGUFGPTConverter

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "hunyuan-dense":
            result["config"]["model_type"] = "hunyuan_v1_dense"
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # patch tokenizer fast path which also calls load_gguf_checkpoint directly
    try:
        import transformers.tokenization_utils_tokenizers as tok_tokenizers

        if hasattr(tok_tokenizers, "load_gguf_checkpoint"):
            tok_tokenizers.load_gguf_checkpoint = patched_load_gguf_checkpoint
    except Exception:
        pass

    orig_get_weights_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(hf_model, processor, model_type=None, **kw):
        mt = hf_model.config.model_type if model_type is None else model_type
        if mt == "hunyuan_v1_dense":
            model_type = "hunyuan-dense"
        return orig_get_weights_map(hf_model, processor, model_type=model_type, **kw)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


_patch_transformers_hunyuan_dense_gguf()

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


class ModelVariant(StrEnum):
    """Available Huihui HY-MT1.5 7B abliterated i1 GGUF model variants for causal language modeling."""

    HUIHUI_HY_MT1_5_7B_ABLITERATED_I1_GGUF = "HUIHUI_HY_MT1_5_7B_ABLITERATED_I1_GGUF"


class ModelLoader(ForgeModel):
    """Huihui HY-MT1.5 7B abliterated i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_HY_MT1_5_7B_ABLITERATED_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-HY-MT1.5-7B-abliterated-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_HY_MT1_5_7B_ABLITERATED_I1_GGUF

    GGUF_FILE = "Huihui-HY-MT1.5-7B-abliterated.i1-Q4_K_M.gguf"

    sample_text = (
        "Translate the following segment into Chinese: The weather is nice today."
    )

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
            model="Huihui HY-MT1.5 7B abliterated i1 GGUF",
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
