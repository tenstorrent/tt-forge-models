# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DevQuasar arcee-ai Trinity Large Preview GGUF model loader implementation for causal language modeling.
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


def _patch_transformers_afmoe_gguf():
    """Monkey-patch transformers to add afmoe GGUF architecture support.

    Transformers 5.x has AfmoeForCausalLM but the GGUF loading path does not
    recognize the 'afmoe' architecture name.  We register the config key
    mapping, tokenizer converter, and a tensor processor that splits the
    stacked per-expert weight tensors into the individual tensors expected by
    AfmoeExperts (a ModuleList of AfmoeMLP instances).
    """
    import re
    import numpy as np

    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        TensorProcessor,
        GGUFTensor,
    )
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFLlamaConverter,
    )

    if "afmoe" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    # 1. Register config field mapping (GGUF suffix → AfmoeConfig param name).
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["afmoe"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "expert_feed_forward_length": "moe_intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": "head_dim",
        "attention.sliding_window": "sliding_window",
        "attention.sliding_window_pattern": "global_attn_every_n_layers",
        "vocab_size": "vocab_size",
        "expert_count": "num_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_shared_count": "num_shared_experts",
        "leading_dense_block_count": "num_dense_layers",
    }
    GGUF_SUPPORTED_ARCHITECTURES.append("afmoe")

    # 2. Tokenizer: arcee-ai Trinity uses a Llama-3 BPE tokenizer.
    if "afmoe" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["afmoe"] = GGUFLlamaConverter

    # 3. Tensor processor: split stacked expert tensors into per-expert weights.
    # GGUF stores all expert projections stacked:
    #   blk.N.ffn_{gate,up,down}_exps  shape [num_experts, ...]
    # The HF AfmoeExperts ModuleList expects individual tensors:
    #   model.layers.N.mlp.experts.{i}.{gate,up,down}_proj.weight
    class AfmoeMoETensorProcessor(TensorProcessor):
        _STACKED = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)_exps(?:\.weight)?$")

        def __init__(self, config=None):
            super().__init__(config=config)
            self._num_experts = (config or {}).get("num_experts", 64)

        def perform_fallback_tensor_mapping(
            self, gguf_to_hf_name_map, suffix, qual_name, hf_name
        ):
            # Individual expert tensors are handled entirely in process(); add a
            # sentinel entry so tensor_key_mapping includes the stacked name and
            # load_gguf_checkpoint calls process() for it even if the HF model's
            # state_dict did not produce a mapping via the normal channel.
            m = re.match(
                r"model\.layers\.(\d+)\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)",
                hf_name,
            )
            if m:
                bid = m.group(1)
                w = m.group(2).replace("_proj", "")
                key = f"blk.{bid}.ffn_{w}_exps{suffix}"
                if key not in gguf_to_hf_name_map:
                    # Sentinel value; the actual write is done inside process().
                    gguf_to_hf_name_map[key] = f"__afmoe_stacked__{bid}__{w}"

        def process(self, weights, name, **kwargs):
            m = self._STACKED.match(name)
            if m:
                bid, proj = m.group(1), m.group(2)
                parsed_parameters = kwargs.get("parsed_parameters")
                if parsed_parameters is not None:
                    for exp_id in range(self._num_experts):
                        hf_key = f"model.layers.{bid}.mlp.experts.{exp_id}.{proj}_proj.weight"
                        parsed_parameters["tensors"][hf_key] = torch.from_numpy(
                            np.copy(weights[exp_id])
                        )
                return GGUFTensor(weights, None, {})
            return GGUFTensor(weights, name, {})

    TENSOR_PROCESSORS["afmoe"] = AfmoeMoETensorProcessor


# Apply the patch at import time.
_patch_transformers_afmoe_gguf()


class ModelVariant(StrEnum):
    """Available DevQuasar arcee-ai Trinity Large Preview GGUF model variants for causal language modeling."""

    TRINITY_LARGE_PREVIEW_Q4_K_M_GGUF = "TRINITY_LARGE_PREVIEW_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """DevQuasar arcee-ai Trinity Large Preview GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.TRINITY_LARGE_PREVIEW_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="DevQuasar/arcee-ai.Trinity-Large-Preview-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TRINITY_LARGE_PREVIEW_Q4_K_M_GGUF

    GGUF_FILE = "arcee-ai.Trinity-Large-Preview.Q4_K_M-00001-of-00016.gguf"

    sample_text = "Give me a short introduction to large language models."

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
            model="DevQuasar arcee-ai Trinity Large Preview GGUF",
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
            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                shard_specs[mlp.experts.gate_up_proj] = (None, "model", "batch")
                shard_specs[mlp.experts.down_proj] = (None, "batch", "model")
            if hasattr(mlp, "shared_expert"):
                shard_specs[mlp.shared_expert.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_expert.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_expert.down_proj.weight] = ("batch", "model")

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
