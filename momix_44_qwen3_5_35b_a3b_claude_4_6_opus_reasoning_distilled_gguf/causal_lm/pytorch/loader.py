# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Momix-44 Qwen3.5 35B-A3B Claude 4.6 Opus Reasoning Distilled GGUF model loader implementation for causal language modeling.
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


def _patch_transformers_qwen35moe_gguf():
    """Monkey-patch transformers to add qwen35moe GGUF architecture support.

    Transformers 5.x has Qwen3_5MoeForCausalLM but lacks GGUF loading support
    for the qwen35moe architecture. The challenge is that transformers uses a
    substring check ("qwen3moe" in architecture) which incorrectly matches
    "qwen35moe", routing it to Qwen3MoeForCausalLM (per-expert weights) instead
    of the correct Qwen3_5MoeForCausalLM (packed gate_up_proj weights). We fix
    this by detecting the qwen35moe case via the full_attention_interval field
    (unique to Qwen3.5 MoE) and rerouting to qwen3_5_moe_text.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen35moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # 1. Register qwen35moe as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen35moe")

    # 2. Add config mapping for qwen35moe. Note: due to the substring match
    #    "qwen3moe" in "qwen35moe", transformers rewrites the GGUF architecture
    #    to "qwen3_moe" before key lookup. So we add full_attention_interval to
    #    the qwen3_moe mapping so it gets captured from qwen35moe GGUF files.
    GGUF_TO_TRANSFORMERS_MAPPING["config"].setdefault("qwen35moe", {}).update(
        {
            "context_length": "max_position_embeddings",
            "block_count": "num_hidden_layers",
            "feed_forward_length": "intermediate_size",
            "embedding_length": "hidden_size",
            "rope.dimension_count": None,
            "rope.freq_base": "rope_theta",
            "attention.key_length": "head_dim",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.layer_norm_rms_epsilon": "rms_norm_eps",
            "vocab_size": "vocab_size",
            "expert_count": "num_experts",
            "expert_used_count": "num_experts_per_tok",
            "full_attention_interval": "full_attention_interval",
        }
    )
    # Because qwen35moe keys become qwen3_moe keys via substring replacement,
    # we must also register full_attention_interval in the qwen3_moe mapping.
    GGUF_TO_TRANSFORMERS_MAPPING["config"].setdefault("qwen3_moe", {}).setdefault(
        "full_attention_interval", "full_attention_interval"
    )

    # 3. Register a custom tensor processor for qwen35moe.
    #    The standard Qwen2MoeTensorProcessor merges ffn_gate_exps + ffn_up_exps
    #    into gate_up_proj via GGUF_MOE_WEIGHTS_PATTERN which requires a ".weight"
    #    suffix. But qwen35moe GGUF files use packed expert tensors without
    #    ".weight" (because Qwen3_5MoeExperts uses nn.Parameter, not nn.Linear).
    #    We subclass and relax the pattern to handle both cases.
    import re

    from transformers.modeling_gguf_pytorch_utils import (
        Qwen2MoeTensorProcessor,
    )

    class Qwen35MoeTensorProcessor(Qwen2MoeTensorProcessor):
        GGUF_MOE_WEIGHTS_PATTERN = re.compile(
            r"(?P<name>.*\.ffn_(?P<w>gate|down|up)_exps)(?:\.weight)?$"
        )

    if "qwen35moe" not in TENSOR_PROCESSORS:
        TENSOR_PROCESSORS["qwen35moe"] = Qwen35MoeTensorProcessor

    # 4. Register tokenizer converter
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen35moe"] = GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
        GGUF_TO_FAST_CONVERTERS["qwen3_5_moe_text"] = GGUF_TO_FAST_CONVERTERS[
            "qwen3_moe"
        ]

    # 5. Patch load_gguf_checkpoint to detect qwen35moe (disguised as qwen3_moe
    #    due to the substring match) and convert it to qwen3_5_moe_text so the
    #    correct Qwen3_5MoeForCausalLM is instantiated with packed expert weights.
    orig_load = gguf_utils.load_gguf_checkpoint

    def _is_qwen35moe_gguf(gguf_path):
        """Check if a GGUF file has qwen35moe architecture."""
        try:
            from gguf import GGUFReader
            from transformers.modeling_gguf_pytorch_utils import read_field

            reader = GGUFReader(gguf_path)
            arch = read_field(reader, "general.architecture")
            return bool(arch) and arch[0] == "qwen35moe"
        except Exception:
            return False

    def patched_load_gguf_checkpoint(
        gguf_checkpoint_path, return_tensors=False, **kwargs
    ):
        # Detect qwen35moe by reading the GGUF architecture field directly.
        # We cannot rely on the transformed model_type because transformers
        # uses a substring match ("qwen3moe" in "qwen35moe") that misidentifies
        # qwen35moe as qwen3_moe before we can intercept it.
        is_qwen35moe = _is_qwen35moe_gguf(gguf_checkpoint_path)
        result = orig_load(
            gguf_checkpoint_path, return_tensors=return_tensors, **kwargs
        )
        if is_qwen35moe:
            config = result.get("config", {})
            config["model_type"] = "qwen3_5_moe_text"
            num_layers = config.get("num_hidden_layers", 40)
            interval = config.pop("full_attention_interval", 4)
            layer_types = []
            for i in range(num_layers):
                if (i + 1) % interval == 0:
                    layer_types.append("full_attention")
                else:
                    layer_types.append("linear_attention")
            config["layer_types"] = layer_types
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # Also patch modules that imported load_gguf_checkpoint directly
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # 6. Patch get_gguf_hf_weights_map to handle qwen3_5_moe_text -> qwen3moe.
    #    We use "qwen3moe" (arch 30) not "qwen35moe" (arch 35) because in the
    #    qwen35moe name map, gate_up_proj maps to ffn_gate_up_exps (which does
    #    not exist in the GGUF file). In the qwen3moe name map, gate_up_proj
    #    returns None, triggering the perform_fallback_tensor_mapping which adds
    #    ffn_gate_exps and ffn_up_exps — matching the actual GGUF tensor names.
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type in ("qwen3_5_moe_text", "qwen3_5_moe"):
            model_type = "qwen3moe"
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


# Apply the monkey-patch at import time
_patch_transformers_qwen35moe_gguf()


class ModelVariant(StrEnum):
    """Available Momix-44 Qwen3.5 35B-A3B Claude 4.6 Opus Reasoning Distilled GGUF model variants for causal language modeling."""

    QWEN3_5_35B_A3B_CLAUDE_4_6_OPUS_REASONING_DISTILLED_GGUF = "35B_A3B_GGUF"


class ModelLoader(ForgeModel):
    """Momix-44 Qwen3.5 35B-A3B Claude 4.6 Opus Reasoning Distilled GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_5_35B_A3B_CLAUDE_4_6_OPUS_REASONING_DISTILLED_GGUF: LLMModelConfig(
            pretrained_model_name="Momix-44/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = (
        ModelVariant.QWEN3_5_35B_A3B_CLAUDE_4_6_OPUS_REASONING_DISTILLED_GGUF
    )

    GGUF_FILE = "Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-Q4_K_M.gguf"

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
            model="Momix-44 Qwen3.5 35B-A3B Claude 4.6 Opus Reasoning Distilled GGUF",
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
        if self.tokenizer.chat_template is None:
            # qwen35moe GGUF tokenizers may not include a chat template.
            # Fall back to the standard Qwen3 im_start/im_end format.
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{'<|im_start|>assistant\n'}}"
                "{% endif %}"
            )

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
            if hasattr(config, "text_config"):
                config.text_config.num_hidden_layers = self.num_layers
                if hasattr(config.text_config, "layer_types"):
                    config.text_config.layer_types = config.text_config.layer_types[
                        : self.num_layers
                    ]
            else:
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
            if hasattr(layer, "self_attn"):
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
