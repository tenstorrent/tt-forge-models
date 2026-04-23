# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


def _apply_gguf_patches():
    """Register nemotron_h_moe GGUF support in transformers and fix tensor-name mapping.

    Must be called before any GGUF file operation (tokenizer or model load).
    Safe to call multiple times; idempotent thanks to the _PATCHED sentinel.
    """
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils

    if getattr(_gguf_utils, "_nemotron_h_moe_patched", False):
        return

    # Register nemotron_h_moe config field mappings so load_gguf_checkpoint
    # can parse the GGUF metadata into the correct HF config keys.
    from transformers.integrations.ggml import GGUF_CONFIG_MAPPING

    if "nemotron_h_moe" not in GGUF_CONFIG_MAPPING:
        GGUF_CONFIG_MAPPING["nemotron_h_moe"] = {
            "block_count": "num_hidden_layers",
            "embedding_length": "hidden_size",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.key_length": "head_dim",
            "attention.layer_norm_rms_epsilon": "layer_norm_epsilon",
            "context_length": "max_position_embeddings",
            "vocab_size": "vocab_size",
            "expert_count": "n_routed_experts",
            "expert_used_count": "num_experts_per_tok",
            "expert_feed_forward_length": "moe_intermediate_size",
            "expert_shared_feed_forward_length": "moe_shared_expert_intermediate_size",
            "expert_shared_count": "n_shared_experts",
            "expert_group_count": "n_group",
            "expert_weights_norm": "norm_topk_prob",
            "moe_latent_size": "moe_latent_size",
            "ssm.state_size": "ssm_state_size",
            "ssm.group_count": "n_groups",
            "ssm.conv_kernel": "conv_kernel",
        }

    if "nemotron_h_moe" not in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
        _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("nemotron_h_moe")

    # The gguf library assigns model_type "nemotron_h" (arch key 81) to the
    # pure-SSM variant and "nemotron_h_moe" (arch key 82) to the MoE variant.
    # transformers uses model_type="nemotron_h" for NemotronHForCausalLM, so
    # get_gguf_hf_weights_map ends up using arch key 81 which lacks MoE tensor
    # names (ffn_*_exps etc.).  Remap to key 82 so all GGUF weights are found.
    _orig_get_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_get_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        effective_type = model_type
        if effective_type is None and hasattr(hf_model, "config"):
            effective_type = hf_model.config.model_type
        if effective_type == "nemotron_h":
            effective_type = "nemotron_h_moe"
        return _orig_get_map(
            hf_model,
            processor,
            model_type=effective_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_map

    # load_gguf_checkpoint returns model_type="nemotron_h_moe" (from the GGUF
    # general.architecture field) but transformers only knows "nemotron_h".
    # Wrap load_gguf_checkpoint everywhere it's imported at module level so the
    # returned config dict always has model_type="nemotron_h".
    _orig_load_gguf = _gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf(*args, **kwargs):
        result = _orig_load_gguf(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "nemotron_h_moe":
            result["config"]["model_type"] = "nemotron_h"
        return result

    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf

    # Propagate the patch to all modules that imported load_gguf_checkpoint
    # at their own module level (top-of-file imports keep a direct reference).
    for _mod_name in (
        "transformers.configuration_utils",
        "transformers.models.auto.tokenization_auto",
        "transformers.tokenization_utils_tokenizers",
    ):
        import sys

        if _mod_name in sys.modules:
            _mod = sys.modules[_mod_name]
            if hasattr(_mod, "load_gguf_checkpoint"):
                _mod.load_gguf_checkpoint = _patched_load_gguf

    _gguf_utils._nemotron_h_moe_patched = True


def _build_nemotron_h_config(
    pretrained_model_name: str, gguf_file: str
) -> "NemotronHConfig":
    """Build NemotronHConfig from GGUF metadata.

    The per-layer feed_forward_length and attention.head_count_kv arrays encode
    the layer type pattern: attention layers have kv>0, moe layers have ffl>0,
    mamba layers have both equal to 0.
    """
    from huggingface_hub import hf_hub_download
    from gguf import GGUFReader
    from transformers import NemotronHConfig

    local_path = hf_hub_download(
        repo_id=pretrained_model_name,
        filename=gguf_file,
        repo_type="model",
    )

    reader = GGUFReader(local_path)

    def _scalar(field_name):
        field = reader.fields[field_name]
        return field.parts[field.data[0]].tolist()[0]

    def _array(field_name):
        field = reader.fields[field_name]
        return [field.parts[i].tolist()[0] for i in field.data]

    ffl = _array("nemotron_h_moe.feed_forward_length")
    kv = _array("nemotron_h_moe.attention.head_count_kv")

    layers_block_type = []
    for f, k in zip(ffl, kv):
        if k > 0:
            layers_block_type.append("attention")
        elif f > 0:
            layers_block_type.append("moe")
        else:
            layers_block_type.append("mamba")

    hidden_size = _scalar("nemotron_h_moe.embedding_length")
    ssm_inner_size = _scalar("nemotron_h_moe.ssm.inner_size")

    return NemotronHConfig(
        vocab_size=_scalar("nemotron_h_moe.vocab_size"),
        hidden_size=hidden_size,
        layers_block_type=layers_block_type,
        num_attention_heads=_scalar("nemotron_h_moe.attention.head_count"),
        num_key_value_heads=max(kv),
        head_dim=_scalar("nemotron_h_moe.attention.key_length"),
        max_position_embeddings=_scalar("nemotron_h_moe.context_length"),
        layer_norm_epsilon=_scalar("nemotron_h_moe.attention.layer_norm_rms_epsilon"),
        ssm_state_size=_scalar("nemotron_h_moe.ssm.state_size"),
        n_groups=_scalar("nemotron_h_moe.ssm.group_count"),
        conv_kernel=_scalar("nemotron_h_moe.ssm.conv_kernel"),
        expand=ssm_inner_size // hidden_size,
        n_routed_experts=_scalar("nemotron_h_moe.expert_count"),
        num_experts_per_tok=_scalar("nemotron_h_moe.expert_used_count"),
        moe_intermediate_size=_scalar("nemotron_h_moe.expert_feed_forward_length"),
        moe_shared_expert_intermediate_size=_scalar(
            "nemotron_h_moe.expert_shared_feed_forward_length"
        ),
        moe_latent_size=_scalar("nemotron_h_moe.moe_latent_size"),
        n_shared_experts=_scalar("nemotron_h_moe.expert_shared_count"),
        n_group=_scalar("nemotron_h_moe.expert_group_count"),
        norm_topk_prob=bool(_scalar("nemotron_h_moe.expert_weights_norm")),
    )


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
    """Available AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model variants for causal language modeling."""

    NEMOTRON_3_SUPER_120B_A12B_Q4_K_M_GGUF = "3_Super_120B_A12B_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_3_SUPER_120B_A12B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="AesSedai/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_3_SUPER_120B_A12B_Q4_K_M_GGUF

    GGUF_FILE = (
        "Q4_K_M/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-Q4_K_M-00001-of-00003.gguf"
    )

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
            model="Nemotron 3 Super 120B A12B AesSedai GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _apply_gguf_patches()

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
        _apply_gguf_patches()

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        config = _build_nemotron_h_config(pretrained_model_name, self.GGUF_FILE)

        if self.num_layers is not None:
            config.layers_block_type = config.layers_block_type[: self.num_layers]

        model_kwargs = {"config": config, "gguf_file": self.GGUF_FILE}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

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

    def load_config(self):
        _apply_gguf_patches()
        self.config = _build_nemotron_h_config(
            self._variant_config.pretrained_model_name, self.GGUF_FILE
        )
        return self.config
