# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek Coder V2 Lite Instruct GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

from ....base import ForgeModel


def _patch_transformers_deepseek2_gguf():
    """Monkey-patch transformers to add deepseek2 GGUF architecture support.

    transformers lacks the config mapping and architecture registration needed
    to load deepseek2 GGUF checkpoints.  Also adds deepseek_v2 to the fast
    tokenizer converters because load_gguf_checkpoint renames the model_type
    from deepseek2 to deepseek_v2 after processing.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "deepseek2" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("deepseek2")

        GGUF_TO_TRANSFORMERS_MAPPING["config"]["deepseek2"] = {
            "context_length": "max_position_embeddings",
            "block_count": "num_hidden_layers",
            "feed_forward_length": "intermediate_size",
            "embedding_length": "hidden_size",
            "rope.freq_base": "rope_theta",
            "rope.dimension_count": "qk_rope_head_dim",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.layer_norm_rms_epsilon": "rms_norm_eps",
            "attention.key_length": None,
            "attention.value_length": None,
            "attention.key_length_mla": "qk_nope_head_dim",
            "attention.value_length_mla": "v_head_dim",
            "attention.q_lora_rank": "q_lora_rank",
            "attention.kv_lora_rank": "kv_lora_rank",
            "vocab_size": "vocab_size",
            "expert_count": "n_routed_experts",
            "expert_used_count": "num_experts_per_tok",
            "expert_shared_count": "n_shared_experts",
            "expert_group_count": "n_group",
            "expert_group_used_count": "topk_group",
            "expert_weights_scale": "routed_scaling_factor",
            "expert_weights_norm": "norm_topk_prob",
            "leading_dense_block_count": "first_k_dense_replace",
            "expert_feed_forward_length": "moe_intermediate_size",
        }

        orig_load = gguf_utils.load_gguf_checkpoint

        def patched_load_gguf_checkpoint(*args, **kwargs):
            result = orig_load(*args, **kwargs)
            config = result.get("config", {})
            if config.get("model_type") == "deepseek2":
                config["model_type"] = "deepseek_v2"
            return result

        gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

        import transformers.models.auto.tokenization_auto as tok_auto
        import transformers.configuration_utils as config_utils
        import transformers.modeling_utils as modeling_utils

        for mod in (tok_auto, config_utils, modeling_utils):
            if hasattr(mod, "load_gguf_checkpoint"):
                mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "deepseek2" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["deepseek2"] = GGUFQwen2Converter
    # load_gguf_checkpoint renames deepseek2 → deepseek_v2 in model_type;
    # the tokenizer path uses model_type as the architecture key, so both
    # names need to be present in GGUF_TO_FAST_CONVERTERS.
    if "deepseek_v2" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["deepseek_v2"] = GGUFQwen2Converter

    # get_gguf_hf_weights_map looks up model_type in gguf.MODEL_ARCH_NAMES to
    # find the gguf arch enum.  After the load_gguf_checkpoint patch above the
    # model_type is "deepseek_v2", but MODEL_ARCH_NAMES only has "deepseek2".
    # Patch the function to remap "deepseek_v2" → "deepseek2" before the lookup.
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils2

    orig_get_map = _gguf_utils2.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type == "deepseek_v2":
            model_type = "deepseek2"
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    _gguf_utils2.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


_patch_transformers_deepseek2_gguf()
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
    """Available DeepSeek Coder V2 Lite Instruct GGUF model variants for causal language modeling."""

    DEEPSEEK_CODER_V2_LITE_INSTRUCT_GGUF = "Coder_V2_Lite_Instruct_GGUF"
    DEEPSEEK_CODER_V2_LITE_13B_INSTRUCT_SFT_MATH7K_I1_GGUF = (
        "Coder_V2_Lite_13B_Instruct_sft_math7k_i1_GGUF"
    )


class ModelLoader(ForgeModel):
    """DeepSeek Coder V2 Lite Instruct GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_CODER_V2_LITE_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="second-state/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            max_length=128,
        ),
        ModelVariant.DEEPSEEK_CODER_V2_LITE_13B_INSTRUCT_SFT_MATH7K_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Deepseek-Coder-V2-Lite-13B-Instruct-sft-math7k-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_CODER_V2_LITE_INSTRUCT_GGUF

    _GGUF_FILES = {
        ModelVariant.DEEPSEEK_CODER_V2_LITE_INSTRUCT_GGUF: "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
        ModelVariant.DEEPSEEK_CODER_V2_LITE_13B_INSTRUCT_SFT_MATH7K_I1_GGUF: "Deepseek-Coder-V2-Lite-13B-Instruct-sft-math7k.i1-Q4_K_M.gguf",
    }

    sample_text = "Write a bubble sort algorithm in Python."

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

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
            model="DeepSeek Coder V2 Lite Instruct GGUF",
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
        tokenizer_kwargs["gguf_file"] = self.gguf_file

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
        model_kwargs["gguf_file"] = self.gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.gguf_file
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # The GGUF tokenizer may include extra special tokens beyond vocab_size;
        # resize embeddings so their IDs don't cause out-of-range errors.
        if len(self.tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(self.tokenizer))

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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
