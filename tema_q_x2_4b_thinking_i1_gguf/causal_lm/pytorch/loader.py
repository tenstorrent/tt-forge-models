# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tema Q-X2 4B Thinking GGUF model loader implementation for causal language modeling.

This model uses the Qwen3.5 (qwen35) architecture: a hybrid Mamba-2/transformer
model (Qwen3Next in transformers). Config is parsed from GGUF metadata.
"""
import importlib.metadata
from typing import Optional

import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
import transformers.utils.import_utils as _tx_import_utils
from transformers import AutoTokenizer, Qwen3NextConfig, Qwen3NextForCausalLM
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

_tx_import_utils.PACKAGE_DISTRIBUTION_MAPPING = (
    importlib.metadata.packages_distributions()
)


def _patch_qwen35_tokenizer():
    """Register qwen35 as a qwen3 alias so the GGUF tokenizer loads correctly."""
    if "qwen35" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen35",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3"],
            )
    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen35", GGUF_TO_FAST_CONVERTERS["qwen3"])
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3_5_text", GGUF_TO_FAST_CONVERTERS["qwen3"]
        )


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen35 tokenizer support."""
    _patch_qwen35_tokenizer()
    return _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )


_patch_qwen35_tokenizer()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

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
    """Available Tema Q-X2 4B Thinking GGUF model variants for causal language modeling."""

    TEMA_Q_X2_4B_THINKING_I1_GGUF = "Tema_Q_X2_4B_Thinking_i1_GGUF"


class ModelLoader(ForgeModel):
    """Tema Q-X2 4B Thinking GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.TEMA_Q_X2_4B_THINKING_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Tema_Q-X2-4B-Thinking-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TEMA_Q_X2_4B_THINKING_I1_GGUF

    GGUF_FILE = "Tema_Q-X2-4B-Thinking.i1-Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language model."

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
            model="Tema Q-X2 4B Thinking GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_gguf_path(self) -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=self.GGUF_FILE,
        )

    def _build_qwen3next_config(self, gguf_path: str) -> Qwen3NextConfig:
        """Parse GGUF metadata and return Qwen3NextConfig.

        Qwen3.5 (qwen35 in GGUF) is a Mamba-2/transformer hybrid identical to
        Qwen3Next in transformers, but with dense FFN (num_experts=0).
        """
        import gguf

        reader = gguf.GGUFReader(gguf_path)

        def get_int(key: str) -> int:
            f = reader.fields[key]
            return int(f.parts[f.data[0]])

        def get_float(key: str) -> float:
            f = reader.fields[key]
            return float(f.parts[f.data[0]])

        num_v_heads = get_int("qwen35.ssm.time_step_rank")
        ssm_inner_size = get_int("qwen35.ssm.inner_size")
        head_v_dim = ssm_inner_size // num_v_heads
        vocab_size = len(reader.fields["tokenizer.ggml.tokens"].data)

        config = Qwen3NextConfig(
            num_hidden_layers=get_int("qwen35.block_count"),
            hidden_size=get_int("qwen35.embedding_length"),
            intermediate_size=get_int("qwen35.feed_forward_length"),
            num_attention_heads=get_int("qwen35.attention.head_count"),
            num_key_value_heads=get_int("qwen35.attention.head_count_kv"),
            head_dim=get_int("qwen35.attention.key_length"),
            rms_norm_eps=get_float("qwen35.attention.layer_norm_rms_epsilon"),
            rope_theta=get_float("qwen35.rope.freq_base"),
            vocab_size=vocab_size,
            full_attention_interval=get_int("qwen35.full_attention_interval"),
            linear_num_value_heads=num_v_heads,
            linear_value_head_dim=head_v_dim,
            linear_num_key_heads=get_int("qwen35.ssm.group_count"),
            linear_key_head_dim=get_int("qwen35.ssm.state_size"),
            linear_conv_kernel_dim=get_int("qwen35.ssm.conv_kernel"),
            num_experts=0,
            tie_word_embeddings=False,
        )
        config.eos_token_id = get_int("tokenizer.ggml.eos_token_id")
        config.pad_token_id = get_int("tokenizer.ggml.padding_token_id")
        return config

    def _load_tokenizer(self, dtype_override=None):
        _tx_import_utils.PACKAGE_DISTRIBUTION_MAPPING = (
            importlib.metadata.packages_distributions()
        )
        _tx_import_utils.is_gguf_available.cache_clear()
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
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        gguf_path = self._get_gguf_path()
        config = self._build_qwen3next_config(gguf_path)

        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
            config.layer_types = config.layer_types[: self.num_layers]

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Qwen3NextForCausalLM(config)

        if dtype_override is not None:
            model = model.to(dtype_override)

        self.config = model.config
        self.model = model
        return model.eval()

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
            enable_thinking=True,
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
        _tx_import_utils.PACKAGE_DISTRIBUTION_MAPPING = (
            importlib.metadata.packages_distributions()
        )
        _tx_import_utils.is_gguf_available.cache_clear()
        gguf_path = self._get_gguf_path()
        self.config = self._build_qwen3next_config(gguf_path)
        return self.config
