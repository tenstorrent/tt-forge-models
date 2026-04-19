# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kai-3B-Instruct GGUF model loader implementation for causal language modeling.
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


def _patch_transformers_smollm3_gguf():
    """Monkey-patch transformers to add smollm3 GGUF architecture support.

    The GGUF file declares architecture as 'smollm3' which transformers
    does not natively support. The backbone is identical to llama,
    so we reuse its config mapping and remap model_type to 'llama'.
    Also patches GGUFTokenizerSkeleton to default bos_token_id to None,
    working around a transformers bug where the llama tokenizer converter
    accesses bos_token_id unconditionally.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "smollm3" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("smollm3")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["smollm3"] = GGUF_TO_TRANSFORMERS_MAPPING[
        "config"
    ]["llama"]

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFLlamaConverter,
        GGUFTokenizerSkeleton,
    )
    import transformers.integrations.ggml as ggml_mod

    _orig_skeleton_init = GGUFTokenizerSkeleton.__init__

    def _patched_skeleton_init(self, dict_):
        _orig_skeleton_init(self, dict_)
        if not hasattr(self, "bos_token_id"):
            self.bos_token_id = None

    GGUFTokenizerSkeleton.__init__ = _patched_skeleton_init

    from tokenizers import Tokenizer, AddedToken
    from tokenizers.models import BPE
    import numpy as np

    def _patched_tokenizer(self, proto):
        vocab_scores = self.vocab(self.proto)
        merges = self.merges(self.proto)
        bpe_vocab = {word: i for i, (word, _score) in enumerate(vocab_scores)}

        unk_token = (
            proto.tokens[proto.unk_token_id] if proto.unk_token_id is not None else None
        )
        bos_token = (
            proto.tokens[proto.bos_token_id]
            if getattr(proto, "bos_token_id", None) is not None
            else None
        )
        eos_token = (
            proto.tokens[proto.eos_token_id]
            if getattr(proto, "eos_token_id", None) is not None
            else None
        )

        tokenizer = Tokenizer(
            BPE(
                bpe_vocab,
                merges,
                unk_token=unk_token,
                fuse_unk=True,
                byte_fallback=True,
            )
        )

        special_tokens = []

        if not hasattr(self.proto, "token_type"):
            if unk_token is not None:
                special_tokens.append(
                    AddedToken(unk_token, normalized=False, special=True)
                )
            if bos_token is not None:
                special_tokens.append(
                    AddedToken(bos_token, normalized=False, special=True)
                )
            if eos_token is not None:
                special_tokens.append(
                    AddedToken(eos_token, normalized=False, special=True)
                )
        else:
            special_tokens_idx = np.where(np.array(self.proto.token_type) == 3)[0]
            for idx in special_tokens_idx:
                special_tokens.append(
                    AddedToken(self.proto.tokens[idx], normalized=False, special=True)
                )

        if len(special_tokens) != 0:
            tokenizer.add_special_tokens(special_tokens)

        if len(self.proto.added_tokens) != 0:
            tokenizer.add_tokens(
                [
                    AddedToken(added_token, normalized=False, special=False)
                    for added_token in self.proto.added_tokens
                ]
            )

        self.additional_kwargs["unk_token"] = unk_token
        self.additional_kwargs["bos_token"] = bos_token
        self.additional_kwargs["eos_token"] = eos_token

        if self.is_llama_3_tokenizer:
            self.additional_kwargs["add_prefix_space"] = None
            self.additional_kwargs["clean_up_tokenization_spaces"] = True
            self.additional_kwargs["legacy"] = False
            self.original_tokenizer.legacy = False

        return tokenizer

    GGUFLlamaConverter.tokenizer = _patched_tokenizer

    if "smollm3" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["smollm3"] = GGUFLlamaConverter

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "smollm3":
            config["model_type"] = "llama"
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_transformers_smollm3_gguf()


class ModelVariant(StrEnum):
    """Available Kai-3B-Instruct GGUF model variants for causal language modeling."""

    KAI_3B_INSTRUCT_I1_Q4_K_M = "I1_Q4_K_M"


class ModelLoader(ForgeModel):
    """Kai-3B-Instruct GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.KAI_3B_INSTRUCT_I1_Q4_K_M: LLMModelConfig(
            pretrained_model_name="SimplySara/Kai-3B-Instruct-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KAI_3B_INSTRUCT_I1_Q4_K_M

    GGUF_FILE = "Kai-3B-Instruct-i1-Q4_K_M.gguf"

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
            model="Kai-3B-Instruct GGUF",
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
