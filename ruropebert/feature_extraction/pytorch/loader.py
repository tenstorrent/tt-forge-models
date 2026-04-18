# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ruRoPEBert model loader implementation for feature extraction.
"""
import glob
import os
import sys

import torch
from typing import Optional

from transformers import AutoTokenizer, AutoModel

from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


def _patch_transformers_compat():
    """Patch missing transformers APIs needed by ruRoPEBert custom code.

    The ruRoPEBert model uses custom code written for older transformers
    versions. This patches:
    - find_pruneable_heads_and_indices (removed in transformers 5.x)
    - PreTrainedModel.get_head_mask (removed in transformers 5.x)
    """
    from transformers import pytorch_utils

    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned):
            mask = torch.ones(n_heads, head_size)
            for head in heads:
                if head not in already_pruned:
                    mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        pytorch_utils.find_pruneable_heads_and_indices = (
            find_pruneable_heads_and_indices
        )

    from transformers import PreTrainedModel

    if not hasattr(PreTrainedModel, "get_head_mask"):

        def get_head_mask(
            self, head_mask, num_hidden_layers, is_attention_chunked=False
        ):
            if head_mask is not None:
                head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
                if is_attention_chunked:
                    head_mask = head_mask.unsqueeze(-1)
            else:
                head_mask = [None] * num_hidden_layers
            return head_mask

        PreTrainedModel.get_head_mask = get_head_mask


_patch_transformers_compat()


def _find_cached_modeling_file():
    """Find the cached modeling_rope_bert.py in HF modules cache."""
    hf_home = os.environ.get(
        "HF_HOME",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
    )
    return glob.glob(
        os.path.join(
            hf_home,
            "modules",
            "transformers_modules",
            "*",
            "*ruRoPEBert*",
            "*",
            "modeling_rope_bert.py",
        )
    )


def _patch_cached_model_source():
    """Patch the cached ruRoPEBert source to avoid XLA-incompatible gathers.

    The original apply_rotary_pos_emb uses cos[position_ids] which generates
    gather ops that fail in the TT-MLIR compiler. This replaces them with
    simple unsqueeze ops since cos/sin are already sliced to seq_len by
    BertRotaryEmbedding.forward.
    """
    matches = _find_cached_modeling_file()
    if not matches:
        return False

    modeling_path = matches[0]
    with open(modeling_path, "r") as f:
        source = f.read()

    old_rope = "cos = cos[position_ids].unsqueeze(unsqueeze_dim)"
    new_rope = "cos = cos.unsqueeze(0).unsqueeze(unsqueeze_dim)"

    old_rope_sin = "sin = sin[position_ids].unsqueeze(unsqueeze_dim)"
    new_rope_sin = "sin = sin.unsqueeze(0).unsqueeze(unsqueeze_dim)"

    old_import = (
        "from transformers.pytorch_utils import apply_chunking_to_forward,"
        " find_pruneable_heads_and_indices, prune_linear_layer"
    )
    new_import = """from transformers.pytorch_utils import apply_chunking_to_forward, prune_linear_layer

try:
    from transformers.pytorch_utils import find_pruneable_heads_and_indices
except ImportError:
    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned):
        mask = torch.ones(n_heads, head_size)
        for head in heads:
            if head not in already_pruned:
                mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index"""

    modified = False
    if old_rope in source:
        source = source.replace(old_rope, new_rope)
        source = source.replace(old_rope_sin, new_rope_sin)
        modified = True
    if old_import in source:
        source = source.replace(old_import, new_import)
        modified = True

    if modified:
        with open(modeling_path, "w") as f:
            f.write(source)
        pyc_dir = os.path.join(os.path.dirname(modeling_path), "__pycache__")
        if os.path.isdir(pyc_dir):
            for fname in os.listdir(pyc_dir):
                if "modeling_rope_bert" in fname:
                    os.remove(os.path.join(pyc_dir, fname))

    return modified


class ModelVariant(StrEnum):
    """Available ruRoPEBert model variants for feature extraction."""

    TOCHKA_AI_RUROPEBERT_CLASSIC_BASE_512 = "Tochka-AI/ruRoPEBert-classic-base-512"


class ModelLoader(ForgeModel):
    """ruRoPEBert model loader implementation for feature extraction."""

    _VARIANTS = {
        ModelVariant.TOCHKA_AI_RUROPEBERT_CLASSIC_BASE_512: LLMModelConfig(
            pretrained_model_name="Tochka-AI/ruRoPEBert-classic-base-512",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TOCHKA_AI_RUROPEBERT_CLASSIC_BASE_512

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="ruRoPEBert",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            model_name = self._variant_config.pretrained_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_name = self._variant_config.pretrained_model_name

        if not _find_cached_modeling_file():
            AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
            )

        patched = _patch_cached_model_source()
        if patched:
            for key in list(sys.modules.keys()):
                if "ruRoPEBert" in key or "rope_bert" in key:
                    del sys.modules[key]

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="eager",
            **model_kwargs,
        )
        model.eval()

        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            "Привет, как дела? Давай поговорим о чём-нибудь интересном.",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def output_postprocess(self, output, inputs=None):
        if inputs is None:
            inputs = self.load_inputs()

        attention_mask = inputs["attention_mask"]

        if isinstance(output, (tuple, list)):
            token_embeddings = output[0]
        elif hasattr(output, "last_hidden_state"):
            token_embeddings = output.last_hidden_state
        else:
            token_embeddings = output

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sentence_embeddings

    def decode_output(self, outputs, inputs=None):
        return self.output_postprocess(outputs, inputs=inputs)

    def unpack_forward_output(self, fwd_output):
        tensors = []

        if hasattr(fwd_output, "last_hidden_state"):
            tensors.append(fwd_output.last_hidden_state.flatten())
        if (
            hasattr(fwd_output, "pooler_output")
            and fwd_output.pooler_output is not None
        ):
            tensors.append(fwd_output.pooler_output.flatten())

        if tensors:
            return torch.cat(tensors, dim=0)
        return fwd_output
