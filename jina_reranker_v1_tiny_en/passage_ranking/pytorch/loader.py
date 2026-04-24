# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Jina Reranker v1 tiny English model loader implementation for passage ranking.
"""
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers.dynamic_module_utils import get_class_in_module
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Jina Reranker v1 tiny English model variants for passage ranking."""

    TINY_EN = "tiny-en"


class ModelLoader(ForgeModel):
    """Jina Reranker v1 tiny English model loader implementation for passage ranking."""

    _VARIANTS = {
        ModelVariant.TINY_EN: ModelConfig(
            pretrained_model_name="jinaai/jina-reranker-v1-tiny-en",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_EN

    sample_pairs = [
        (
            "Organic skincare products for sensitive skin",
            "Natural organic skincare range for sensitive skin",
        ),
        (
            "Organic skincare products for sensitive skin",
            "Eco-friendly kitchenware for modern homes",
        ),
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="JinaRerankerV1TinyEn",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        # transformers>=5 no longer provides default values for unknown config
        # attributes; the custom JinaBert model reads several standard BERT
        # defaults that JinaBertConfig does not declare, so we set them here.
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True, num_labels=1
        )
        _bert_defaults = {
            "is_decoder": False,
            "add_cross_attention": False,
            "chunk_size_feed_forward": 0,
            "output_attentions": False,
            "output_hidden_states": False,
        }
        for attr, default in _bert_defaults.items():
            if not hasattr(config, attr):
                object.__setattr__(config, attr, default)

        # transformers>=5 get_init_context() unconditionally wraps model __init__
        # in torch.device("meta").  JinaBert has several custom operations that
        # run during __init__ and require real CPU tensors:
        #   - JinaBertEncoder.rebuild_alibi_tensor uses torch.arange (meta
        #     context makes it return a meta tensor) and torch.Tensor(list) (CPU)
        #     causing a device mismatch on multiply.
        #   - JinaBertEmbeddings registers position_ids / token_type_ids with
        #     persistent=False, so they are never restored from checkpoint and
        #     remain as meta tensors that break inference.
        #
        # Fix: override get_init_context on JinaBertForSequenceClassification
        # to strip the torch.device("meta") context.  This makes all __init__
        # tensors materialise on CPU instead, at the cost of slightly higher peak
        # memory during loading (acceptable for this tiny model).
        #
        # get_class_in_module() re-executes the module file if
        # __transformers_module_hash__ is not set, overwriting any patch applied
        # via importlib.import_module.  We call get_class_in_module ourselves
        # first (which sets __transformers_module_hash__), then apply the patch.
        # Subsequent calls by from_pretrained find the hash cached and skip
        # re-execution, so the patch survives into model construction.
        _config_mod_name = type(config).__module__
        if _config_mod_name and "configuration_bert" in _config_mod_name:
            _modeling_rel_path = (
                _config_mod_name.replace("configuration_bert", "modeling_bert").replace(
                    ".", "/"
                )
                + ".py"
            )
            _jina_cls = get_class_in_module(
                "JinaBertForSequenceClassification", _modeling_rel_path
            )
            if isinstance(_jina_cls, type) and not getattr(
                _jina_cls, "_no_meta_init_patched", False
            ):

                @classmethod  # type: ignore[misc]
                def _patched_get_init_ctx(cls, dtype, is_quantized, _is_ds_init_called):
                    # JinaBert uses torch.arange/torch.zeros in __init__ to create
                    # CPU integer buffers (position_ids, token_type_ids, alibi).
                    # transformers>=5 init contexts include:
                    #   1. torch.device("meta") — makes tensors meta, breaking alibi
                    #   2. local_torch_dtype(dtype) — sets default dtype to bfloat16,
                    #      which corrupts integer buffer initialization under the TT
                    #      TorchFunctionMode override.
                    # Return empty list so __init__ runs with plain CPU defaults.
                    return []

                _jina_cls.get_init_context = _patched_get_init_ctx
                _jina_cls._no_meta_init_patched = True

        model_kwargs = {
            "config": config,
            "trust_remote_code": True,
        }
        model_kwargs |= kwargs

        # JinaBertEncoder computes the ALiBi tensor in __init__, which requires
        # real CPU tensors.  Passing dtype to from_pretrained triggers meta-device
        # lazy loading in transformers>=5 that conflicts with the TT torch
        # overrides, so we load in float32 and cast afterwards.
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # transformers>=5 replaces all non-persistent buffers with torch.empty_like
        # after loading the checkpoint (modeling_utils.py _finalize_model_loading).
        # This leaves position_ids, token_type_ids, and alibi with garbage values.
        # Re-register them with the correct initialisation.
        max_pos = model.config.max_position_embeddings
        emb = model.bert.embeddings
        emb.register_buffer(
            "position_ids",
            torch.arange(max_pos).expand((1, -1)),
            persistent=False,
        )
        emb.register_buffer(
            "token_type_ids",
            torch.zeros((1, max_pos), dtype=torch.long),
            persistent=False,
        )
        enc = model.bert.encoder
        enc.register_buffer(
            "alibi",
            enc.rebuild_alibi_tensor(size=max_pos),
            persistent=False,
        )

        if dtype_override is not None:
            model = model.to(dtype_override)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        queries = [pair[0] for pair in self.sample_pairs]
        passages = [pair[1] for pair in self.sample_pairs]

        inputs = self.tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=1024,
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.float32:
                        inputs[key] = value.to(dtype_override)

        return inputs
