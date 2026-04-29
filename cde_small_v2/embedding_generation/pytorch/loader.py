# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CDE-Small-V2 (Contextual Document Embeddings) model loader implementation for embedding generation.
"""
import torch
from transformers import AutoModel, AutoTokenizer
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
    """Available CDE-Small-V2 model variants for embedding generation."""

    CDE_SMALL_V2 = "cde-small-v2"


class ModelLoader(ForgeModel):
    """CDE-Small-V2 model loader implementation for embedding generation."""

    _VARIANTS = {
        ModelVariant.CDE_SMALL_V2: ModelConfig(
            pretrained_model_name="jxm/cde-small-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CDE_SMALL_V2

    sample_sentences = [
        "search_document: The cat sits on the mat",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="CDE-Small-V2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            "answerdotai/ModernBERT-base", **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # Two transformers 5.x breaking changes with this model:
        #
        # 1. CDE-Small-V2's __init__ calls from_pretrained for its dataset
        #    backbone (load_embedder_and_tokenizer). Transformers 5.x always
        #    initializes models inside torch.device("meta") via get_init_context;
        #    nested from_pretrained calls fail inside that context with a
        #    RuntimeError. Patch get_init_context to skip the meta device step.
        #
        # 2. The remote model code never calls self.post_init(), which is now
        #    required for transformers 5.x to set all_tied_weights_keys before
        #    _finalize_model_loading accesses it. Patch
        #    _adjust_tied_keys_with_tied_pointers to call post_init() on first
        #    use if the attribute is missing.
        from transformers.modeling_utils import PreTrainedModel

        _orig_get_init_context = PreTrainedModel.__dict__["get_init_context"]
        _orig_adjust = PreTrainedModel._adjust_tied_keys_with_tied_pointers

        @classmethod
        def _no_meta_get_init_context(cls, dtype, is_quantized, _is_ds_init_called):
            ctx = _orig_get_init_context.__func__(cls, dtype, is_quantized, _is_ds_init_called)
            return [c for c in ctx if not isinstance(c, torch.device)]

        def _patched_adjust(self, missing_keys_and_mismatched):
            if not hasattr(self, "all_tied_weights_keys"):
                self.post_init()
            return _orig_adjust(self, missing_keys_and_mismatched)

        PreTrainedModel.get_init_context = _no_meta_get_init_context
        PreTrainedModel._adjust_tied_keys_with_tied_pointers = _patched_adjust
        try:
            model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        finally:
            PreTrainedModel.get_init_context = _orig_get_init_context
            PreTrainedModel._adjust_tied_keys_with_tied_pointers = _orig_adjust

        # Without meta device, the nested from_pretrained for the dataset backbone
        # loads in float32, and the outer checkpoint only overwrites the params
        # that match its keys (leaving mismatched backbone params as float32).
        # Explicitly cast the whole model to ensure dtype consistency.
        if dtype_override is not None:
            model = model.to(dtype_override)

        # 3. Transformers 5.x _move_missing_keys_from_meta_to_device() unconditionally
        #    replaces all non-persistent buffers with torch.empty_like() (uninitialized
        #    garbage), even when the model was not loaded on meta device.  The outer
        #    ContextualDocumentEmbeddingTransformer.from_pretrained runs this on the full
        #    model tree, trashing the ModernBertRotaryEmbedding.{layer_type}_inv_freq
        #    buffers that the inner ModernBERT from_pretrained had just initialized.
        #    The CDE model's _initialize_weights is a no-op for ModernBertRotaryEmbedding,
        #    so the buffers remain garbage.  Re-initialize them here explicitly.
        from transformers.models.modernbert.modeling_modernbert import ModernBertRotaryEmbedding
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _ROPE_INIT_FNS

        for _module in model.modules():
            if isinstance(_module, ModernBertRotaryEmbedding):
                for _layer_type in _module.layer_types:
                    _rope_init_fn = _module.compute_default_rope_parameters
                    if _module.rope_type[_layer_type] != "default":
                        _rope_init_fn = _ROPE_INIT_FNS[_module.rope_type[_layer_type]]
                    _inv_freq, _ = _rope_init_fn(_module.config, layer_type=_layer_type)
                    _inv_freq = _inv_freq.float()
                    _module.register_buffer(f"{_layer_type}_inv_freq", _inv_freq, persistent=False)
                    _module.register_buffer(f"{_layer_type}_original_inv_freq", _inv_freq.clone(), persistent=False)

        # 4. mean_pool uses (int64_tensor + 1e-20_python_float) as denominator,
        #    which promotes bfloat16 numerators to float32 via PyTorch scalar
        #    type promotion, causing dtype mismatches in downstream linear layers.
        #    Patch mean_pool in the remote module to preserve the input dtype.
        import sys

        _cde_mod = next(
            (sys.modules[k] for k in sys.modules if "cde_hyphen_small_hyphen_v2" in k),
            None,
        )
        if _cde_mod is not None and hasattr(_cde_mod, "mean_pool"):

            def _mean_pool_typed(hidden_states, attention_mask):
                B, _S, D = hidden_states.shape
                unmasked_outputs = hidden_states * attention_mask[..., None]
                denom = (
                    attention_mask.sum(dim=1)[:, None].to(hidden_states.dtype) + 1e-20
                )
                pooled = unmasked_outputs.sum(dim=1) / denom
                assert pooled.shape == (B, D)
                return pooled

            _cde_mod.mean_pool = _mean_pool_typed

        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_sentences,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        # ContextualDocumentEmbeddingTransformer.forward() requires dataset context
        # tensors as positional arguments. Use the same sample sentences as the corpus.
        dataset_inputs = self.tokenizer(
            self.sample_sentences,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs["dataset_input_ids"] = dataset_inputs["input_ids"]
        inputs["dataset_attention_mask"] = dataset_inputs["attention_mask"]

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

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
