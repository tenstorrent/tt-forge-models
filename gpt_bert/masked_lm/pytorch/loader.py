# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-BERT model loader implementation for masked language modeling.
"""

import sys
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from transformers import PreTrainedModel
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


class ModelVariant(StrEnum):
    """Available GPT-BERT model variants for masked language modeling."""

    BABYLM_BASELINE_100M_MASKED_FOCUS = "BabyLM_Baseline_100M_Masked_Focus"


class ModelLoader(ForgeModel):
    """GPT-BERT model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.BABYLM_BASELINE_100M_MASKED_FOCUS: LLMModelConfig(
            pretrained_model_name="BabyLM-community/babylm-baseline-100m-gpt-bert-masked-focus",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BABYLM_BASELINE_100M_MASKED_FOCUS

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        pretrained_model_name = self._variant_config.pretrained_model_name
        self.model_name = pretrained_model_name
        self.sample_text = "The capital of France is <mask>."
        self.max_length = 128
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses default.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="GPT-BERT",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load GPT-BERT model for masked language modeling from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The GPT-BERT model instance.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        # Pre-load config to patch torch.dtype JSON serialization bug in the
        # custom GPT-BERT config (its to_dict doesn't convert torch.dtype to
        # string like the standard PretrainedConfig does).
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        original_to_dict = config.__class__.to_dict

        def _patched_to_dict(self_inner):
            output = original_to_dict(self_inner)
            if "torch_dtype" in output and isinstance(
                output["torch_dtype"], torch.dtype
            ):
                output["torch_dtype"] = str(output["torch_dtype"])
            return output

        config.__class__.to_dict = _patched_to_dict

        model_kwargs = {"config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # GPTBERTForMaskedLM.__init__ does not call self.post_init(), so
        # all_tied_weights_keys (required by transformers 5.x _finalize_model_loading)
        # is never initialised. Patch _finalize_model_loading to call post_init()
        # when the attribute is absent; restore immediately after loading.
        _orig_finalize = PreTrainedModel.__dict__["_finalize_model_loading"].__func__

        @staticmethod
        def _patched_finalize(model, load_config, loading_info):
            if not hasattr(model, "all_tied_weights_keys"):
                # Initialize only the attribute needed by _finalize_model_loading;
                # calling post_init() would re-run _init_weights which crashes on
                # LayerNorm modules with elementwise_affine=False (bias=None).
                model.all_tied_weights_keys = (
                    model.get_expanded_tied_weights_keys(all_submodels=False)
                )
            # GPTBERTPreTrainedModel._init_weights does not guard bias=None, so
            # _finalize_model_loading->_initialize_missing_keys->initialize_weights
            # crashes on LayerNorm(elementwise_affine=False). Mark them as already
            # initialized so _initialize_weights skips them.
            for m in model.modules():
                if isinstance(m, nn.LayerNorm) and not m.elementwise_affine:
                    m._is_hf_initialized = True
            return _orig_finalize(model, load_config, loading_info)

        PreTrainedModel._finalize_model_loading = _patched_finalize
        try:
            model = AutoModelForMaskedLM.from_pretrained(
                self.model_name, trust_remote_code=True, **model_kwargs
            )
        finally:
            PreTrainedModel._finalize_model_loading = staticmethod(_orig_finalize)

        self._patch_inplace_set_slice_dtype(model)
        self._reinit_position_indices(model)

        model.eval()
        return model

    def _patch_inplace_set_slice_dtype(self, model):
        # The cached modeling_gpt_bert.py uses `torch.Tensor().to(device)` to
        # create `ret`, which is float32 regardless of `full_tensor.dtype`.
        # This causes `ret.set_(bfloat16_slice)` to fail with a dtype mismatch
        # when the model is loaded with torch_dtype=bfloat16.
        # Fix: replace the forward with one that creates `ret` from the correct
        # dtype via torch.empty.
        for key, mod in sys.modules.items():
            if "babylm" in key and "modeling_gpt_bert" in key and hasattr(
                mod, "InPlaceSetSlice"
            ):

                @staticmethod
                def _fixed_forward(ctx, full_tensor, last_slice, x_idx, x_val):
                    full_tensor[x_idx] = x_val
                    ctx.x_idx = x_idx
                    # Original code uses torch.Tensor().to(device) which creates a
                    # float32 empty tensor; ret.set_(bfloat16_slice) then fails.
                    # Avoid set_ entirely: return the slice directly.  x_idx is
                    # constant at every call site (the for-loop is unrolled by
                    # torch.compile), so the returned slice has a static shape.
                    return full_tensor[:x_idx + 1]

                mod.InPlaceSetSlice.forward = _fixed_forward
                break

    def _reinit_position_indices(self, model):
        # Transformers 5.x constructs models on meta device; non-persistent
        # buffers (like Attention.position_indices) are NOT in the checkpoint,
        # so they are materialised from meta tensors as uninitialised memory.
        # Re-run the exact computation from Attention.__init__ to fix them.
        for attn in model.model.attention_layers:
            cfg = attn.config
            pi = (
                torch.arange(cfg.max_position_embeddings, dtype=torch.long).unsqueeze(1)
                - torch.arange(cfg.max_position_embeddings, dtype=torch.long).unsqueeze(0)
            )
            pi = attn.make_log_bucket_position(
                pi, cfg.position_bucket_size, cfg.max_position_embeddings
            )
            pi = cfg.position_bucket_size - 1 + pi
            attn.register_buffer("position_indices", pi, persistent=False)

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for GPT-BERT masked language modeling.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for masked language modeling."""
        inputs = self.load_inputs()
        logits = co_out[0]
        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_token = self.tokenizer.decode(predicted_token_id)
        print("The predicted token for the <mask> is:", predicted_token)

    def load_config(self):
        """Load and return the configuration for the GPT-BERT model variant.

        Returns:
            The configuration object for the GPT-BERT model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )

        return self.config
