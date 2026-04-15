# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-BERT model loader implementation for masked language modeling.
"""

import sys

import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_utils import PreTrainedModel
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

    @staticmethod
    def _patch_custom_classes():
        """Patch custom GPT-BERT classes for transformers 5.x compatibility.

        The remote model code has three issues:
        1. Config overrides to_dict()/to_json_string() without handling
           torch.dtype serialization (crashes on f-string config logging).
        2. Model classes don't call post_init(), so all_tied_weights_keys
           is never set.
        3. _init_weights assumes LayerNorm always has bias/weight, but some
           are created with elementwise_affine=False.
        """
        for module_name, module in sys.modules.items():
            if "transformers_modules" not in module_name:
                continue
            if "gpt_bert" not in module_name:
                continue
            for attr_name in dir(module):
                cls = getattr(module, attr_name)
                if not isinstance(cls, type):
                    continue
                # Fix config: use parent to_dict/to_json_string which handle
                # torch.dtype serialization
                if issubclass(cls, PretrainedConfig) and cls is not PretrainedConfig:
                    cls.to_dict = PretrainedConfig.to_dict
                    cls.to_json_string = PretrainedConfig.to_json_string
                    cls.__repr__ = PretrainedConfig.__repr__
                if not issubclass(cls, PreTrainedModel) or cls is PreTrainedModel:
                    continue
                # Fix _init_weights: guard against LayerNorm without bias/weight
                if hasattr(cls, "_init_weights"):
                    orig_init_weights = cls._init_weights

                    def _safe_init_weights(self, module, _orig=orig_init_weights):
                        if isinstance(module, nn.LayerNorm) and module.bias is None:
                            return
                        return _orig(self, module)

                    cls._init_weights = _safe_init_weights
                # Fix post_init: set all_tied_weights_keys if missing
                original_init = cls.__init__

                def _patched_init(self, *args, _orig=original_init, **kwargs):
                    _orig(self, *args, **kwargs)
                    if not hasattr(self, "all_tied_weights_keys"):
                        self.all_tied_weights_keys = {}

                cls.__init__ = _patched_init

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

        # Load config to trigger download of remote code, then pre-load the
        # model class so we can patch it for transformers 5.x compatibility
        # before from_pretrained runs.
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        model_class = get_class_from_dynamic_module(
            config.auto_map["AutoModelForMaskedLM"],
            self.model_name,
            trust_remote_code=True,
        )
        self._patch_custom_classes()

        model = model_class.from_pretrained(
            self.model_name, config=config, trust_remote_code=True, **kwargs
        )
        if dtype_override is not None:
            model = model.to(dtype=dtype_override)
        model.eval()
        return model

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
