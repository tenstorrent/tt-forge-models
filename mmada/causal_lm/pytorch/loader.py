# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MMaDA model loader implementation for causal language modeling.
"""

from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _patch_transformers_mmada():
    """Load LLaDA model code from GSAI-ML/LLaDA-8B-Instruct (which ships the
    configuration_llada.py and modeling_llada.py files that MMaDA-8B-MixCoT
    omits from its HuggingFace repo) and apply two transformers-5.x shims:

    1. LLaDAModelLM.__init__ does not call self.post_init(), so the instance
       attribute ``all_tied_weights_keys`` required by transformers 5.x
       _adjust_tied_keys_with_tied_pointers() is never set.  We add it at the
       end of __init__.

    2. transformers 5.x calls tie_weights(missing_keys=…, recompute_mapping=…)
       but LLaDAModelLM.tie_weights() accepts no arguments.  We wrap it to
       absorb the extra kwargs.

    Returns (LLaDAConfig, LLaDAModelLM).
    """
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    LLADA_BASE = "GSAI-ML/LLaDA-8B-Instruct"

    # Fetch and register both the config class and the model class from the
    # base repo that ships the required source files.
    llada_cfg_instance = AutoConfig.from_pretrained(LLADA_BASE, trust_remote_code=True)
    LLaDAConfig = type(llada_cfg_instance)

    LLaDAModelLM = get_class_from_dynamic_module(
        "modeling_llada.LLaDAModelLM", LLADA_BASE
    )

    # Shim 1: set all_tied_weights_keys after __init__ (transformers 5.x API).
    _orig_init = LLaDAModelLM.__init__

    def _patched_init(self, config, model=None, init_params=False):
        _orig_init(self, config, model=model, init_params=init_params)
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(
                all_submodels=False
            )

    LLaDAModelLM.__init__ = _patched_init

    # Shim 2: absorb extra kwargs added to tie_weights in transformers 5.x.
    _orig_tie = LLaDAModelLM.tie_weights

    def _patched_tie(self, **kwargs):
        _orig_tie(self)

    LLaDAModelLM.tie_weights = _patched_tie

    return LLaDAConfig, LLaDAModelLM


class ModelVariant(StrEnum):
    """Available MMaDA model variants."""

    MMADA_8B_MIXCOT = "8B_MixCoT"


class ModelLoader(ForgeModel):
    """MMaDA model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MMADA_8B_MIXCOT: ModelConfig(
            pretrained_model_name="Gen-Verse/MMaDA-8B-MixCoT",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MMADA_8B_MIXCOT

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MMaDA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        LLaDAConfig, LLaDAModelLM = _patch_transformers_mmada()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        # Load the MMaDA config via the registered LLaDAConfig class (the
        # MMaDA repo ships no configuration_llada.py so AutoConfig fails).
        config = LLaDAConfig.from_pretrained(pretrained_model_name)

        # transformers 5.x PretrainedConfig.__init__ drops unknown kwargs such
        # as use_cache; set it explicitly so forward() can read it.
        if not hasattr(config, "use_cache"):
            config.use_cache = False

        if self.num_layers is not None:
            config.n_layers = self.num_layers

        model = LLaDAModelLM.from_pretrained(
            pretrained_model_name, config=config, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        test_input = "What is the capital of France?"

        inputs = self.tokenizer(test_input, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])

    def load_config(self):
        LLaDAConfig, _ = _patch_transformers_mmada()
        self.config = LLaDAConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        if not hasattr(self.config, "use_cache"):
            self.config.use_cache = False
        return self.config
