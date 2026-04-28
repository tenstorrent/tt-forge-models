# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Florence-2-Flux-Large model loader implementation (PyTorch).
"""

import functools

import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
)
from transformers.tokenization_utils_tokenizers import PreTrainedTokenizerFast
from tokenizers import AddedToken
from typing import Optional
from PIL import Image

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...tools.utils import get_file


def _tokenizer_compat_load(pretrained_model_name, **kwargs):
    """Load a tokenizer with a fix for transformers 5.x dict-in-special-tokens bug.

    transformers 5.x does not convert dict entries in additional_special_tokens
    (from special_tokens_map.json) to AddedToken objects before handing them to
    the Rust tokenizer backend, causing TypeError in add_tokens().  Patch
    _add_tokens temporarily to do the conversion.
    """
    _orig_add_tokens = PreTrainedTokenizerFast._add_tokens

    @functools.wraps(_orig_add_tokens)
    def _patched_add_tokens(self, new_tokens, special_tokens=False):
        fixed = []
        for t in new_tokens:
            if isinstance(t, dict):
                t = AddedToken(
                    t["content"],
                    lstrip=t.get("lstrip", False),
                    rstrip=t.get("rstrip", False),
                    single_word=t.get("single_word", False),
                    normalized=t.get("normalized", False),
                    special=special_tokens,
                )
            fixed.append(t)
        return _orig_add_tokens(self, fixed, special_tokens=special_tokens)

    PreTrainedTokenizerFast._add_tokens = _patched_add_tokens
    try:
        return AutoTokenizer.from_pretrained(pretrained_model_name, **kwargs)
    finally:
        PreTrainedTokenizerFast._add_tokens = _orig_add_tokens


def _florence2_compat_load(pretrained_model_name, **kwargs):
    """Load Florence-2 with workarounds for transformers 5.x + remote code compat.

    transformers 5.x moved generation params (including forced_bos_token_id) out
    of PretrainedConfig into GenerationConfig, so the custom remote code raises
    AttributeError when it checks self.forced_bos_token_id after super().__init__().
    Patch the init temporarily to restore the attribute.

    The DaViT vision encoder calls torch.linspace().item() during model construction.
    On meta device this raises RuntimeError; patch linspace to force CPU.
    """
    _orig_pc_init = PretrainedConfig.__init__

    @functools.wraps(_orig_pc_init)
    def _patched_pc_init(self, **kw):
        _orig_pc_init(self, **kw)
        if not hasattr(self, "forced_bos_token_id"):
            self.forced_bos_token_id = None

    _orig_linspace = torch.linspace

    def _cpu_linspace(*args, **kw):
        kw["device"] = "cpu"
        return _orig_linspace(*args, **kw)

    PretrainedConfig.__init__ = _patched_pc_init
    torch.linspace = _cpu_linspace
    try:
        return AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **kwargs,
        )
    finally:
        PretrainedConfig.__init__ = _orig_pc_init
        torch.linspace = _orig_linspace


class ModelVariant(StrEnum):
    """Available Florence-2-Flux-Large model variants."""

    LARGE = "Large"


class ModelLoader(ForgeModel):
    """Florence-2-Flux-Large image captioning model loader implementation."""

    _VARIANTS = {
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="gokaygokay/Florence-2-Flux-Large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.image_processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Florence-2-Flux-Large",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_CAPT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        name = self._variant_config.pretrained_model_name
        self.tokenizer = _tokenizer_compat_load(name, trust_remote_code=True)
        self.image_processor = AutoImageProcessor.from_pretrained(
            name, trust_remote_code=True, use_fast=False
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = _florence2_compat_load(
            pretrained_model_name,
            attn_implementation="eager",
            **model_kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_processor()

        image_path = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_path)).convert("RGB")

        prompt = "<DESCRIPTION>"
        text_inputs = self.tokenizer(prompt, return_tensors="pt")
        pixel_values = self.image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ]
        inputs = {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "pixel_values": pixel_values,
        }

        # Florence-2 is a seq2seq model that requires decoder_input_ids
        decoder_start_token_id = self.tokenizer.bos_token_id or 2
        inputs["decoder_input_ids"] = torch.full(
            (1, 1), decoder_start_token_id, dtype=torch.long
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        return fwd_output

    def decode_output(self, **kwargs):
        outputs = kwargs.get("outputs")
        if outputs is None:
            return None

        if self.tokenizer is None:
            self._load_processor()

        if isinstance(outputs, torch.Tensor):
            if outputs.dtype in (torch.long, torch.int32, torch.int64):
                token_ids = outputs
            else:
                token_ids = outputs.argmax(dim=-1)
        else:
            token_ids = outputs

        return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
