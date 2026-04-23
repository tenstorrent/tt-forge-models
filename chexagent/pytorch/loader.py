# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CheXagent model loader implementation for chest X-ray vision-language tasks.
"""

import os
import transformers
import transformers.utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional

# CheXagent's custom tokenizer imports is_tf_available which was removed in
# transformers 5.x; inject a stub before the dynamic module is loaded.
if not hasattr(transformers.utils, "is_tf_available"):
    transformers.utils.is_tf_available = lambda: False

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


class ModelVariant(StrEnum):
    """Available CheXagent model variants."""

    CHEXAGENT_2_3B = "chexagent_2_3b"


class ModelLoader(ForgeModel):
    """CheXagent model loader implementation for chest X-ray vision-language tasks."""

    _VARIANTS = {
        ModelVariant.CHEXAGENT_2_3B: ModelConfig(
            pretrained_model_name="StanfordAIMI/CheXagent-2-3b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CHEXAGENT_2_3B

    sample_image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
    )
    sample_text = "Describe the findings in this chest X-ray."
    sample_system_prompt = "You are a helpful assistant."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="CheXagent",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        # transformers 5.x removed spaces_between_special_tokens as a positional
        # arg from PreTrainedTokenizerBase._decode; the custom tokenizer passes it
        # positionally which now fails. Patch the class to call super() correctly.
        import sys as _sys
        from transformers import PreTrainedTokenizer

        tok_cls = type(self.tokenizer)
        tok_module = _sys.modules.get(tok_cls.__module__, None)
        _replace_closed_tag = (
            getattr(tok_module, "_replace_closed_tag", None) if tok_module else None
        )
        if _replace_closed_tag is not None and hasattr(tok_cls, "_decode"):

            def _fixed_decode(
                self_inner,
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=None,
                spaces_between_special_tokens=True,
                **kwargs,
            ):
                def _decode_imgurl(img_token_ids):
                    assert (
                        img_token_ids[0] == self_inner.img_start_id
                        and img_token_ids[-1] == self_inner.img_end_id
                    )
                    img_token_ids = img_token_ids[1:-1]
                    img_token_ids = img_token_ids[
                        : img_token_ids.index(self_inner.img_pad_id)
                    ]
                    return (
                        [self_inner.img_start_id]
                        + img_token_ids
                        + [self_inner.img_end_id]
                    )

                token_ids = _replace_closed_tag(
                    token_ids,
                    self_inner.img_start_id,
                    self_inner.img_end_id,
                    _decode_imgurl,
                )
                return PreTrainedTokenizer._decode(
                    self_inner,
                    token_ids,
                    skip_special_tokens,
                    clean_up_tokenization_spaces,
                    **kwargs,
                )

            tok_cls._decode = _fixed_decode
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.tokenizer is None:
            self._load_tokenizer()

        # The model's modeling_visual.py asserts transformers.__version__ == "4.40.0".
        # The APIs it relies on remain available in 5.x, so we spoof the version
        # for the duration of the dynamic module import.
        import transformers.modeling_utils as _tmu

        _orig_ctx_fn = _tmu.get_torch_context_manager_or_global_device

        def _no_meta_ctx():
            # The outer from_pretrained uses a meta-device context for lazy init.
            # CheXagent's __init__ then calls AutoModel.from_pretrained to load its
            # visual backbone, which fails if a meta device is detected. Return None
            # so the inner load proceeds on CPU instead of raising.
            result = _orig_ctx_fn()
            import torch

            return None if result == torch.device("meta") else result

        _tmu.get_torch_context_manager_or_global_device = _no_meta_ctx

        _orig_version = transformers.__version__
        transformers.__version__ = "4.40.0"
        try:
            # CheXagentConfig doesn't set pad_token_id; in transformers 5.x accessing
            # it raises AttributeError, so we load the config and set it explicitly.
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
                config.pad_token_id = config.eos_token_id
            # The model's _init_rope uses rope_scaling["type"] (the 4.x key name) and
            # rope_scaling["factor"]. In transformers 5.x the config stores rope params
            # inline with rope_type="default" (no scaling). Clear rope_scaling so the
            # model falls through to the default no-scaling RoPE path.
            if (
                config.rope_scaling
                and config.rope_scaling.get("rope_type") == "default"
            ):
                config.rope_scaling = None
            # low_cpu_mem_usage=False disables the meta-device context so that the
            # model's __init__ can call from_pretrained for its visual sub-module.
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name,
                config=config,
                low_cpu_mem_usage=False,
                **model_kwargs,
            )
        finally:
            transformers.__version__ = _orig_version
            _tmu.get_torch_context_manager_or_global_device = _orig_ctx_fn

        # The visual backbone is loaded inside the outer meta-device context.
        # Non-persistent buffers (e.g. SigLIP position_ids) are moved from meta
        # to CPU via torch.empty_like, leaving garbage values. Reinitialize them
        # with the correct sequential position values regardless of device.
        import torch as _torch
        import types as _types

        for _mod in model.modules():
            if hasattr(_mod, "position_ids") and isinstance(
                _mod.position_ids, _torch.Tensor
            ):
                n = _mod.position_ids.shape[-1]
                _mod.position_ids = _torch.arange(n).expand((1, -1))

        # In transformers 5.x SiglipVisionTransformer.forward never populates
        # hidden_states in its return value (only last_hidden_state). The custom
        # CLIPModel.forward in modeling_visual.py calls hidden_states[-1], which
        # equals last_hidden_state. Patch CLIPModel.forward to fall back.
        #
        # Additionally, CLIPModel's pos_embed and proj are created as float32 in
        # __init__ while the surrounding LLM and attn_pool weights are bfloat16.
        # Cast the entire CLIPModel (including inner SigLIP) to one consistent
        # dtype so forward_resampler doesn't hit dtype-mismatch in attn_pool.
        for _mod in model.modules():
            if type(_mod).__name__ == "CLIPModel":
                target_dtype = next(_mod.attn_pool.parameters()).dtype
                _mod.to(target_dtype)

                def _patched_clip_forward(_self, x):
                    result = _self.model(x, output_hidden_states=True)
                    feat = (
                        result.hidden_states[-1]
                        if result.hidden_states is not None
                        else result.last_hidden_state
                    )
                    return _self.forward_resampler(feat)

                _mod.forward = _types.MethodType(_patched_clip_forward, _mod)

        model.eval()
        return model

    # Persistent local image used by load_inputs; the path is encoded into the
    # input tokens and must still exist when the model's forward() runs.
    _SAMPLE_IMAGE_PATH = os.path.join(
        os.path.dirname(__file__), "sample_chest_xray.png"
    )

    @classmethod
    def _ensure_sample_image(cls) -> str:
        """Create a synthetic 384×384 grayscale PNG if it doesn't exist yet."""
        if not os.path.exists(cls._SAMPLE_IMAGE_PATH):
            from PIL import Image as _PILImage
            import numpy as _np

            img = _PILImage.fromarray(
                _np.zeros((384, 384), dtype=_np.uint8), mode="L"
            ).convert("RGB")
            img.save(cls._SAMPLE_IMAGE_PATH)
        return cls._SAMPLE_IMAGE_PATH

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        # Use a local synthetic image to avoid network access during inference.
        image_path = self._ensure_sample_image()
        query = self.tokenizer.from_list_format(
            [
                {"image": image_path},
                {"text": self.sample_text},
            ]
        )

        conv = [
            {"from": "system", "value": self.sample_system_prompt},
            {"from": "human", "value": query},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        )
        # transformers 5.x returns a BatchEncoding; 4.x returned the tensor directly.
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids.input_ids

        return {"input_ids": input_ids}
