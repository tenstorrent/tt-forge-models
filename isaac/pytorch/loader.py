# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Isaac model loader implementation for multimodal visual question answering
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
from PIL import Image
from typing import Optional

# Transformers 5.x removed the bare "eager" key from ALL_ATTENTION_FUNCTIONS (only
# "paged|eager" remains). modular_isaac.py captures _ORIGINAL_ATTENTION_FUNCTIONS["eager"]
# at import time; if absent, _isaac_eager_forward raises ValueError even for
# IsaacVisionAttention. Alias the real "paged|eager" implementation (which handles GQA
# via repeat_kv) so non-vision text attention works correctly too.
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as _ALL_ATTN_FNS

if "eager" not in _ALL_ATTN_FNS:
    if "paged|eager" in _ALL_ATTN_FNS:
        _ALL_ATTN_FNS.register("eager", _ALL_ATTN_FNS["paged|eager"])

# Transformers 5.x removed SlidingWindowCache; inject a stub so the trust_remote_code
# module for Isaac (modular_isaac.py) can be imported (uses it only in isinstance checks).
import transformers.cache_utils as _cache_utils

if not hasattr(_cache_utils, "SlidingWindowCache"):
    from transformers.cache_utils import Cache as _Cache

    class _SlidingWindowCache(_Cache):
        pass

    _cache_utils.SlidingWindowCache = _SlidingWindowCache

# Transformers 5.x renamed DefaultFastImageProcessorKwargs → ImagesKwargs.
# Isaac's modular code inherits from it; alias it back so the module imports.
import transformers.image_processing_utils_fast as _img_utils

if not hasattr(_img_utils, "DefaultFastImageProcessorKwargs"):
    from transformers.image_processing_utils_fast import ImagesKwargs as _ImagesKwargs

    _img_utils.DefaultFastImageProcessorKwargs = _ImagesKwargs

# Transformers 5.x moved TensorType from tokenization_utils to tokenization_utils_base.
# Isaac's modular code imports it from the old location; inject it back.
import transformers.tokenization_utils as _tok_utils

if not hasattr(_tok_utils, "TensorType"):
    from transformers.tokenization_utils_base import TensorType as _TensorType

    _tok_utils.TensorType = _TensorType

# Transformers 5.x changed RoPE config: rope_theta must live inside rope_parameters dict.
# IsaacConfig.__init__ can overwrite rope_parameters with an incomplete _rope_scaling dict
# (missing rope_theta) after convert_rope_params_to_dict already set it correctly.
# Patch compute_default_rope_parameters to fall back to config.rope_theta attribute.
def _make_rope_patch(cls):
    orig = cls.compute_default_rope_parameters

    @staticmethod
    def patched(config, device=None, seq_len=None):
        params = config.rope_parameters or {}
        if "rope_theta" not in params:
            params = dict(params)
            params["rope_theta"] = getattr(config, "rope_theta", 10000.0)
            config.rope_parameters = params
        return orig(config, device, seq_len)

    cls.compute_default_rope_parameters = patched


from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding as _Qwen3RotaryEmbedding
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRotaryEmbedding as _Qwen25VLRotaryEmbedding

_make_rope_patch(_Qwen3RotaryEmbedding)
_make_rope_patch(_Qwen25VLRotaryEmbedding)

from ...tools.utils import get_file, cast_input_to_type
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
    """Available Isaac model variants."""

    ISAAC_0_2_2B_PREVIEW = "0.2_2B_Preview"


class ModelLoader(ForgeModel):
    """Isaac model loader implementation for multimodal visual question answering tasks."""

    _VARIANTS = {
        ModelVariant.ISAAC_0_2_2B_PREVIEW: ModelConfig(
            pretrained_model_name="PerceptronAI/Isaac-0.2-2B-Preview",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ISAAC_0_2_2B_PREVIEW

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Isaac",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        # Load config first so we can override the vision attn implementation.
        # IsaacVisionConfig defaults to "flash_attention_2" which requires CUDA;
        # override to "sdpa" before instantiating the model.
        config = AutoConfig.from_pretrained(pretrained_model_name, trust_remote_code=True)
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "_attn_implementation"):
            config.vision_config._attn_implementation = "sdpa"

        model_kwargs = {"trust_remote_code": True, "attn_implementation": "eager", "config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        # attn_implementation="eager" propagates recursively and overrides the
        # vision_config._attn_implementation we set before loading. IsaacVisionAttention's
        # eager path uses a broken packed-sequence matmul ([L,H,D]@[L,D,H]=[L,H,H] instead
        # of the correct [H,L,L]). sdpa_document_mask_forward handles packed sequences
        # correctly. Re-patch after loading so vision attention uses sdpa.
        for module in model.modules():
            if type(module).__name__ == "IsaacVisionAttention":
                module.vision_config._attn_implementation = "sdpa"
                break  # all share the same vision_config object

        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file)

        # Isaac's chat template expects string content; list-style content produces
        # empty content and the <image> token never appears in the text.
        # Use the string format recommended in the HuggingFace README.
        vision_token = getattr(self.processor, "vision_token", "<image>")
        conversation = [
            {
                "role": "user",
                "content": f"{vision_token}\nWhat is shown in this image?",
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(text=text_prompt, images=[image], return_tensors="pt")

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].is_floating_point():
                    inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return dict(inputs)

    def decode_output(self, outputs, input_length=None):
        if self.processor is None:
            self._load_processor()

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return self.processor.decode(next_token_id)
