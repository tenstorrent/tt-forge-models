# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Emu3-Gen model loader implementation for text-to-image generation.
"""

import sys
import torch
import transformers
from typing import Optional
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module

# transformers 5.x removed is_torch_fx_available; modeling_emu3.py (remote code) still imports it.
# Inject a shim so the remote module can load without ImportError.
if not hasattr(transformers.utils.import_utils, "is_torch_fx_available"):

    def _is_torch_fx_available():
        return False

    transformers.utils.import_utils.is_torch_fx_available = _is_torch_fx_available
    sys.modules["transformers.utils.import_utils"].__dict__[
        "is_torch_fx_available"
    ] = _is_torch_fx_available

from ....tools.utils import cast_input_to_type
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
    """Available Emu3-Gen model variants."""

    EMU3_GEN = "Gen"


def _reinit_rope_caches(model: torch.nn.Module, dtype=None) -> None:
    """Re-initialize RoPE inv_freq + cos/sin caches after from_pretrained.

    transformers 5.x uses init_empty_weights() (meta device) during loading.
    persistent=False buffers (inv_freq, cos_cached, sin_cached) are not saved
    in the checkpoint, so they come out of from_pretrained as uninitialized
    float32 tensors.  We recompute them here in the correct dtype so that
    attention layers produce valid outputs.
    """
    target_dtype = dtype if dtype is not None else torch.float32
    device = next(model.parameters()).device

    for module in model.modules():
        if not hasattr(module, "inv_freq"):
            continue
        if not hasattr(module, "base") or not hasattr(module, "dim"):
            continue
        # Recompute inv_freq = 1 / (base^(2k/dim)) in float32, then cast.
        arange = torch.arange(0, module.dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (module.base ** (arange / module.dim))
        module.register_buffer("inv_freq", inv_freq.to(target_dtype), persistent=False)
        # Recompute the cos/sin cache for the full max_position_embeddings.
        max_pos = module.max_position_embeddings
        t = torch.arange(max_pos, dtype=inv_freq.dtype, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos().to(target_dtype)
        sin_cached = emb.sin().to(target_dtype)
        module.register_buffer("cos_cached", cos_cached, persistent=False)
        module.register_buffer("sin_cached", sin_cached, persistent=False)
        module.max_seq_len_cached = max_pos


class ModelLoader(ForgeModel):
    """Emu3-Gen model loader implementation for text-to-image generation tasks."""

    _VARIANTS = {
        ModelVariant.EMU3_GEN: ModelConfig(
            pretrained_model_name="BAAI/Emu3-Gen",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EMU3_GEN

    VISION_TOKENIZER_NAME = "BAAI/Emu3-VisionTokenizer"

    sample_prompt = "a portrait of young girl. masterpiece, film grained, best quality."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.image_processor = None
        self.image_tokenizer = None
        self.processor = None
        self.model = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Emu3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        # transformers 5.x encode(list) returns [[id], [id], ...] instead of [id, id, ...].
        # build_const_helper expects a flat list; flatten it.
        _orig_encode = self.tokenizer.encode
        def _encode_flat(text, *args, **kwargs):
            result = _orig_encode(text, *args, **kwargs)
            if isinstance(result, list) and result and isinstance(result[0], list):
                return [sub[0] for sub in result]
            return result
        self.tokenizer.encode = _encode_flat
        return self.tokenizer

    def _load_image_processor(self):
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.VISION_TOKENIZER_NAME, trust_remote_code=True, use_fast=False
        )
        return self.image_processor

    def _load_image_tokenizer(self, dtype_override=None):
        kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override
        self.image_tokenizer = AutoModel.from_pretrained(
            self.VISION_TOKENIZER_NAME, **kwargs
        )
        self.image_tokenizer.eval()
        return self.image_tokenizer

    def _load_processor(self, dtype_override=None):
        if self.image_processor is None:
            self._load_image_processor()
        if self.image_tokenizer is None:
            self._load_image_tokenizer(dtype_override=dtype_override)
        if self.tokenizer is None:
            self._load_tokenizer()

        Emu3Processor = get_class_from_dynamic_module(
            "processing_emu3.Emu3Processor",
            self._variant_config.pretrained_model_name,
        )
        # transformers 5.x ProcessorMixin.get_attributes() now inspects __init__
        # signatures: "vision_tokenizer" contains "tokenizer" and is counted as a
        # third attribute, but Emu3Processor.__init__ only passes 2 args to
        # super().__init__(). Override get_attributes to return the 2 actual attributes.
        Emu3Processor.get_attributes = classmethod(
            lambda cls: ["image_processor", "tokenizer"]
        )
        self.processor = Emu3Processor(
            self.image_processor, self.image_tokenizer, self.tokenizer
        )
        return self.processor

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        # Load config and reset rope_scaling: transformers 5.x injects
        # {'rope_theta': ..., 'rope_type': 'default'} but the remote
        # modeling code only handles None/'linear'/'dynamic'.
        if self.config is None:
            self.load_config()
        self.config.rope_scaling = None

        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
            "config": self.config,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        self.config = model.config

        # transformers 5.x uses init_empty_weights() (meta device) during loading;
        # persistent=False buffers like inv_freq are not in the checkpoint and come
        # out uninitialized. Recompute RoPE caches for every attention layer so that
        # cos_cached/sin_cached are valid in the requested dtype.
        _reinit_rope_caches(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)
        if self.config is None:
            self.load_config()

        inputs = self.processor(
            text=self.sample_prompt,
            mode="G",
            ratio="1:1",
            image_area=self.config.image_area,
            return_tensors="pt",
            padding="longest",
        )

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].is_floating_point():
                    inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        result = dict(inputs)
        # image_size is metadata for constrained generation, not a model forward() arg
        result.pop("image_size", None)
        # transformers 5.x removed DynamicCache.get_usable_length(); disable KV cache
        # for single forward-pass inference (fresh cache = length 0 either way)
        result["use_cache"] = False
        return result

    def decode_output(self, outputs):
        if self.processor is None:
            self._load_processor()

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            return self.processor.decode(outputs[0])

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        return self.tokenizer.decode(next_token_id)
