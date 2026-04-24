# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
zai-org CogVLM2-Llama3-Chat-19B model loader implementation for multimodal visual question answering.

CogVLM2-Llama3-Chat-19B is a vision-language model based on Meta-Llama-3-8B-Instruct
with a vision encoder (~19B total parameters). Both the tokenizer and model require
trust_remote_code=True, and inputs are prepared through the model's custom
build_conversation_input_ids() method rather than an AutoProcessor.
"""

import sys
import types
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


def _apply_rotary_pytorch(
    x,
    cos,
    sin,
    seqlen_offsets=0,
    cu_seqlens=None,
    max_seqlen=None,
    interleaved=False,
    inplace=False,
    conjugate=False,
):
    """Pure PyTorch rotary embedding fallback replacing the triton/CUDA kernel."""
    batch, nheads, seqlen, headdim = x.shape
    rotary_dim = cos.shape[-1] * 2
    cos_b = cos.unsqueeze(1)
    sin_b = sin.unsqueeze(1)
    if conjugate:
        sin_b = -sin_b
    result = x if inplace else x.clone()
    x0 = x[..., : rotary_dim // 2].clone()
    x1 = x[..., rotary_dim // 2 : rotary_dim].clone()
    result[..., : rotary_dim // 2] = x0 * cos_b - x1 * sin_b
    result[..., rotary_dim // 2 : rotary_dim] = x0 * sin_b + x1 * cos_b
    return result


def _patch_cogvlm2_rotary_emb():
    """Patch triton rotary embedding in cogvlm2 util module with pure PyTorch."""
    for key, module in list(sys.modules.items()):
        if "cogvlm2" in key.lower() and key.endswith("util"):
            module.apply_rotary = _apply_rotary_pytorch
            module.apply_rotary_emb = _apply_rotary_pytorch
            module.apply_rotary_emb_func = _apply_rotary_pytorch
            break


def _install_xformers_stub():
    """Inject a minimal xformers stub so cogvlm2's visual.py can be imported.

    The model's vision encoder calls xops.memory_efficient_attention, which we
    replace with a standard scaled-dot-product attention fallback.
    """
    if "xformers" in sys.modules:
        return

    def _memory_efficient_attention(q, k, v, scale=None, **kwargs):
        if scale is None:
            scale = q.shape[-1] ** -0.5
        attn = torch.matmul(q * scale, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

    xformers = types.ModuleType("xformers")
    xformers_ops = types.ModuleType("xformers.ops")
    xformers_ops.memory_efficient_attention = _memory_efficient_attention
    xformers.ops = xformers_ops
    sys.modules["xformers"] = xformers
    sys.modules["xformers.ops"] = xformers_ops


from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available zai-org CogVLM2-Llama3-Chat-19B model variants."""

    COGVLM2_LLAMA3_CHAT_19B = "llama3-chat-19B"


class ModelLoader(ForgeModel):
    """zai-org CogVLM2-Llama3-Chat-19B model loader for multimodal visual question answering."""

    _VARIANTS = {
        ModelVariant.COGVLM2_LLAMA3_CHAT_19B: ModelConfig(
            pretrained_model_name="zai-org/cogvlm2-llama3-chat-19B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.COGVLM2_LLAMA3_CHAT_19B

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="zai-org CogVLM2-Llama3-Chat-19B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        _install_xformers_stub()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        _install_xformers_stub()
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()
        _patch_cogvlm2_rotary_emb()

        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for CogVLM2-Llama3-Chat-19B.

        CogVLM2 exposes a custom build_conversation_input_ids() method on the model
        itself that tokenizes the query and preprocesses the image together. The
        returned tensors are unbatched, so we add a leading batch dimension here.
        """
        if self.tokenizer is None:
            self._load_tokenizer()
        if self.model is None:
            self.load_model(dtype_override=dtype_override)

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer,
            query=self.sample_text,
            images=[image],
            template_version="chat",
        )

        image_tensor = input_by_model["images"][0]
        if dtype_override is not None:
            image_tensor = image_tensor.to(dtype_override)

        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0),
            "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0),
            "attention_mask": input_by_model["attention_mask"].unsqueeze(0),
            "images": [[image_tensor]],
        }

        if batch_size > 1:
            inputs["input_ids"] = inputs["input_ids"].repeat_interleave(
                batch_size, dim=0
            )
            inputs["token_type_ids"] = inputs["token_type_ids"].repeat_interleave(
                batch_size, dim=0
            )
            inputs["attention_mask"] = inputs["attention_mask"].repeat_interleave(
                batch_size, dim=0
            )
            inputs["images"] = [[image_tensor] for _ in range(batch_size)]

        return inputs

    def decode_output(self, outputs, input_length=None):
        """Decode model outputs into human-readable text."""
        if isinstance(outputs, str):
            return outputs

        if self.tokenizer is None:
            self._load_tokenizer()

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        return self.tokenizer.decode(next_token_id)
