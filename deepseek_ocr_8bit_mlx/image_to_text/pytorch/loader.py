# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-OCR 8-bit MLX model loader implementation for document OCR tasks.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.models.llama import modeling_llama as _llama_module
import transformers.utils.import_utils as _trf_import_utils
from typing import Optional

# LlamaFlashAttention2 was removed in transformers 5.x; the model's remote code
# imports it at module level so we must shim it before trust_remote_code loading.
# Additionally, transformers 5.x LlamaAttention requires position_embeddings=(cos, sin)
# and returns 2 values, but the remote DeepseekV2 code uses the 4.x interface
# (position_ids input, 3 return values). Patch forward at the class level to bridge.
if not hasattr(_llama_module, "LlamaFlashAttention2"):
    _llama_module.LlamaFlashAttention2 = _llama_module.LlamaAttention

_orig_llama_attn_fwd = _llama_module.LlamaAttention.forward


def _compat_llama_attn_fwd(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    position_embeddings=None,
    **kwargs,
):
    if position_embeddings is None:
        bsz, seq_len = hidden_states.shape[:2]
        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=hidden_states.device)
                .unsqueeze(0)
                .expand(bsz, -1)
            )
        rope_theta = float(getattr(self.config, "rope_theta", 10000.0))
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(
                    0, self.head_dim, 2, dtype=torch.float32, device=position_ids.device
                )
                / self.head_dim
            )
        )
        inv_freq_exp = inv_freq[None, :, None].expand(bsz, -1, 1)
        freqs = (inv_freq_exp @ position_ids[:, None, :].float()).transpose(1, 2)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(hidden_states.dtype)
        sin = emb.sin().to(hidden_states.dtype)
        position_embeddings = (cos, sin)
    result = _orig_llama_attn_fwd(
        self,
        hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
    )
    if isinstance(result, tuple) and len(result) == 2:
        return result + (None,)
    return result


_llama_module.LlamaAttention.forward = _compat_llama_attn_fwd

# is_torch_fx_available was removed in transformers 5.x; shim before remote code loads.
if not hasattr(_trf_import_utils, "is_torch_fx_available"):
    _trf_import_utils.is_torch_fx_available = lambda: True

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
from ....tools.utils import get_file

# Reuse preprocessing utilities from DeepSeek-OCR
from ....deepseek.deepseek_ocr.pytorch.src.model_utils import preprocess


def _patch_deepseekocr_cuda_calls(model):
    """Patch DeepseekOCRModel.forward to remove hard-coded .cuda() calls."""
    import sys
    import types

    for name, module in model.named_modules():
        if type(module).__name__ != "DeepseekOCRModel":
            continue
        cls = type(module)
        orig_forward = cls.forward

        def _patched_forward(self, *args, **kwargs):
            import torch as _torch

            _orig_cuda = _torch.Tensor.cuda

            def _noop_cuda(t, *a, **kw):
                return t

            _torch.Tensor.cuda = _noop_cuda
            try:
                return orig_forward(self, *args, **kwargs)
            finally:
                _torch.Tensor.cuda = _orig_cuda

        cls.forward = _patched_forward
        break


class ModelVariant(StrEnum):
    """Available DeepSeek-OCR 8-bit MLX model variants."""

    DEEPSEEK_OCR_8BIT = "DeepSeek_OCR_8bit"


class ModelLoader(ForgeModel):
    """DeepSeek-OCR 8-bit MLX model loader implementation for document OCR tasks."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_OCR_8BIT: ModelConfig(
            pretrained_model_name="mlx-community/DeepSeek-OCR-8bit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_OCR_8BIT

    sample_prompt = "<image>\n<|grounding|>Convert the document to markdown. "

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DeepSeek-OCR 8-bit MLX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True, "use_safetensors": True}
        # MLX quantized variants may have mismatched weight shapes
        model_kwargs["ignore_mismatched_sizes"] = True
        model_kwargs |= kwargs

        # transformers 5.x rejects MLX quantization configs (no quant_method); strip it.
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if hasattr(config, "quantization_config") and not hasattr(
            config.quantization_config, "quant_method"
        ):
            del config.quantization_config
        model_kwargs["config"] = config

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)

        # CLIPVisionEmbeddings.position_ids gets corrupted to large values after model.to()
        # due to an expand-tensor aliasing issue. Clamp it before each forward call.
        def _clip_pos_ids_hook(module, args):
            emb = getattr(module, "position_embedding", None)
            ids = getattr(module, "position_ids", None)
            if isinstance(emb, nn.Embedding) and ids is not None:
                n = emb.weight.shape[0]
                if int(ids.max()) >= n:
                    module.position_ids = ids.clamp(0, n - 1)

        for module in model.modules():
            if type(module).__name__ == "CLIPVisionEmbeddings":
                module.register_forward_pre_hook(_clip_pos_ids_hook)

        # The remote model code calls .cuda() on a mask tensor; patch it away.
        _patch_deepseekocr_cuda_calls(model)

        model.config.return_dict = False
        model.config.use_cache = False

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        image_file = get_file(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
        )

        inputs = preprocess(
            tokenizer=self.tokenizer,
            prompt=self.sample_prompt,
            image_file=image_file,
            base_size=1024,
            image_size=640,
            crop_mode=True,
        )

        if dtype_override is not None:
            for idx, (images_crop, images_ori) in enumerate(inputs["images"]):
                inputs["images"][idx] = (
                    images_crop.to(dtype_override),
                    images_ori.to(dtype_override),
                )

        return inputs
