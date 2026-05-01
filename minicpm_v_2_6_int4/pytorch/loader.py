# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM-V-2_6-int4 model loader implementation for multimodal inference
"""

import importlib
import sys
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoProcessor
from transformers.integrations.tensor_parallel import ALL_PARALLEL_STYLES

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

# Fix parallel styles issue for torch 2.7.0+ compatibility - works fine in torch 2.3.1
if ALL_PARALLEL_STYLES is None:
    import transformers.modeling_utils as mu

    mu.ALL_PARALLEL_STYLES = ["rowwise", "colwise", "headwise"]

# Monkey patch Resampler for compatibility - Fixes: Resampler doesn't have _initialize_weights method in torch 2.7.0
original_getattr = nn.Module.__getattr__


def patched_getattr(self, name):
    if name == "_initialize_weights" and self.__class__.__name__ == "Resampler":

        def _initialize_weights(module_self):
            if hasattr(module_self, "_init_weights"):
                module_self._init_weights(module_self)

        return _initialize_weights
    return original_getattr(self, name)


nn.Module.__getattr__ = patched_getattr


def _patch_cached_remote_files():
    """Fix transformers-5.x and TT-XLA incompatibilities in cached remote model files."""
    cache_base = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "modules"
        / "transformers_modules"
    )
    glob_prefix = "openbmb/MiniCPM_hyphen_V_hyphen_2_6_hyphen_int4/*"

    # Fix 1: MiniCPMV.__init__ never calls self.post_init(), so all_tied_weights_keys
    # (added in transformers 5.x) is never initialized, causing AttributeError in
    # _adjust_tied_keys_with_tied_pointers during from_pretrained.
    for path in cache_base.glob(f"{glob_prefix}/modeling_minicpmv.py"):
        text = path.read_text()
        old = "        self.terminators = ['<|im_end|>', '<|endoftext|>']\n"
        new = "        self.terminators = ['<|im_end|>', '<|endoftext|>']\n        self.post_init()\n"
        if old in text and new not in text:
            path.write_text(text.replace(old, new, 1))

    # Fix 2: get_vllm_embedding() filters tgt_sizes to tensors only, then calls
    # torch.vstack(). When tgt_sizes contains Python [h,w] lists (needed so
    # max_patch_len is a concrete Python int for static shapes), the filter removes
    # all entries leaving an empty list, which torch.vstack rejects. Handle both cases.
    for path in cache_base.glob(f"{glob_prefix}/modeling_minicpmv.py"):
        text = path.read_text()
        old = (
            "                tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]\n"
            "                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)\n"
        )
        new = (
            "                if tgt_sizes and isinstance(tgt_sizes[0], torch.Tensor):\n"
            "                    tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]\n"
            "                    tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)\n"
            "                else:\n"
            "                    tgt_sizes = torch.tensor(tgt_sizes, dtype=torch.int32)\n"
        )
        if old in text and new not in text:
            path.write_text(text.replace(old, new, 1))

    # Fix 3: MiniCPMVBatchFeature.to() uses cast_tensor() which calls
    # torch.is_floating_point(v) unconditionally. When tgt_sizes contains Python
    # int leaves (we convert them to avoid XLA dynamic-shape alignment padding),
    # cast_tensor receives bare ints and torch.is_floating_point(int) raises TypeError.
    for path in cache_base.glob(f"{glob_prefix}/image_processing_minicpmv.py"):
        text = path.read_text()
        old = (
            "        def cast_tensor(v):\n"
            "            # check if v is a floating point\n"
            "            if torch.is_floating_point(v):"
        )
        new = (
            "        def cast_tensor(v):\n"
            "            if not isinstance(v, torch.Tensor):\n"
            "                return v\n"
            "            # check if v is a floating point\n"
            "            if torch.is_floating_point(v):"
        )
        if old in text and new not in text:
            path.write_text(text.replace(old, new, 1))

    # Fix 4: torch.max() in resampler _adjust_pos_cache returns an XLA tensor on TT,
    # which makes max_h/max_w dynamic scalars. Use element-access for bs=1.
    for path in cache_base.glob(f"{glob_prefix}/resampler.py"):
        text = path.read_text()
        old4 = (
            "        max_h = torch.max(tgt_sizes[:, 0])\n"
            "        max_w = torch.max(tgt_sizes[:, 1])\n"
        )
        new4 = (
            "        max_h = int(tgt_sizes[:, 0][0]) if len(tgt_sizes) == 1"
            " else int(torch.max(tgt_sizes[:, 0]))\n"
            "        max_w = int(tgt_sizes[:, 1][0]) if len(tgt_sizes) == 1"
            " else int(torch.max(tgt_sizes[:, 1]))\n"
        )
        if old4 in text and new4 not in text:
            text = text.replace(old4, new4, 1)

        # Fix 5: torch.max(patch_len) returns a dynamic XLA scalar; use element-access
        # for bs=1 so max_patch_len is a concrete Python int.
        new5 = (
            "        max_patch_len = int(patch_len[0]) if len(patch_len) == 1"
            " else int(torch.max(patch_len))\n"
        )
        for old5 in [
            "        max_patch_len = torch.max(patch_len)\n",
            "        max_patch_len = int(torch.max(patch_len))\n",
        ]:
            if old5 in text and new5 not in text:
                text = text.replace(old5, new5, 1)
                break

        # Fix 6: XLA pads the last dimension of tensors to multiples of 8.
        # Pre-align max_patch_len to the next multiple of 8, then pad x and pos_embed
        # to the same aligned length. Use float dtype for key_padding_mask.
        old6 = (
            "        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool, device=device)\n"
            "\n"
            "        pos_embed = []\n"
            "        for i in range(bs):\n"
            "            tgt_h, tgt_w = tgt_sizes[i]\n"
            "            pos_embed.append(self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype))  # patches * D\n"
            "            key_padding_mask[i, patch_len[i]:] = True\n"
            "\n"
            "        pos_embed = torch.nn.utils.rnn.pad_sequence(\n"
            "            pos_embed, batch_first=True, padding_value=0.0).permute(1, 0, 2)  # BLD => L * B * D\n"
            "\n"
            "        x = self.kv_proj(x)  # B * L * D\n"
            "        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D\n"
        )
        new6 = (
            "        max_patch_len = ((max_patch_len + 7) // 8) * 8\n"
            "        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=dtype, device=device)\n"
            "\n"
            "        pos_embed = []\n"
            "        for i in range(bs):\n"
            "            tgt_h, tgt_w = tgt_sizes[i]\n"
            "            pos_embed.append(self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype))  # patches * D\n"
            "            key_padding_mask[i, int(patch_len[i]):] = float('-inf')\n"
            "\n"
            "        pos_embed = torch.nn.utils.rnn.pad_sequence(\n"
            "            pos_embed, batch_first=True, padding_value=0.0)\n"
            "        if pos_embed.shape[1] < max_patch_len:\n"
            "            pos_embed = pad(pos_embed, (0, 0, 0, max_patch_len - pos_embed.shape[1]))\n"
            "        pos_embed = pos_embed.permute(1, 0, 2)  # BLD => L * B * D\n"
            "\n"
            "        x = self.kv_proj(x)  # B * L * D\n"
            "        x = self.ln_kv(x)\n"
            "        if x.shape[1] < max_patch_len:\n"
            "            x = pad(x, (0, 0, 0, max_patch_len - x.shape[1]))\n"
            "        x = x.permute(1, 0, 2)  # L * B * D\n"
        )
        if old6 in text and new6 not in text:
            text = text.replace(old6, new6, 1)

        path.write_text(text)

    # Fix 7: XLA pads the last dimension of all_pixel_values to a multiple of 8.
    # Pre-align max_patches to the next multiple of 8 so CPU and TT pad identically.
    for path in cache_base.glob(f"{glob_prefix}/modeling_minicpmv.py"):
        text = path.read_text()
        old7 = (
            "                max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])\n"
            "\n"
            "                all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True,\n"
            "                                                                   padding_value=0.0)\n"
            "                B, L, _ = all_pixel_values.shape\n"
            "                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)\n"
            "\n"
            "                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)\n"
            "                for i in range(B):\n"
            "                    patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True\n"
        )
        new7 = (
            "                max_patches = ((int(torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])) + 7) // 8) * 8\n"
            "\n"
            "                all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True,\n"
            "                                                                   padding_value=0.0)\n"
            "                B, L, _ = all_pixel_values.shape\n"
            "                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)\n"
            "                _L_aligned = max_patches * all_pixel_values.shape[2]\n"
            "                if L < _L_aligned:\n"
            "                    all_pixel_values = torch.nn.functional.pad(all_pixel_values, (0, _L_aligned - L))\n"
            "\n"
            "                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)\n"
            "                for i in range(B):\n"
            "                    patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True\n"
        )
        if old7 in text and new7 not in text:
            path.write_text(text.replace(old7, new7, 1))

    # Fix 10: Eliminate Dynamo graph break — compute max_patches from data['tgt_sizes']
    # (the original Python [[h,w],...] list, never moved to TT device) using pure-Python
    # arithmetic so there are no tensor ops and no graph break.
    for path in cache_base.glob(f"{glob_prefix}/modeling_minicpmv.py"):
        text = path.read_text()
        old10 = (
            "                max_patches = ((int(torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])) + 7) // 8) * 8\n"
        )
        new10 = (
            "                max_patches = ((max(int(r[0]) * int(r[1]) for r in data['tgt_sizes']) + 7) // 8) * 8\n"
        )
        if old10 in text and new10 not in text:
            path.write_text(text.replace(old10, new10, 1))

    # Fix 11: patch_attn_mask built with in-place bool assignments on TT device doesn't
    # propagate correctly (XLA copy semantics). Build on CPU, then .to(device).
    for path in cache_base.glob(f"{glob_prefix}/modeling_minicpmv.py"):
        text = path.read_text()
        old11 = (
            "                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)\n"
            "                for i in range(B):\n"
            "                    patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True\n"
        )
        new11 = (
            "                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool)\n"
            "                for i in range(B):\n"
            "                    patch_attn_mask[i, 0, :int(tgt_sizes[i][0]) * int(tgt_sizes[i][1])] = True\n"
            "                patch_attn_mask = patch_attn_mask.to(device)\n"
        )
        if old11 in text and new11 not in text:
            path.write_text(text.replace(old11, new11, 1))

    # Fix 9: Replace scatter_-on-view / functional-scatter with cat+slice for vision
    # token placement. scatter_() on a view doesn't propagate on XLA (copy semantics),
    # and aten.scatter.src is untested in the TT backend. cat+slice uses well-supported
    # ops only. Requires image_bound to be Python int lists (Fix 8 in load_inputs()).
    for path in cache_base.glob(f"{glob_prefix}/modeling_minicpmv.py"):
        text = path.read_text()
        old9 = (
            "        bs = len(data['input_ids'])\n"
            "        for i in range(bs):\n"
            "            cur_vs_hs = vision_hidden_states[i]\n"
            "            if len(cur_vs_hs) > 0:\n"
            "                cur_vllm_emb = vllm_embedding[i]\n"
            "                cur_image_bound = data['image_bound'][i]\n"
            "                if len(cur_image_bound) > 0:\n"
            "                    image_indices = torch.stack(\n"
            "                        [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]\n"
            "                    ).to(vllm_embedding.device)\n"
            "\n"
            "                    cur_vllm_emb.scatter_(0, image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),\n"
            "                                          cur_vs_hs.view(-1, cur_vs_hs.shape[-1]))\n"
            "                elif self.training:\n"
            "                    cur_vllm_emb += cur_vs_hs[0].mean() * 0\n"
            "\n"
            "        return vllm_embedding, vision_hidden_states\n"
        )
        new9 = (
            "        bs = len(data['input_ids'])\n"
            "        rows = []\n"
            "        for i in range(bs):\n"
            "            cur_vs_hs = vision_hidden_states[i]\n"
            "            cur_vllm_emb = vllm_embedding[i]\n"
            "            if len(cur_vs_hs) > 0:\n"
            "                cur_image_bound = data['image_bound'][i]\n"
            "                if len(cur_image_bound) > 0:\n"
            "                    parts = []\n"
            "                    prev_end = 0\n"
            "                    for k, r in enumerate(cur_image_bound):\n"
            "                        start, end = r[0], r[1]\n"
            "                        parts.append(cur_vllm_emb[prev_end:start])\n"
            "                        parts.append(cur_vs_hs[k])\n"
            "                        prev_end = end\n"
            "                    parts.append(cur_vllm_emb[prev_end:])\n"
            "                    cur_vllm_emb = torch.cat(parts, dim=0)\n"
            "                elif self.training:\n"
            "                    cur_vllm_emb = cur_vllm_emb + cur_vs_hs[0].mean() * 0\n"
            "            rows.append(cur_vllm_emb)\n"
            "        vllm_embedding = torch.stack(rows, dim=0)\n"
            "\n"
            "        return vllm_embedding, vision_hidden_states\n"
        )
        if old9 in text and new9 not in text:
            path.write_text(text.replace(old9, new9, 1))

    # Invalidate module cache so patched files are re-imported by from_pretrained.
    for key in list(sys.modules):
        if "MiniCPM_hyphen_V_hyphen_2_6_hyphen_int4" in key or (
            "minicpm" in key.lower() and "int4" in key.lower()
        ):
            del sys.modules[key]
    importlib.invalidate_caches()


def _dequantize_bnb4_to_bf16(model):
    """Replace bitsandbytes Linear4bit modules with standard nn.Linear (BF16)."""
    import bitsandbytes as bnb

    for name in list(dict(model.named_modules()).keys()):
        module = reduce(getattr, name.split("."), model) if name else model
        if not isinstance(module, bnb.nn.Linear4bit):
            continue
        with torch.no_grad():
            weight_data = bnb.functional.dequantize_4bit(
                module.weight.data,
                module.weight.quant_state,
            ).to(torch.bfloat16)
        new_linear = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            dtype=torch.bfloat16,
        )
        new_linear.weight = nn.Parameter(weight_data)
        if module.bias is not None:
            new_linear.bias = nn.Parameter(module.bias.data.to(torch.bfloat16))
        parent_name, child_name = (
            (name.rsplit(".", 1)) if "." in name else ("", name)
        )
        parent = reduce(getattr, parent_name.split("."), model) if parent_name else model
        setattr(parent, child_name, new_linear)
    return model


class MiniCPMVForwardWrapper(nn.Module):
    """Wraps MiniCPMV to accept processor output as kwargs instead of a 'data' dict."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, pixel_values, tgt_sizes, image_bound, **kwargs):
        seq_len = input_ids.shape[1]
        position_ids = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand_as(input_ids)
        )
        data = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "tgt_sizes": tgt_sizes,
            "image_bound": image_bound,
            "position_ids": position_ids,
        }
        return self.model.forward(data, **kwargs)


@dataclass
class MiniCPMVInt4Config(ModelConfig):
    """Configuration specific to MiniCPM-V-2_6-int4 models"""

    pretrained_model_name: str = "openbmb/MiniCPM-V-2_6-int4"


class ModelVariant(StrEnum):
    """Available MiniCPM-V-2_6-int4 model variants."""

    DEFAULT = "Default"


_VARIANTS = {
    ModelVariant.DEFAULT: MiniCPMVInt4Config(
        pretrained_model_name="openbmb/MiniCPM-V-2_6-int4",
    ),
}


class ModelLoader(ForgeModel):
    """MiniCPM-V-2_6-int4 model loader implementation for multimodal inference."""

    _VARIANTS = _VARIANTS
    DEFAULT_VARIANT = ModelVariant.DEFAULT

    sample_text = "Describe this image in detail."

    def __init__(self, variant=None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[StrEnum] = None) -> ModelInfo:
        return ModelInfo(
            model="MiniCPM-V 2.6 int4",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, **kwargs):
        _patch_cached_remote_files()

        config = self._variant_config

        model = AutoModel.from_pretrained(
            config.pretrained_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            **kwargs,
        )

        # Dequantize bitsandbytes NF4 Linear4bit -> nn.Linear BF16 for TT hardware.
        _dequantize_bnb4_to_bf16(model)

        model.eval()

        if self.processor is None:
            self._load_processor()

        return MiniCPMVForwardWrapper(model)

    def load_inputs(self, **kwargs) -> Dict[str, Any]:
        if self.processor is None:
            self._load_processor()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file).convert("RGB")

        msgs = [{"role": "user", "content": [image, self.sample_text]}]

        # Replicate the chat() method's input processing logic
        copy_msgs = deepcopy(msgs)
        images = []
        for msg in copy_msgs:
            content = msg["content"]
            if isinstance(content, str):
                content = [content]
            cur_msgs = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_msgs.append("(<image>./</image>)")
                elif isinstance(c, str):
                    cur_msgs.append(c)
            msg["content"] = "\n".join(cur_msgs)

        prompt = self.processor.tokenizer.apply_chat_template(
            copy_msgs, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            [prompt], [images], return_tensors="pt", max_length=8192
        )
        inputs.pop("image_sizes", None)

        # Fix 8: Convert tgt_sizes from tensor to Python [[h, w], ...] pairs so
        # modeling_minicpmv.py Fix 10 can compute max_patches with pure-Python
        # arithmetic (no graph break). The processor returns a [batch, slices, 2]
        # tensor; iterating gives [slices, 2] sub-tensors; .tolist() gives [[h,w],...].
        if "tgt_sizes" in inputs:
            flat_tgt = []
            for ts in inputs["tgt_sizes"]:
                if isinstance(ts, torch.Tensor):
                    flat_tgt.extend(ts.tolist())
                else:
                    flat_tgt.extend(ts)
            inputs["tgt_sizes"] = flat_tgt

        # Fix 8 (cont): Keep image_bound as Python int lists so to_device() doesn't
        # move bounds to TT device. torch.arange(TT_scalar, TT_scalar) would call
        # .item() internally, triggering PJRT Error 13 and making scatter a no-op.
        if "image_bound" in inputs:
            inputs["image_bound"] = [
                bounds.tolist() if isinstance(bounds, torch.Tensor) else bounds
                for bounds in inputs["image_bound"]
            ]

        return dict(inputs)

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        return fwd_output
