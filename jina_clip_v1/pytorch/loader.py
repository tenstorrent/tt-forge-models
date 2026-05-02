# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Jina CLIP v1 model loader implementation for image-text similarity.
"""
import math
from functools import partial as _partial

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.checkpoint import checkpoint as _checkpoint
from transformers import AutoModel, AutoProcessor
from typing import Optional

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


def _recompute_rope_buffers(model):
    """Recompute VisionRotaryEmbeddingFast non-persistent buffers after meta init.

    Under transformers 5.x, meta device init leaves freqs_cos/freqs_sin as NaN
    after weight loading because persistent=False buffers are absent from the
    checkpoint.  Recompute them on CPU using config parameters.
    """
    try:
        pt_seq_len = model.config.vision_config.pt_hw_seq_len
    except AttributeError:
        return

    for module in model.modules():
        if type(module).__name__ != "VisionRotaryEmbeddingFast":
            continue

        cos_buf = module.freqs_cos
        N, D = cos_buf.shape
        ft_seq_len = int(round(math.sqrt(N)))
        assert ft_seq_len * ft_seq_len == N, f"Unexpected rope buffer shape: {cos_buf.shape}"
        dim = D // 2

        theta = 10000
        half_freqs = torch.arange(0, dim, 2)[: (dim // 2)].float()
        freqs = 1.0 / (theta ** (half_freqs / dim))
        t = torch.arange(ft_seq_len).float() / ft_seq_len * pt_seq_len
        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = freqs.repeat_interleave(2, dim=-1)
        freqs_combined = torch.cat(
            [
                freqs[:, None, :].expand(ft_seq_len, ft_seq_len, dim),
                freqs[None, :, :].expand(ft_seq_len, ft_seq_len, dim),
            ],
            dim=-1,
        )

        new_cos = freqs_combined.cos().view(-1, D)
        new_sin = freqs_combined.sin().view(-1, D)

        module.register_buffer("freqs_cos", new_cos, persistent=False)
        module.register_buffer("freqs_sin", new_sin, persistent=False)


def _fix_eva_rope_forward_accumulation(model):
    """Replace EVAVisionTransformer.forward_features to stop partial wrappers accumulating.

    The original forward_features re-wraps self.rope.forward with a new partial on every
    eval call; Dynamo sees the graph change and recompiles past its limit, producing
    wrong results.  Fix: use a stable pre-created partial so Dynamo's guard is stable.
    """
    for eva in model.modules():
        if type(eva).__name__ != "EVAVisionTransformer":
            continue
        if eva.rope is None:
            continue

        rope = eva.rope
        orig_rope_fwd = type(rope).forward.__get__(rope, type(rope))
        eval_fwd = _partial(orig_rope_fwd, patch_indices_keep=None)

        def _fwd_feat(
            x,
            return_all_features=False,
            _eva=eva,
            _eval_fwd=eval_fwd,
            _orig_fwd=orig_rope_fwd,
        ):
            x = _eva.patch_embed(x)
            batch_size, seq_len, _ = x.size()
            cls_tokens = _eva.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            if _eva.pos_embed is not None:
                x = x + _eva.pos_embed
            x = _eva.pos_drop(x)
            if _eva.rope is not None:
                if _eva.training and not isinstance(_eva.patch_dropout, nn.Identity):
                    x, patch_indices_keep = _eva.patch_dropout(x)
                    _eva.rope.forward = _partial(
                        _orig_fwd, patch_indices_keep=patch_indices_keep
                    )
                else:
                    _eva.rope.forward = _eval_fwd
                    x = _eva.patch_dropout(x)
            else:
                x = _eva.patch_dropout(x)
            rel_pos_bias = _eva.rel_pos_bias() if _eva.rel_pos_bias is not None else None
            for blk in _eva.blocks:
                if _eva.grad_checkpointing:
                    x = _checkpoint(blk, x, (rel_pos_bias,))
                else:
                    x = blk(x, rel_pos_bias=rel_pos_bias)
            if not return_all_features:
                x = _eva.norm(x)
                if _eva.fc_norm is not None:
                    return _eva.fc_norm(x.mean(1))
                else:
                    return x[:, 0]
            return x

        eva.forward_features = _fwd_feat


class ModelVariant(StrEnum):
    """Available Jina CLIP v1 model variants."""

    JINA_CLIP_V1 = "jina-clip-v1"


class ModelLoader(ForgeModel):
    """Jina CLIP v1 model loader for image-text similarity tasks."""

    _VARIANTS = {
        ModelVariant.JINA_CLIP_V1: ModelConfig(
            pretrained_model_name="jinaai/jina-clip-v1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.JINA_CLIP_V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Jina-CLIP-v1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Jina CLIP v1 model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Jina CLIP v1 model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # Transformers 5.x unconditionally initializes models on the meta device.
        # EVAVisionTransformer calls .item() on a torch.linspace result during __init__
        # to compute stochastic-depth rates, which raises
        #   "Tensor.item() cannot be called on meta tensors"
        # Patch torch.linspace to force CPU device so .item() succeeds.
        _orig_linspace = torch.linspace

        def _cpu_linspace(*args, **kw):
            with torch.device("cpu"):
                return _orig_linspace(*args, **kw)

        torch.linspace = _cpu_linspace
        try:
            model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        finally:
            torch.linspace = _orig_linspace

        # VisionRotaryEmbeddingFast registers freqs_cos/freqs_sin as non-persistent
        # buffers. Under the meta device context, torch.arange produces meta tensors;
        # since the buffers are absent from the checkpoint (persistent=False),
        # transformers materializes them with uninitialized (NaN) data after loading.
        # Recompute them on CPU using config params.
        _recompute_rope_buffers(model)

        # forward_features re-wraps self.rope.forward with a new partial each eval
        # call, causing Dynamo recompilation past its limit. Replace with a stable version.
        _fix_eva_rope_forward_accumulation(model)

        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Jina CLIP v1 model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size (default 1).

        Returns:
            dict: Input tensors for the model.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                pretrained_model_name,
                trust_remote_code=True,
            )

        # Load sample image via get_file to avoid load_dataset/spacy collision
        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        self.text_prompts = ["a photo of a cat", "a photo of a dog"]

        inputs = self.processor(
            text=self.text_prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        # Replicate tensors for batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs

    def post_process(self, outputs):
        """Post-process model outputs to extract similarity scores.

        Args:
            outputs: Raw model output.
        """
        if self.text_prompts is None:
            self.text_prompts = ["a photo of a cat", "a photo of a dog"]

        logits_per_image = outputs[0]
        probs = logits_per_image.softmax(dim=1)

        for i, text in enumerate(self.text_prompts):
            print(f"Probability of '{text}':", probs[0, i].item())

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        Args:
            fwd_output: Output from the model's forward pass.

        Returns:
            torch.Tensor: Concatenated flattened outputs for backward pass.
        """
        if isinstance(fwd_output, tuple):
            tensors = []
            for item in fwd_output:
                if isinstance(item, torch.Tensor):
                    tensors.append(item.flatten())
                elif hasattr(item, "last_hidden_state"):
                    tensors.append(item.last_hidden_state.flatten())
            if tensors:
                return torch.cat(tensors, dim=0)
        return fwd_output
