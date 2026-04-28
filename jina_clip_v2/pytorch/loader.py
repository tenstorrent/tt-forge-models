# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Jina CLIP v2 model loader implementation for image-text similarity.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as _F
import torch.nn.utils.parametrize as _parametrize
from functools import partial as _partial
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
from datasets import load_dataset


def _recompute_rope_buffers(model):
    """Recompute VisionRotaryEmbeddingFast non-persistent buffers after meta init.

    Under transformers 5.x, meta device init leaves freqs_cos/freqs_sin uninitialized
    after weight loading because persistent=False buffers are absent from the checkpoint.
    Recompute them directly on CPU using config parameters.
    """
    try:
        pt_seq_len = model.config.vision_config.pt_hw_seq_len
    except AttributeError:
        return

    for module in model.modules():
        if type(module).__name__ != "VisionRotaryEmbeddingFast":
            continue

        cos_buf = module.freqs_cos
        # Buffer shape: [ft_seq_len^2, 2*half_head_dim]
        N, D = cos_buf.shape
        ft_seq_len = int(round(math.sqrt(N)))
        assert ft_seq_len * ft_seq_len == N, f"Unexpected rope buffer shape: {cos_buf.shape}"
        dim = D // 2  # half_head_dim

        # Reproduce VisionRotaryEmbeddingFast.__init__ on CPU (freqs_for='lang', theta=10000)
        theta = 10000
        half_freqs = torch.arange(0, dim, 2)[: (dim // 2)].float()
        freqs = 1.0 / (theta ** (half_freqs / dim))  # [dim//2]
        t = torch.arange(ft_seq_len).float() / ft_seq_len * pt_seq_len  # [ft_seq_len]
        freqs = torch.einsum("..., f -> ... f", t, freqs)  # [ft_seq_len, dim//2]
        freqs = freqs.repeat_interleave(2, dim=-1)  # [ft_seq_len, dim]
        freqs_combined = torch.cat(
            [
                freqs[:, None, :].expand(ft_seq_len, ft_seq_len, dim),
                freqs[None, :, :].expand(ft_seq_len, ft_seq_len, dim),
            ],
            dim=-1,
        )  # [ft_seq_len, ft_seq_len, 2*dim]

        new_cos = freqs_combined.cos().view(-1, D)
        new_sin = freqs_combined.sin().view(-1, D)

        module.register_buffer("freqs_cos", new_cos, persistent=False)
        module.register_buffer("freqs_sin", new_sin, persistent=False)


def _fix_eva_rope_forward_accumulation(model):
    """Replace EVAVisionTransformer.forward_features to stop partial wrappers accumulating.

    The original forward_features does on every eval call:
      self.rope.forward = partial(self.rope.forward, patch_indices_keep=None)
    Each call creates a new callable object; Dynamo sees the graph change and
    recompiles, hitting the limit (18 calls > 8 limit) and producing wrong results.

    Fix: replace the instance's forward_features with a closure that uses a
    pre-created stable partial (same object every call) so Dynamo's guard is stable.
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


def _lora_linear_forward(self, input, task_id=None, residual=False):
    """Shared LoRA linear forward for all LoRA layers.

    Defined at module level (not inside a loop) so Dynamo sees the same
    function object for every layer.  After remove_parametrizations,
    type(self) == nn.Linear for all layers, keeping Dynamo's type guard
    stable across 100+ linear layers.
    """
    if task_id is not None:
        delta_w = torch.mm(
            self._lora_B[task_id], self._lora_A[task_id]
        ) * self._lora_scaling
        weights = self.weight + delta_w
    else:
        weights = self.weight
    out = _F.linear(input, weights, self.bias)
    if residual:
        return out, input
    return out


def _fix_lora_forward_dynamo_guards(model):
    """Fix Dynamo recompile limit from LoRA parametrize-created unique subclasses.

    torch.nn.utils.parametrize.register_parametrization creates a unique
    Python subclass per module instance.  With 28 transformer layers each
    having 4 LoRA-parametrized linears, Dynamo guards fail on type(self)
    for each unique subclass, recompiling 18+ times (>8 limit) and falling
    back to eager with wrong outputs (pcc=-1.0).

    Fix: remove parametrize (restoring nn.Linear for all layers), store
    LoRA A/B weights as registered buffers so Dynamo treats them as symbolic
    graph inputs, and install the single module-level _lora_linear_forward so
    Dynamo compiles once (type guard and function guard both stable).
    """
    for module in model.modules():
        if not (
            isinstance(module, nn.Linear)
            and hasattr(module, "parametrizations")
            and hasattr(module.parametrizations, "weight")
            and len(module.parametrizations.weight) > 0
        ):
            continue
        lp = module.parametrizations.weight[0]
        if type(lp).__name__ != "LoRAParametrization":
            continue

        # Capture LoRA state before removing parametrize
        lora_A = lp.lora_A.detach().clone()   # [num_tasks, rank, fan_in]
        lora_B = lp.lora_B.detach().clone()   # [num_tasks, fan_out, rank]
        scaling = float(lp.scaling)

        # Remove parametrize → restores module.__class__ to nn.Linear (shared type)
        # leave_parametrized=True: module.weight becomes the materialized weight
        # (LoRAParametrization.forward is identity so base weight is unchanged)
        _parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)

        # Store LoRA weights as buffers so Dynamo treats them as symbolic inputs
        module.register_buffer("_lora_A", lora_A, persistent=False)
        module.register_buffer("_lora_B", lora_B, persistent=False)
        module.register_buffer(
            "_lora_scaling",
            torch.tensor(scaling, dtype=lora_A.dtype),
            persistent=False,
        )

        # Install the single shared forward (replaces old per-layer new_forward)
        module.forward = _lora_linear_forward.__get__(module, module.__class__)


class ModelVariant(StrEnum):
    """Available Jina CLIP v2 model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Jina CLIP v2 model loader for image-text similarity tasks."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="jinaai/jina-clip-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="JINA_CLIP_V2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant."""
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Jina CLIP v2 model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Jina CLIP v2 model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True, "return_dict": False}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # Transformers 5.x unconditionally initializes models on the meta device
        # (get_init_context always appends torch.device("meta"), the old
        # low_cpu_mem_usage kwarg is silently ignored).  EVAVisionTransformer
        # calls .item() on a torch.linspace result during __init__ to compute
        # stochastic-depth rates, which raises
        #   "Tensor.item() cannot be called on meta tensors"
        # Patch torch.linspace to force CPU device so .item() succeeds.
        # The nested torch.device("cpu") context overrides the outer meta context
        # for this specific call only; model parameters continue to use meta device
        # (ensuring correct dtype after weight loading).
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
        # buffers computed from torch.arange during __init__.  Under the meta device
        # context, those arange calls produce meta tensors; since the buffers are
        # absent from the checkpoint (persistent=False), transformers materializes
        # them with uninitialized (NaN) data after loading.  Recompute them on CPU
        # using the config params to restore correct values.
        _recompute_rope_buffers(model)

        # forward_features re-wraps self.rope.forward with a new partial each eval
        # call, causing Dynamo to recompile 18 times (limit=8) and fall back to eager.
        # Replace each EVAVisionTransformer's forward_features with a stable version.
        _fix_eva_rope_forward_accumulation(model)

        # parametrize.register_parametrization creates a unique Python subclass per
        # module instance.  With 100+ LoRA-parametrized linears, Dynamo guards fail on
        # type(self) for each unique subclass (18/8 recompiles → eager, pcc=-1.0).
        # Remove parametrize, store LoRA A/B as buffers, install a single shared forward.
        _fix_lora_forward_dynamo_guards(model)

        # The text tower's config has dtype=float32, so the nested _from_config
        # overrides the outer bfloat16 context; the text parameters end up float32
        # while the vision parameters are bfloat16.  Cast the whole model to unify.
        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Jina CLIP v2 model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size (default 1).

        Returns:
            dict: Input tensors containing pixel values, input IDs, and attention masks.
        """
        if self.processor is None:
            self._load_processor()

        # Load image from HuggingFace dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

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
        """Post-process Jina CLIP v2 model outputs to extract similarity scores.

        Args:
            outputs: Raw model output tuple.
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
            fwd_output: Output from the model's forward pass (tuple)

        Returns:
            torch.Tensor: Concatenated flattened outputs for backward pass
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
