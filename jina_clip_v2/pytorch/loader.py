# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Jina CLIP v2 model loader implementation for image-text similarity.
"""
import math
import torch
import torch.nn as nn
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


def _recompute_xlm_roberta_rope_buffers(model):
    """Recompute XLM-RoBERTa RotaryEmbedding.inv_freq non-persistent buffers after meta init.

    Under transformers 5.x, meta device init leaves inv_freq uninitialized after
    weight loading because persistent=False buffers are absent from the checkpoint.
    Recompute on CPU using the module's own _compute_inv_freq method.
    """
    for module in model.modules():
        if type(module).__name__ != "RotaryEmbedding":
            continue
        if not hasattr(module, "_compute_inv_freq"):
            continue
        new_inv_freq = module._compute_inv_freq(device="cpu")
        module.register_buffer("inv_freq", new_inv_freq, persistent=False)


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


def _merge_lora_for_default_task(model):
    """Pre-merge LoRA delta for the default task into base weights.

    mha.py and mlp.py dispatch LoRA via adapter_mask, using nonzero() for
    dynamic batch routing and in-place scatter to write back results.  Both
    operations produce dynamic tensor shapes that TT's static-shape compiler
    cannot handle, yielding pcc=-1.0 text embeddings.

    Fix: for inference we always use the default task (task_id=0, 'retrieval.
    query').  Pre-merge that single delta into each LoRA-parametrized weight at
    load time so every module becomes a plain nn.Linear / nn.Embedding.  Then
    set _default_loraid=None on the text tower so hf_model.py never constructs
    an adapter_mask, keeping the entire text encoder on the static code path.
    """
    text_model = getattr(model, "text_model", None)
    default_loraid = getattr(text_model, "_default_loraid", None)
    if default_loraid is None:
        return

    for module in model.modules():
        if not (
            hasattr(module, "parametrizations")
            and hasattr(module.parametrizations, "weight")
            and len(module.parametrizations.weight) > 0
        ):
            continue
        lp = module.parametrizations.weight[0]
        if type(lp).__name__ != "LoRAParametrization":
            continue

        is_linear = isinstance(module, nn.Linear)
        is_embedding = isinstance(module, nn.Embedding)
        if not (is_linear or is_embedding):
            continue

        # Capture LoRA state before removing parametrize.
        # Linear:    lora_A = [num_tasks, rank, fan_in],  lora_B = [num_tasks, fan_out, rank]
        # Embedding: lora_A = [num_tasks, fan_in, rank],  lora_B = [num_tasks, rank, fan_out]
        lora_A = lp.lora_A.detach().clone()
        lora_B = lp.lora_B.detach().clone()
        scaling = float(lp.scaling)

        # Remove parametrize — materialises the base weight (LoRAParametrization.forward
        # is the identity, so the materialised weight equals the original base weight).
        _parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)

        # Compute the LoRA delta for the default task and merge into the base weight.
        if is_linear:
            # Linear:    delta = lora_B[task] @ lora_A[task]  →  [fan_out, fan_in]
            delta_w = torch.mm(lora_B[default_loraid], lora_A[default_loraid]) * scaling
        else:
            # Embedding: delta = lora_A[task] @ lora_B[task]  →  [fan_in, fan_out]
            delta_w = torch.mm(lora_A[default_loraid], lora_B[default_loraid]) * scaling
        module.weight = nn.Parameter(module.weight.data + delta_w)

        # Remove any per-instance forward override left by modeling_lora.py so
        # the module falls back to the plain nn.Linear / nn.Embedding forward.
        module.__dict__.pop("forward", None)

    # Prevent hf_model.py from constructing adapter_mask (and triggering the
    # nonzero / in-place-scatter path in mha.py / mlp.py).
    if text_model is not None:
        text_model._default_loraid = None


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

        # RotaryEmbedding.inv_freq in the XLM-RoBERTa text encoder is likewise a
        # non-persistent buffer; recompute it on CPU using the module's own helper.
        _recompute_xlm_roberta_rope_buffers(model)

        # forward_features re-wraps self.rope.forward with a new partial each eval
        # call, causing Dynamo to recompile 18 times (limit=8) and fall back to eager.
        # Replace each EVAVisionTransformer's forward_features with a stable version.
        _fix_eva_rope_forward_accumulation(model)

        # The adapter_mask path in mha.py / mlp.py uses nonzero() (dynamic shapes) and
        # in-place scatter — both incompatible with TT static-shape compilation (pcc=-1.0).
        # Pre-merge the default-task LoRA delta into each base weight at load time so
        # every LoRA module becomes a plain nn.Linear / nn.Embedding and adapter_mask
        # is never constructed during the forward pass.
        _merge_lora_for_default_task(model)

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
