# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Jina CLIP v2 model loader implementation for image-text similarity.
"""
import math
import torch
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
