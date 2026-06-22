# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HunyuanImage-3.0 loader.

tencent/HunyuanImage-3.0 is a unified autoregressive multimodal Mixture-of-Experts
model that generates images by next-token / diffusion prediction inside a single
decoder-only transformer (``HunyuanImage3ForCausalMM``, custom code via
``trust_remote_code=True``). It is NOT a diffusers-style pipeline with separable
``text_encoder`` / ``transformer`` / ``vae`` checkpoints: the text encoder is the
transformer's own token embeddings, and the VAE (``AutoencoderKLConv3D``) and
SigLIP2 vision tower (image *input* understanding only) are submodules constructed
inside the model from sub-configs.

Footprint (from ``model.safetensors.index.json``): 83.0B parameters / 168.5 GB
(transformer + heads in bf16, the VAE in fp32). The 32 decoder layers each carry
64 routed experts (top-8) + 1 shared
expert, hence the large total vs. the ~13B activated per token. The model is far
too large for a single Wormhole/Blackhole chip and is intended to be sharded across
a multi-chip device (e.g. galaxy-bh: 32x Blackhole, 1 TB DRAM => ~5.3 GB/chip of
weights). Use the ``sharding-model-analysis`` skill to generate the mesh / partition
specs for the decoder backbone before running on device.

Image generation in the source pipeline runs a ``FlowMatchDiscreteScheduler``
(``diff_infer_steps=50``) denoising loop in host Python that calls this transformer
once per step, then decodes the resulting latents with the VAE. For a single
forward-pass bringup this loader exercises the transformer's text-prefill path
(``mode="gen_text"``), which is the compute-dominant component and the sharding
target.
"""

from typing import Optional

import torch

from ...base import ForgeModel
from ...config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

_REPO_ID = "tencent/HunyuanImage-3.0"

# Default device dtype for the checkpoint (config.json: torch_dtype == bfloat16).
DTYPE = torch.bfloat16


def _patch_cuda_compat():
    """Neutralize CUDA-only calls in HunyuanMoE on non-CUDA backends.

    ``HunyuanMoE.forward`` hard-codes two CUDA-only operations:
    ``torch.cuda.set_device(hidden_states.device.index)`` and
    ``with nvtx.range("MoE")``. On any non-CUDA backend (CPU here, and the XLA /
    Tenstorrent device path) the index is ``None`` / CUDA is unavailable and these
    raise. Make them no-ops when CUDA is unavailable while preserving the original
    behaviour on a real CUDA device. This is a known portability issue in the
    upstream custom code (worth fixing there).
    """
    if torch.cuda.is_available():
        return

    _orig_set_device = torch.cuda.set_device

    def _safe_set_device(device):
        try:
            idx = getattr(device, "index", device)
            if idx is None:
                return
            return _orig_set_device(device)
        except Exception:
            return

    torch.cuda.set_device = _safe_set_device

    import contextlib

    @contextlib.contextmanager
    def _noop_range(*args, **kwargs):
        yield

    torch.cuda.nvtx.range = _noop_range
    torch.cuda.nvtx.range_push = lambda *a, **k: None
    torch.cuda.nvtx.range_pop = lambda *a, **k: None


class ModelVariant(StrEnum):
    """Available HunyuanImage-3.0 variants."""

    BASE = "hunyuan_image_3_moe"


class ModelLoader(ForgeModel):
    """Loads the unified HunyuanImage-3.0 multimodal MoE transformer."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(pretrained_model_name=_REPO_ID),
    }
    DEFAULT_VARIANT = ModelVariant.BASE

    # A short prompt is enough to exercise the text-prefill forward path.
    sample_text = "A cinematic photo of a red fox in a snowy forest at sunrise."
    max_length = 32
    # RoPE base and per-head dim taken from config.json (rope_theta, attention_head_dim).
    rope_theta = 10000.0
    head_dim = 128

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="HunyuanImage3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load and cache the tokenizer (custom code)."""
        from transformers import AutoTokenizer

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
                trust_remote_code=True,
            )
        return self._tokenizer

    def load_model(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load the HunyuanImage3ForCausalMM model.

        Args:
            dtype_override: optional torch dtype; defaults to bfloat16.

        Returns:
            torch.nn.Module: the unified multimodal MoE transformer in eval mode.

        Note:
            83.0B parameters / 168.5 GB. ``attn_implementation="eager"``
            avoids the optional flash-attention dependency; the model exposes an
            SDPA attention path as well. The model must be sharded across a
            multi-chip device to fit.
        """
        from transformers import AutoModelForCausalLM

        _patch_cuda_compat()
        dtype = dtype_override if dtype_override is not None else DTYPE
        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        model.eval()
        return model

    def _build_text_rope(self, seq_len: int, dtype: torch.dtype):
        """Build the 2D-RoPE (cos, sin) for a pure-text sequence.

        Mirrors the model's ``build_2d_rope`` with ``image_infos=None``: for text
        the x and y position grids are both ``arange(seq_len)``. Shapes returned are
        ``[1, seq_len, head_dim]`` to match ``custom_pos_emb`` expected by forward().
        """
        n_elem = self.head_dim
        theta = 1.0 / (
            self.rope_theta ** (torch.arange(0, n_elem, 2).float() / n_elem)
        )
        theta = theta.reshape(1, n_elem // 4, 2)  # [1, head_dim/4, 2]
        pos = torch.arange(0, seq_len)
        all_pos = torch.stack((pos, pos), dim=1).unsqueeze(1).float()  # [seq_len, 1, 2]
        idx_theta = (all_pos * theta).reshape(seq_len, n_elem // 2).repeat(1, 2)
        cos = torch.cos(idx_theta).unsqueeze(0).to(dtype)  # [1, seq_len, head_dim]
        sin = torch.sin(idx_theta).unsqueeze(0).to(dtype)
        return cos, sin

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Build sample inputs for the transformer's text-prefill forward path.

        The ``gen_text`` forward path requires the rotary position embeddings to be
        precomputed and passed in via ``custom_pos_emb`` (the model itself never
        builds them in forward()), together with ``position_ids``.

        Returns:
            dict: forward() kwargs selecting the text-prefill path:
              - ``input_ids``      [1, max_length] int64
              - ``position_ids``   [1, max_length] int64 (``arange``)
              - ``custom_pos_emb`` tuple(cos, sin), each [1, max_length, head_dim]
              - ``mode="gen_text"`` and ``use_cache=False``

        Note:
            ``attention_mask`` is intentionally omitted (left as None), matching the
            model's own ``prepare_model_inputs``: the SDPA attention path feeds the
            mask straight to ``scaled_dot_product_attention`` and a padding mask is
            not used during the prefill step.
        """
        dtype = dtype_override if dtype_override is not None else DTYPE
        tokenizer = self._load_tokenizer()
        encoded = tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        seq_len = encoded["input_ids"].shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)
        cos, sin = self._build_text_rope(seq_len, dtype)
        return {
            "input_ids": encoded["input_ids"],
            "position_ids": position_ids,
            "custom_pos_emb": (cos, sin),
            "mode": "gen_text",
            "use_cache": False,
        }
