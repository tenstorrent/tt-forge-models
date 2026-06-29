# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step 1.5 DiT denoiser loader (text-to-music diffusion transformer).

The key compute component of the ACE-Step 1.5 music-generation pipeline. The
full ``AceStepConditionGenerationModel`` (custom_code) wraps a condition encoder
(text/lyric/timbre), an FSQ tokenizer/detokenizer and the per-denoising-step DiT
transformer (``AceStepDiTModel``, reachable via ``model.decoder``). This loader
isolates that DiT backbone — the part that runs once per flow-matching step — so
it can be compiled and run as a single forward pass on device.
"""
import torch
from typing import Optional

from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module

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

# Pinned snapshot of the custom_code repo this loader was brought up against.
_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"


class ModelVariant(StrEnum):
    """Available ACE-Step 1.5 denoiser variants."""

    TURBO = "turbo"


class ModelLoader(ForgeModel):
    """ACE-Step 1.5 DiT denoiser (AceStepDiTModel) loader."""

    _VARIANTS = {
        ModelVariant.TURBO: ModelConfig(
            pretrained_model_name="ACE-Step/Ace-Step1.5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TURBO

    # Native representative clip: 10 s @ 25 Hz latent rate = 250 frames (the
    # base config the model's own reference test uses).
    _LATENT_FRAMES = 250  # T, even -> divisible by patch_size=2 (no dynamic pad)
    _COND_SEQ_LEN = 200  # packed text+lyric+timbre conditioning length
    _LATENT_DIM = 64  # acoustic latent channels (noisy latent)
    _CONTEXT_DIM = 128  # src_latents(64) + chunk_masks(64)
    _COND_DIM = 2048  # condition encoder hidden size (== model hidden_size)

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._dit = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="acestep_denoiser",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_dit(self, dtype_override=None):
        """Build the DiT backbone (AceStepDiTModel) and load only ``decoder.*``.

        The custom modeling/config .py files live in the ``acestep-v15-turbo``
        subfolder, but transformers resolves auto_map modules from the repo root,
        so we materialize the subfolder locally and load from that path. We
        instantiate the DiT directly (rather than the full
        AceStepConditionGenerationModel) to skip the FSQ audio tokenizer — it is
        unused for a denoising step and its __init__ trips on meta-tensor init.
        """
        if self._dit is None:
            root = snapshot_download(
                self._variant_config.pretrained_model_name,
                revision=_REVISION,
                allow_patterns=["acestep-v15-turbo/*"],
            )
            local_dir = f"{root}/acestep-v15-turbo"
            dtype = dtype_override if dtype_override is not None else torch.bfloat16

            config_cls = get_class_from_dynamic_module(
                "configuration_acestep_v15.AceStepConfig", local_dir
            )
            dit_cls = get_class_from_dynamic_module(
                "modeling_acestep_v15_turbo.AceStepDiTModel", local_dir
            )
            config = config_cls.from_pretrained(local_dir)
            dit = dit_cls(config)

            state = load_file(f"{local_dir}/model.safetensors")
            decoder_state = {
                k[len("decoder.") :]: v
                for k, v in state.items()
                if k.startswith("decoder.")
            }
            missing, unexpected = dit.load_state_dict(decoder_state, strict=False)
            if missing:
                raise RuntimeError(f"Missing decoder weights: {missing[:8]} ...")
            self._dit = dit.to(dtype).eval()
        return self._dit

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the DiT denoiser (AceStepDiTModel) backbone."""
        return self._load_dit(dtype_override=dtype_override)

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        """Build a single flow-matching denoising-step's inputs for the DiT.

        Shapes follow the model's reference test (10 s @ 25 Hz = 250 frames).
        The DiT receives the noisy latent, the packed conditioning embeddings
        and the source-latent context already produced upstream by the condition
        encoder; here we synthesize representative tensors of the correct shapes
        and dtype (the same on CPU and device, so PCC is a like-for-like check).
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        T = self._LATENT_FRAMES
        S = self._COND_SEQ_LEN
        B = batch_size

        g = torch.Generator().manual_seed(0)

        hidden_states = torch.randn(B, T, self._LATENT_DIM, generator=g).to(dtype)
        context_latents = torch.randn(B, T, self._CONTEXT_DIM, generator=g).to(dtype)
        encoder_hidden_states = torch.randn(B, S, self._COND_DIM, generator=g).to(dtype)
        encoder_attention_mask = torch.ones(B, S, dtype=dtype)
        attention_mask = torch.ones(B, T, dtype=dtype)
        # Flow-matching timesteps (t and reference r), one scalar per batch elem.
        timestep = torch.full((B,), 1.0, dtype=dtype)
        timestep_r = torch.full((B,), 0.0, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "timestep_r": timestep_r,
            "attention_mask": attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "context_latents": context_latents,
            "use_cache": False,
        }
