# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Molmo2-8B vision-tower loader.

Molmo2 (`Molmo2ForConditionalGeneration`, allenai/Molmo2-8B) is a multimodal
image-text-to-text VLM: a SigLIP-style ViT image encoder + a pooling/projection
adapter feeding a Qwen3-8B-style text decoder. The model ships as custom code
(`trust_remote_code=True`).

This loader brings up the **vision tower** (`Molmo2VisionTransformer`,
i.e. `model.vision_backbone.image_vit`) as a single clean forward pass. The full
`Molmo2VisionBackbone.forward` and the top-level VLM forward contain
data-dependent indexing / `.item()` graph breaks (build_batched_images, boolean
gather), so the inner ViT is the component validated on device — the text
decoder is brought up separately under ``molmo2/causal_lm``.
"""

import torch
from typing import Optional

from transformers import AutoModelForImageTextToText

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


def _patch_default_rope():
    """Restore ``ROPE_INIT_FUNCTIONS['default']`` for Molmo2's custom modeling code.

    allenai/Molmo2-8B (custom code, transformers_version 5.5.1) constructs its
    rotary embedding via ``ROPE_INIT_FUNCTIONS['default']``, but transformers
    >=5.5 dropped the ``'default'`` key (standard RoPE is now expressed as
    ``'proportional'`` with ``factor=1.0`` / ``partial_rotary_factor=1.0`` — the
    inverse-frequency formula is identical, ``1/base**(arange(0,d,2)/d)``).
    Registering the proportional implementation under ``'default'`` lets the
    model build without forcing a transformers downgrade (which would risk
    pulling the torch / torch-xla stack out of sync).
    """
    from transformers.modeling_rope_utils import (
        ROPE_INIT_FUNCTIONS,
        _compute_proportional_rope_parameters,
    )

    ROPE_INIT_FUNCTIONS.setdefault("default", _compute_proportional_rope_parameters)


class _VisionTowerWrapper(torch.nn.Module):
    """Thin wrapper exposing the ViT image encoder as a single-tensor forward.

    ``Molmo2VisionTransformer`` returns the per-layer list of hidden states; we
    return the last layer's hidden state so the device comparison sees one
    clean tensor instead of a 25-element pytree.
    """

    def __init__(self, image_vit: torch.nn.Module):
        super().__init__()
        self.image_vit = image_vit

    def forward(self, pixel_patches: torch.Tensor) -> torch.Tensor:
        hidden_states = self.image_vit(pixel_patches)
        return hidden_states[-1]


class ModelVariant(StrEnum):
    """Available Molmo2 vision-tower variants."""

    MOLMO2_8B = "8b"


class ModelLoader(ForgeModel):
    """Loader for the Molmo2-8B vision tower (SigLIP-style ViT image encoder)."""

    _VARIANTS = {
        ModelVariant.MOLMO2_8B: LLMModelConfig(
            pretrained_model_name="allenai/Molmo2-8B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOLMO2_8B

    # Pin the custom-code / weight revision so reruns are reproducible.
    _REVISION = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"

    # ViT patch geometry: 378x378 image / patch 14 -> 27x27 = 729 patches;
    # each patch is 14*14*3 = 588 raw pixel values (the patch_embedding input).
    _NUM_PATCH = 729
    _N_PIXELS = 588

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="molmo2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load the Molmo2 vision tower (image_vit) wrapped as a single-tensor module.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The vision-tower wrapper.
        """
        _patch_default_rope()
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        full_model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            revision=self._REVISION,
            **model_kwargs,
        )

        image_vit = full_model.model.vision_backbone.image_vit
        wrapper = _VisionTowerWrapper(image_vit).eval()
        if dtype_override is not None:
            wrapper = wrapper.to(dtype_override)
        return wrapper

    # Geometry of one image: 378x378 RGB, patch 14 -> 27x27 grid of patches.
    _IMG = 378
    _PATCH = 14
    _GRID = 27  # 378 / 14

    def _build_patchified_image(self, batch_size: int) -> torch.Tensor:
        """Build a deterministic, structured, SigLIP-normalized patchified image.

        The vision tower consumes *patchified normalized pixel values*
        (``[batch, 729, 588]``), not raw noise. We synthesize a smooth structured
        image (color gradients + a block), apply SigLIP normalization
        (``x/127.5 - 1`` → ~[-1, 1]), and patchify it into 27x27 patches of
        14*14*3 = 588 values — the same statistics the encoder sees for a real
        image, which avoids the artificially low PCC that pure gaussian noise
        produces under bf16. The Molmo2 image processor itself is bypassed here
        because its current build is incompatible with the installed transformers
        (rejects the ``image_use_col_tokens`` kwarg); this synthesizes an
        equivalent normalized patch tensor directly.
        """
        g, p = self._GRID, self._PATCH
        ramp_y = torch.linspace(0, 255, self._IMG).view(self._IMG, 1, 1)
        ramp_x = torch.linspace(0, 255, self._IMG).view(1, self._IMG, 1)
        img = torch.zeros(self._IMG, self._IMG, 3)
        img[..., 0:1] = ramp_y
        img[..., 1:2] = ramp_x
        img[..., 2:3] = (ramp_y + ramp_x) / 2
        img[100:200, 100:200, :] = torch.tensor([200.0, 50.0, 50.0])
        # SigLIP-style normalization to ~[-1, 1].
        img = img / 127.5 - 1.0
        # Patchify: (G*p, G*p, 3) -> (G, G, p, p, 3) -> (G*G, p*p*3).
        patches = (
            img.reshape(g, p, g, p, 3)
            .permute(0, 2, 1, 3, 4)
            .reshape(g * g, p * p * 3)
        )
        return patches.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Return sample pixel-patch inputs for the vision tower.

        Args:
            dtype_override: Optional torch.dtype for the input tensor.
            batch_size: Batch size for the inputs.

        Returns:
            dict: ``{"pixel_patches": Tensor[batch, 729, 588]}``.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        pixel_patches = self._build_patchified_image(batch_size).to(dtype)
        return {"pixel_patches": pixel_patches}
