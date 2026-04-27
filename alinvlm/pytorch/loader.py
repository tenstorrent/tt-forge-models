# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AlinVLM model loader implementation for image to text tasks.
"""

import sys
import types

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionModel,
    Qwen3VLModel,
    Qwen3VLTextModel,
)
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

# ---------------------------------------------------------------------------
# sys.modules stabilisation for Dynamo tracing
#
# DynamicLoader.import_model_loader registers model loaders under
# 'tt-forge-models.*' keys.  When Dynamo traces a compiled function it calls
# import_source(fn.__module__) for every function it encounters.  If
# fn.__module__ = 'tt_forge_models.minicpmv_2_6.pytorch.loader' is absent from
# sys.modules, Python re-imports it from disk, re-running module-level code
# (including nn.Module.__getattr__ = patched_getattr) mid-trace.  This changes
# the code object at a call site the speculation log has already recorded,
# causing SpeculationLogDivergence on the very next forward call.
#
# Fix: before the first torch.compile trace, mirror every 'tt-forge-models.*'
# entry into 'tt_forge_models.*' and stub any missing intermediate packages.
# Once every module source is already in sys.modules, Dynamo's import_source
# becomes a no-op and nn.Module.__getattr__ stays stable throughout tracing.
# ---------------------------------------------------------------------------


def _stabilize_forge_models_sys_modules() -> None:
    """Copy 'tt-forge-models.*' sys.modules entries into 'tt_forge_models.*'.

    Call this once before torch.compile is first invoked on a Qwen3VL model.
    """
    # Mirror loader modules from dashes-namespace to underscores-namespace.
    for key in list(sys.modules.keys()):
        if key.startswith("tt-forge-models."):
            under_key = "tt_forge_models." + key[len("tt-forge-models."):]
            if under_key not in sys.modules:
                sys.modules[under_key] = sys.modules[key]

    # Add synthetic namespace stubs for any missing intermediate packages so
    # Python does not attempt to import them (and run __init__.py) on demand.
    for key in list(sys.modules.keys()):
        if not key.startswith("tt_forge_models."):
            continue
        parts = key.split(".")
        for i in range(2, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in sys.modules:
                stub = types.ModuleType(parent)
                stub.__path__ = []  # mark as package
                stub.__package__ = parent
                sys.modules[parent] = stub


# ---------------------------------------------------------------------------
# Monkey-patches for Qwen3VL on TT hardware
#
# Two root causes prevent compilation:
#
#  1. Qwen3VLVisionModel uses nn.Conv3d (patch embedding) and sequences of up
#     to 11,008 tokens that overflow TT L1 (max 1.5 MB per core).  Fix: run
#     the visual encoder eagerly on CPU via torch.compiler.disable.
#
#  2. Several methods call .tolist() on tensors that may land on TT device
#     (image_grid_thw, input_ids).  Fix: move them to CPU before calling
#     the original methods; get_rope_index is also disabled so its control-
#     flow runs eagerly on CPU.
# ---------------------------------------------------------------------------

_orig_visual_forward = Qwen3VLVisionModel.forward


@torch.compiler.disable(recursive=True)
def _patched_visual_forward(self, hidden_states, grid_thw, **kwargs):
    """Run visual encoder eagerly on CPU to avoid TT L1 overflow and Conv3d."""
    param = next(self.parameters(), None)
    if param is not None and param.device.type != "cpu":
        self.cpu()
    if hidden_states.device.type != "cpu":
        hidden_states = hidden_states.cpu()
    if grid_thw.device.type != "cpu":
        grid_thw = grid_thw.cpu()
    return _orig_visual_forward(self, hidden_states, grid_thw, **kwargs)


Qwen3VLVisionModel.forward = _patched_visual_forward

_orig_get_image_features = Qwen3VLModel.get_image_features


@torch.compiler.disable(recursive=True)
def _get_image_features_eager(model_self, pixel_values, image_grid_thw, kwargs):
    """Run get_image_features eagerly on CPU to avoid .tolist() graph breaks.

    image_grid_thw.prod(-1).tolist() inside the original function causes the TT
    backend compiler to emit 15+ recompilations (each aten._local_scalar_dense
    failure adds a new graph break), which fills and overflows the XLA LRU
    computation cache.  Running the whole call disabled prevents the cascade.
    """
    return _orig_get_image_features(model_self, pixel_values, image_grid_thw, **kwargs)


def _patched_get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
    """Delegate to a disabled helper to keep .tolist() out of the compiled graph."""
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.cpu()
    return _get_image_features_eager(self, pixel_values, image_grid_thw, kwargs)


Qwen3VLModel.get_image_features = _patched_get_image_features

_orig_get_rope_index = Qwen3VLModel.get_rope_index


@torch.compiler.disable(recursive=True)
def _patched_get_rope_index(
    self,
    input_ids=None,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    **kwargs,
):
    """Run rope-index computation eagerly on CPU (uses .tolist() for control flow).

    The original method builds 3D position_ids via .tolist() control flow that
    cannot run on TT device tensors.  We detect the caller's device from the
    input tensors (which are on xla:0 during compilation), run the computation
    eagerly on CPU, then move outputs back to that device so the RoPE matmul
    (inv_freq on xla:0, position_ids on xla:0) sees consistent devices.
    """
    # Detect target device from input tensors before moving to CPU.
    # Input tensors are on the TT device (xla:0) during compilation runs.
    target_device = None
    for t in (input_ids, image_grid_thw, video_grid_thw, attention_mask):
        if isinstance(t, torch.Tensor) and t.device.type != "cpu":
            target_device = t.device
            break

    if input_ids is not None and input_ids.device.type != "cpu":
        input_ids = input_ids.cpu()
    if image_grid_thw is not None and image_grid_thw.device.type != "cpu":
        image_grid_thw = image_grid_thw.cpu()
    if video_grid_thw is not None and video_grid_thw.device.type != "cpu":
        video_grid_thw = video_grid_thw.cpu()
    if attention_mask is not None and attention_mask.device.type != "cpu":
        attention_mask = attention_mask.cpu()
    result = _orig_get_rope_index(
        self, input_ids, image_grid_thw, video_grid_thw, attention_mask, **kwargs
    )
    # Move outputs back to caller's device so RoPE matmul doesn't see mixed devices.
    if target_device is not None:
        def _to(t):
            return t.to(target_device) if isinstance(t, torch.Tensor) else t
        if isinstance(result, (tuple, list)):
            result = type(result)(_to(r) for r in result)
        else:
            result = _to(result)
    return result


Qwen3VLModel.get_rope_index = _patched_get_rope_index

_orig_deepstack_process = Qwen3VLTextModel._deepstack_process


@torch.compiler.disable(recursive=True)
def _patched_deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
    """Run deepstack process eagerly to avoid LRU cache overflow.

    hidden_states[visual_pos_masks, :] produces tensors with data-dependent size
    (visual_pos_masks.sum()), so each unique visual token count generates a new
    XLA computation.  Called once per decoder layer inside the decoder loop, this
    floods the XLA LRU computation cache and evicts earlier entries.  Running
    eagerly avoids the flood entirely.
    """
    return _orig_deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds)


Qwen3VLTextModel._deepstack_process = _patched_deepstack_process


class ModelVariant(StrEnum):
    """Available AlinVLM model variants for image to text."""

    ALINVLM_V1_3 = "v1_3"


class ModelLoader(ForgeModel):
    """AlinVLM model loader implementation for image to text tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.ALINVLM_V1_3: LLMModelConfig(
            pretrained_model_name="huiwon/alinvlm_v1_3",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.ALINVLM_V1_3

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="AlinVLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the AlinVLM model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The AlinVLM model instance for image to text.
        """
        # Stabilize sys.modules so Dynamo's import_source does not re-import
        # loaders that patched nn.Module.__getattr__ during collection.
        _stabilize_forge_models_sys_modules()

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"dtype": "auto", "device_map": "auto"}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # use_fast=False: Qwen2VLImageProcessorFast ignores max_pixels; the
        # slow processor respects it and keeps the patch count manageable.
        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name, use_fast=False
        )
        # Cap image to 28x28 → 4 patches → ~17 total tokens (image + text).
        # At 512x512 (~988 patches) the language model sees ~1001 tokens,
        # using 1001*4096*2 ≈ 8.2 MB of L1 CBs, exceeding the 1.5 MB max.
        # 28x28 (the smallest valid grid: 2×2 merge blocks of 14×14 patches)
        # brings seq_len down to ~17, using ~140 KB — well within L1.
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.max_pixels = 28 * 28
            if hasattr(self.processor.image_processor, "min_pixels"):
                self.processor.image_processor.min_pixels = 28 * 28

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the AlinVLM model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs
