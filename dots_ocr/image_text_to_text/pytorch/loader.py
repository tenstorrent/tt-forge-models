# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""dots.ocr model loader implementation for document OCR (image-text-to-text).

dots.ocr (``DotsOCRForCausalLM``) is a multimodal document-parsing model from
rednote-hilab. It pairs a ``dots_vit`` vision tower with a Qwen2-style causal-LM
text decoder. The model ships as ``trust_remote_code`` custom code; the vision
tower defaults to ``flash_attention_2`` and automatically falls back to an eager
attention implementation when ``flash_attn`` is not installed (the CPU/TT case).

The default ``load_inputs`` produces a **text-only** prompt so the discovered
device test exercises the Qwen2 decoder component (the vision-tower patch-embed
Conv2d currently hits a known ``ttir.convolution`` legalization gap on device,
the same gap that blocks qwen_2_5_vl — tt-xla issue 1662). ``load_vision_inputs``
is provided separately for the vision-tower op pre-check.
"""
import torch
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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

# Pin the remote-code revision so the trust_remote_code modules are reproducible.
_REVISION = "c0111ce6bc07803dbc267932ffef0ae3a51dc951"


class ModelVariant(StrEnum):
    """Available dots.ocr model variants."""

    DOTS_OCR_1_7B = "1.7B"


class ModelLoader(ForgeModel):
    """dots.ocr model loader for document OCR (image-text-to-text)."""

    _VARIANTS = {
        ModelVariant.DOTS_OCR_1_7B: LLMModelConfig(
            pretrained_model_name="rednote-hilab/dots.ocr",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DOTS_OCR_1_7B

    # Short OCR-style instruction used to drive the text decoder.
    sample_text = "Extract the text content from the document image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="dots_ocr",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
                revision=_REVISION,
                trust_remote_code=True,
            )
        return self._tokenizer

    def load_model(self, dtype_override=None):
        """Load the full dots.ocr model (vision tower + Qwen2 decoder).

        Args:
            dtype_override: Optional torch.dtype to override the model's default
                (the checkpoint ships in bfloat16).

        Returns:
            torch.nn.Module: DotsOCRForCausalLM instance in eval mode.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True, "revision": _REVISION}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self.config = model.config
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Return text-only inputs that exercise the Qwen2 decoder component.

        Args:
            dtype_override: Unused for integer token inputs; accepted for a
                uniform interface.
            batch_size: Batch size for the inputs.

        Returns:
            dict: ``input_ids`` / ``attention_mask`` tensors.
        """
        tokenizer = self._load_tokenizer()

        # Tokenize to the prompt's natural length (no padding) so the compared
        # logits cover only meaningful token positions — padding positions add
        # numerically degenerate outputs that depress PCC without being part of
        # the real decode path.
        inputs = tokenizer(
            self.sample_text,
            return_tensors="pt",
        )
        inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        if batch_size > 1:
            for key in inputs:
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        return inputs

    def load_vision_inputs(self, dtype_override=None, grid_h=8, grid_w=8):
        """Return synthetic inputs for the dots_vit vision-tower op pre-check.

        Builds a patchified pixel tensor and grid descriptor matching the vision
        tower's expected layout, bypassing the full image processor.

        Args:
            dtype_override: Optional torch.dtype for the pixel tensor.
            grid_h / grid_w: Patch grid dimensions (must be divisible by the
                spatial merge size of 2).

        Returns:
            dict: ``pixel_values`` ``[num_patches, 588]`` and ``grid_thw``.
        """
        if self.config is None:
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name,
                revision=_REVISION,
                trust_remote_code=True,
            )
        vcfg = self.config.vision_config
        patch = vcfg.patch_size if not isinstance(vcfg, dict) else vcfg["patch_size"]
        channels = (
            vcfg.num_channels if not isinstance(vcfg, dict) else vcfg["num_channels"]
        )
        temporal = (
            vcfg.temporal_patch_size
            if not isinstance(vcfg, dict)
            else vcfg["temporal_patch_size"]
        )

        num_patches = grid_h * grid_w
        feat = channels * temporal * patch * patch  # 3 * 1 * 14 * 14 = 588
        dtype = dtype_override or torch.float32
        pixel_values = torch.randn(num_patches, feat, dtype=dtype)
        grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
        return {"pixel_values": pixel_values, "grid_thw": grid_thw}

    def decode_output(self, outputs, inputs=None):
        """Greedy-decode the next-token prediction into a string."""
        tokenizer = self._load_tokenizer()
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_id = int(logits[0, -1].argmax(-1))
        return tokenizer.decode([next_id])
