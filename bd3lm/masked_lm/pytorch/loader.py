# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BD3LM (Block Discrete Denoising Diffusion Language Model) loader
implementation for masked language modeling.
"""
import torch
from typing import Optional

from transformers import AutoModelForMaskedLM, AutoTokenizer

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


class ModelVariant(StrEnum):
    """Available BD3LM model variants."""

    BD3LM_OWT_BLOCK_SIZE_4 = "owt-block_size4"


class ModelLoader(ForgeModel):
    """BD3LM model loader implementation for masked language modeling."""

    # BD3LM uses the gpt2 tokenizer (vocab_size 50258 = gpt2 50257 + 1 mask token).
    _TOKENIZER_NAME = "gpt2"

    _VARIANTS = {
        ModelVariant.BD3LM_OWT_BLOCK_SIZE_4: LLMModelConfig(
            pretrained_model_name="kuleshov-group/bd3lm-owt-block_size4",
            # BD3LM uses cross attention between noised and target blocks, which
            # expects an input length of 2 * model_length (2 * 1024).
            max_length=2048,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BD3LM_OWT_BLOCK_SIZE_4

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BD3LM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self._TOKENIZER_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        # BD3LM's flex attention backend requires CUDA and recent PyTorch;
        # override to sdpa so the loader works on CPU/XLA runtimes.
        # BD3LM's timestep embedding explicitly casts to float32 internally
        # (modeling_bd3lm.py: `t[:, None].float()`), so the model must be
        # loaded in float32 to avoid dtype mismatch in the sigma_map MLP.
        # low_cpu_mem_usage=False ensures gen_mask() runs on real tensors —
        # self.mask is a plain attribute (not a registered buffer), so it
        # stays as a meta tensor when low_cpu_mem_usage=True is used.
        model_kwargs = {"attn_backend": "sdpa", "low_cpu_mem_usage": False}
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        # Regenerate the attention mask on CPU after loading to ensure it is
        # a real tensor (not a meta tensor from low_cpu_mem_usage init).
        if hasattr(model, "backbone") and hasattr(model.backbone, "gen_mask"):
            model.backbone.gen_mask(
                model.backbone.n, model.backbone.block_size, attn_backend="sdpa"
            )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            "The quick brown fox jumps over the lazy dog.",
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        timesteps = torch.zeros(1)

        return {
            "input_ids": inputs["input_ids"],
            "timesteps": timesteps,
        }

    def decode_output(self, outputs):
        if self.tokenizer is None:
            self._load_tokenizer()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        predicted_ids = logits.argmax(dim=-1)
        decoded = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        return decoded
