# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ARC1-mini causal language modeling loader.

meissosisai/arc1-mini is a Phi-4-mini-instruct (Phi3ForCausalLM, model_type=phi3)
fine-tune published only as a GGUF file plus a LoRA adapter (no base safetensors
weights). The merged full-precision weights live in the GGUF, so the model is
loaded via transformers' GGUF support (from_pretrained with gguf_file=...), which
dequantizes the GGUF to fp32 on load (~3.8B params). The config is read from the
GGUF; the tokenizer is read from the repo's tokenizer.json (the GGUF-embedded
tokenizer fails to convert for this model).
"""
import os
import tempfile
from typing import Optional

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    ARC1_MINI = "arc1_mini"


class ModelLoader(ForgeModel):
    """ARC1-mini (Phi-4-mini-instruct fine-tune) GGUF loader."""

    _VARIANTS = {
        ModelVariant.ARC1_MINI: LLMModelConfig(
            pretrained_model_name="meissosisai/arc1-mini",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ARC1_MINI

    # GGUF file (within the HF repo) that holds the full merged weights.
    _GGUF_VARIANTS = {
        ModelVariant.ARC1_MINI: "arc1-mini.gguf",
    }

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Unused for GGUF models (kept for interface compatibility).
        """
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers
        self._gguf_local_dir = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="arc1-mini",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _gguf_file(self) -> str:
        return self._GGUF_VARIANTS[self._variant]

    def _local_gguf_dir(self) -> str:
        """Download the GGUF into an isolated directory and return its path.

        The HF repo also contains a ``adapter_config.json`` (LoRA adapter), which
        makes ``AutoModelForCausalLM.from_pretrained(repo, ...)`` auto-detect a PEFT
        adapter and redirect to the (quantized) base model instead of reading the
        GGUF. Loading the model from a directory that holds only the GGUF avoids
        that redirection. Idempotent: a second call reuses the downloaded file.
        """
        if self._gguf_local_dir is None:
            local_dir = os.path.join(
                tempfile.gettempdir(), f"arc1_mini_gguf_{self._variant.value}"
            )
            os.makedirs(local_dir, exist_ok=True)
            hf_hub_download(
                self._variant_config.pretrained_model_name,
                self._gguf_file(),
                local_dir=local_dir,
            )
            self._gguf_local_dir = local_dir
        return self._gguf_local_dir

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            # Load the tokenizer from the repo (uses the repo's tokenizer.json);
            # the GGUF-embedded tokenizer fails to convert for this model.
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
                gguf_file=self._gguf_file(),
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the ARC1-mini model from its GGUF file.

        transformers dequantizes the GGUF to fp32; the tester applies any
        dtype_override afterwards via ``model.to(...)``.
        """
        self._ensure_tokenizer()

        model_kwargs = {
            "gguf_file": self._gguf_file(),
            "use_cache": False,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._local_gguf_dir(), **model_kwargs
        )
        model.eval()
        if dtype_override is not None:
            model = model.to(dtype_override)

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, prompt: Optional[str] = None):
        self._ensure_tokenizer()
        input_prompt = prompt or "The capital of France is"
        inputs = self.tokenizer(
            input_prompt,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        if dtype_override is not None:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attn_mask = cast_input_to_type(attn_mask, dtype_override)

        return [input_ids, attn_mask]

    def decode_output(self, outputs, dtype_override=None):
        """Decode model logits into the next-token text."""
        self._ensure_tokenizer()
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_token_id = logits[:, -1, :].argmax(dim=-1)
        return self.tokenizer.decode(next_token_id)

    def load_config(self):
        """Load and return the model configuration (read from the GGUF)."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file(),
        )
        return self.config
