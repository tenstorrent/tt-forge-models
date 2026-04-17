# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-1.8B-Llamafied model loader implementation for causal language modeling.
"""
import os
import tempfile

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional

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
    """Available Qwen-1.8B-Llamafied model variants for causal language modeling."""

    QWEN_1_8B_LLAMAFIED = "1.8B-Llamafied"


class ModelLoader(ForgeModel):
    """Qwen-1.8B-Llamafied model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_1_8B_LLAMAFIED: LLMModelConfig(
            pretrained_model_name="KnutJaegersberg/Qwen-1_8B-Llamafied",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_1_8B_LLAMAFIED

    sample_text = "The future of artificial intelligence is"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="Qwen-1.8B-Llamafied",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _prepare_patched_tokenizer_dir(pretrained_model_name: str) -> str:
        """Download tokenizer files and strip an invalid trailing entry in merges.txt.

        The published merges.txt for ``KnutJaegersberg/Qwen-1_8B-Llamafied`` ends
        with a truncated single-token line that the tokenizers BPE parser rejects.
        We materialize the tokenizer assets into a local directory and drop any
        merge line that does not have exactly two space-separated tokens.
        """
        from huggingface_hub import snapshot_download

        local_dir = os.path.join(
            tempfile.gettempdir(),
            f"tt-forge-{pretrained_model_name.replace('/', '_')}-tokenizer",
        )
        snapshot_download(
            pretrained_model_name,
            local_dir=local_dir,
            allow_patterns=[
                "vocab.json",
                "merges.txt",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
            ],
        )

        merges_path = os.path.join(local_dir, "merges.txt")
        if os.path.isfile(merges_path):
            with open(merges_path, encoding="utf-8") as f:
                lines = f.read().split("\n")
            cleaned = [lines[0]] if lines else []
            for line in lines[1:]:
                if not line:
                    continue
                parts = line.split(" ")
                if len(parts) == 2 and parts[0] and parts[1]:
                    cleaned.append(line)
            with open(merges_path, "w", encoding="utf-8") as f:
                f.write("\n".join(cleaned) + "\n")

        return local_dir

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        tokenizer_source = self._prepare_patched_tokenizer_dir(
            self._variant_config.pretrained_model_name
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            **tokenizer_kwargs,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen-1.8B-Llamafied model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Qwen-1.8B-Llamafied model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            **model_kwargs,
        )

        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen-1.8B-Llamafied model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        prompts = [self.sample_text] * batch_size

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        return inputs
