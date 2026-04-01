# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kimi K2 model loader implementation.

Uses a locally modified DeepSeek V3-based model (modeling_deepseek.py) instead of
loading directly from HuggingFace. The modifications adapt the model for
Tenstorrent hardware by replacing cache utilities with a static MLA cache.
"""
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

from .configuration_deepseek import DeepseekV3Config
from .modeling_deepseek import DeepseekV3ForCausalLM


class ModelVariant(StrEnum):
    """Available Kimi K2 model variants."""

    KIMI_K2_INSTRUCT = "kimi_k2_instruct"


class ModelLoader(ForgeModel):
    """Kimi K2 model loader using the locally modified DeepSeek V3-based Transformer."""

    _VARIANTS = {
        ModelVariant.KIMI_K2_INSTRUCT: LLMModelConfig(
            pretrained_model_name="moonshotai/Kimi-K2-Instruct",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KIMI_K2_INSTRUCT

    def __init__(
        self,
        variant=None,
        num_layers: Optional[int] = None,
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional variant string. If None, uses the default variant.
            num_layers: Number of transformer layers to instantiate.
                        If None, uses the full model depth from config.json (61 layers).
        """
        super().__init__(variant)
        self.num_layers = num_layers
        self.tokenizer = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Return model metadata for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string.

        Returns:
            ModelInfo: Information about the model and variant.
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="Kimi-K2",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Unused; kept for API compatibility.

        Returns:
            The loaded tokenizer instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def _load_config(self, num_layers: Optional[int] = None) -> DeepseekV3Config:
        """Load model configuration from the local config.json.

        Args:
            num_layers: If provided, overrides num_hidden_layers in the config.

        Returns:
            DeepseekV3Config: Populated config stored on ``self.config``.
        """
        config_path = Path(__file__).parent / "config.json"
        config = DeepseekV3Config.from_json_file(str(config_path))
        if num_layers is not None:
            config.num_hidden_layers = num_layers
        self.config = config
        return config

    def get_weight_dtype_config_path(self):
        """Return path to weight dtype config file, or None if not available."""
        return None

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Kimi K2 model.

        The model is instantiated from the local config.json and the locally
        modified modeling_deepseek.py. KV cache is disabled for compilation
        compatibility.

        Args:
            dtype_override: Optional torch.dtype to cast the model to after
                            construction (e.g. torch.bfloat16).

        Returns:
            torch.nn.Module: The Kimi K2 model in eval mode.
        """
        config = self._load_config(num_layers=self.num_layers)
        config.use_cache = False

        self._load_tokenizer()

        model = DeepseekV3ForCausalLM(config)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model.eval()

    def load_inputs(self, batch_size: int = 1, seq_len: int = 32):
        """Return sample token inputs for the model.

        Args:
            batch_size: Number of sequences in the batch.
            seq_len: Length of each input sequence.

        Returns:
            torch.Tensor: Integer token tensor of shape (batch_size, seq_len).
        """
        if self.config is None:
            self._load_config()
        return torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
