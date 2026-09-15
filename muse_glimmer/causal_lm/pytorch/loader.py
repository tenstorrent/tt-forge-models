# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Muse-Glimmer 30B model loader implementation.

``meta-models/Muse-Glimmer-30B`` is a multimodal MuseGlimmerForConditionalGeneration
(text + vision). Native support requires transformers>=5.15 (pinned in
requirements.txt). The default path here is text-only causal LM inputs; vision
weights are left unreplicated / out of the shard map for bring-up.
"""

from typing import Optional

import torch

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

# NOTE: ``transformers`` is intentionally NOT imported at module top level.
# This model pins transformers==5.15.0 (see requirements.txt). The test runner
# installs that pin at test time and purges transformers from sys.modules. A
# top-level import would bind Auto* classes to whatever transformers was loaded
# during pytest collection. Auto* classes are imported lazily in the methods
# that use them.


class ModelVariant(StrEnum):
    """Available Muse-Glimmer model variants."""

    MUSE_GLIMMER_30B = "Muse-Glimmer-30B"


class ModelLoader(ForgeModel):
    """Muse-Glimmer 30B multimodal model loader (text-only bring-up path)."""

    _VARIANTS = {
        ModelVariant.MUSE_GLIMMER_30B: LLMModelConfig(
            pretrained_model_name="meta-models/Muse-Glimmer-30B",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MUSE_GLIMMER_30B

    sample_text = "Who are you?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant.
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Muse-Glimmer",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _text_config(self):
        return getattr(self.config, "text_config", self.config)

    def _load_tokenizer(self):
        """Load tokenizer for the current variant."""
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Muse-Glimmer model for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: MuseGlimmerForConditionalGeneration.
        """
        # Lazy import so it binds to the pinned transformers (see module note).
        from transformers import AutoModelForMultimodalLM

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMultimodalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False
        model.eval()
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample text-only inputs for Muse-Glimmer.

        Args:
            dtype_override: Unused for tokenized integer inputs.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length
        if self.tokenizer.chat_template is not None:
            conversation = [{"role": "user", "content": self.sample_text}]
            prompt = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = self.sample_text
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        return inputs

    def unpack_forward_output(self, fwd_output):
        """Extract logits from the text-only CausalLM output."""
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        return super().unpack_forward_output(fwd_output)

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallel."""
        text_cfg = self._text_config()
        attn_heads = text_cfg.num_attention_heads
        mesh_shape = (1, num_devices)
        if attn_heads % mesh_shape[1] != 0:
            raise ValueError(
                f"Cannot evenly distribute {attn_heads} attention heads "
                f"across model axis size {mesh_shape[1]}"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron-style TP map for the Muse-Glimmer text decoder.

        Follows the ``qwen_3_5`` causal-LM pattern (``("model", "batch")`` /
        ``("batch", "model")``), adapted to Muse's ``language_model`` tower and
        gated attention::

            mlp.gate/up → ("model", "batch"); mlp.down → ("batch", "model")
            attn.q/gate/o → qwen-style q/o; k/v left replicated (GQA width 2)
            embed_tokens / lm_head → ("model", "batch")

        Vision weights are unused on the text-only path and stay out of the map.
        """
        shard_specs = {}

        for layer in model.model.language_model.layers:
            mlp = layer.mlp
            shard_specs[mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[mlp.up_proj.weight] = ("model", "batch")
            shard_specs[mlp.down_proj.weight] = ("batch", "model")

            sa = layer.self_attn
            shard_specs[sa.q_proj.weight] = ("batch", "model")
            shard_specs[sa.gate_proj.weight] = ("batch", "model")
            shard_specs[sa.o_proj.weight] = ("model", "batch")

        shard_specs[model.model.language_model.embed_tokens.weight] = (
            "model",
            "batch",
        )
        if hasattr(model, "lm_head"):
            shard_specs[model.lm_head.weight] = ("model", "batch")

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the model variant."""
        from transformers import AutoConfig

        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
