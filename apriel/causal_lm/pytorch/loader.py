# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Apriel model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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
from ....tools.utils import cast_input_to_type


def _patch_transformers_for_apriel():
    """Make the model's `trust_remote_code` modeling file importable on transformers >= 5.

    The Apriel remote code (modeling_apriel.py) was authored against an older
    transformers that exported ``LossKwargs`` from ``transformers.utils``. That
    symbol was renamed to ``TransformersKwargs`` in transformers 5.x. Alias it
    back so ``from transformers.utils import LossKwargs`` keeps working. The
    symbol is used only as a typing hint inside the modeling file, so the alias
    is behaviourally transparent.
    """
    import transformers.utils as _tu

    if not hasattr(_tu, "LossKwargs"):
        replacement = getattr(_tu, "TransformersKwargs", None)
        if replacement is None:
            from typing import Optional, TypedDict

            class replacement(TypedDict, total=False):  # noqa: N801
                num_items_in_batch: Optional[int]

        _tu.LossKwargs = replacement


class ModelVariant(StrEnum):
    """Available Apriel model variants for causal LM."""

    APRIEL_5B_INSTRUCT = "5b_instruct"


class ModelLoader(ForgeModel):
    """Apriel model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.APRIEL_5B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="ServiceNow-AI/Apriel-5B-Instruct",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.APRIEL_5B_INSTRUCT

    # Sample text for causal LM. Long enough to fill a tile-aligned sequence
    # with real tokens so the device run is not dominated by padding positions.
    sample_text = (
        "The history of mathematics spans thousands of years and crosses many "
        "cultures, from ancient Babylon and Egypt to classical Greece, India, "
        "the Islamic world, and modern Europe, shaping science and engineering."
    )

    # Fixed, tile-aligned sequence length for device inputs.
    seq_len = 32

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the
                        model's default.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="Apriel",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        # Apriel ships a pad token, but fall back to eos if it is missing.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Apriel model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model uses the dtype from its config.

        Returns:
            torch.nn.Module: The Apriel model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Apriel ships custom modeling code that needs a small compat shim on
        # transformers >= 5; apply it before the dynamic module is imported.
        _patch_transformers_for_apriel()

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # Replace the in-graph RoPE cos/sin computation with a precomputed
        # lookup table (see _install_precomputed_rope for the rationale).
        self._install_precomputed_rope(model)

        model.eval()

        self.model = model
        self.config = model.config

        return model

    def _install_precomputed_rope(self, model):
        """Swap Apriel's rotary embedding for a precomputed cos/sin lookup table.

        Apriel computes RoPE angles in-graph (``freqs = inv_freq @ position_ids``)
        and feeds them straight into ``cos``/``sin``. The angles reach tens of
        radians, and the device trig kernels overflow to ``inf`` outside a small
        range, which destroys the result (output PCC ~0 vs the CPU reference).

        We instead precompute cos/sin on the host in float32 for all positions up
        to a fixed cap and look them up by ``position_ids`` at run time. The table
        is a graph constant, so the device never evaluates trig and the result
        matches the float32 reference (PCC > 0.99).
        """
        import types

        rotary = model.model.rotary_emb
        max_positions = getattr(model.config, "max_position_embeddings", 4096)
        cap = int(min(max_positions, 4096))

        positions = torch.arange(cap, dtype=torch.float32)
        inv_freq = rotary.inv_freq.detach().to(torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        attention_scaling = float(getattr(rotary, "attention_scaling", 1.0))

        rotary.register_buffer(
            "cos_cached", emb.cos() * attention_scaling, persistent=False
        )
        rotary.register_buffer(
            "sin_cached", emb.sin() * attention_scaling, persistent=False
        )

        def _precomputed_forward(self, x, position_ids):
            cos = self.cos_cached[position_ids].to(dtype=x.dtype)
            sin = self.sin_cached[position_ids].to(dtype=x.dtype)
            return cos, sin

        rotary.forward = types.MethodType(_precomputed_forward, rotary)

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        """Load and return sample inputs for the Apriel model.

        Args:
            dtype_override: Optional torch.dtype applied to floating-point inputs.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors suitable for causal LM.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # Produce a fixed, tile-aligned sequence of fully-valid tokens (no
        # padding). The sample text is long enough to fill seq_len so the
        # attention mask is all ones, which keeps the device result faithful.
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.seq_len,
        )

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested (integer ids stay integer).
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return inputs

    def decode_output(self, outputs, inputs=None):
        """Decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass.
            inputs: Optional input tensors used to generate the outputs.

        Returns:
            str: Decoded prediction for the next tokens.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        predicted_token_ids = logits.argmax(dim=-1)
        return self.tokenizer.decode(
            predicted_token_ids[0], skip_special_tokens=True
        )

    def load_config(self):
        """Load and return the configuration for the Apriel model variant.

        Returns:
            The configuration object for the Apriel model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config
