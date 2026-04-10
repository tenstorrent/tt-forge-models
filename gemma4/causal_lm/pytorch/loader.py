# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma4 model loader implementation for causal language modeling.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional

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
    """Available Gemma4 model variants for causal LM."""

    GEMMA_4_E2B = "E2B"
    GEMMA_4_E2B_IT = "E2B_Instruct"
    GEMMA_4_E4B = "E4B"
    GEMMA_4_E4B_IT = "E4B_Instruct"
    GEMMA_4_26B_A4B = "26B_A4B"
    GEMMA_4_26B_A4B_IT = "26B_A4B_Instruct"
    GEMMA_4_31B = "31B"
    GEMMA_4_31B_IT = "31B_Instruct"


_INSTRUCT_VARIANTS = {
    ModelVariant.GEMMA_4_E2B_IT,
    ModelVariant.GEMMA_4_E4B_IT,
    ModelVariant.GEMMA_4_26B_A4B_IT,
    ModelVariant.GEMMA_4_31B_IT,
}

_SMALL_VARIANTS = {
    ModelVariant.GEMMA_4_E2B,
    ModelVariant.GEMMA_4_E2B_IT,
}

_MOE_VARIANTS = {
    ModelVariant.GEMMA_4_26B_A4B,
    ModelVariant.GEMMA_4_26B_A4B_IT,
}


class ModelLoader(ForgeModel):
    """Gemma4 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_4_E2B: LLMModelConfig(
            pretrained_model_name="google/gemma-4-E2B",
            max_length=256,
        ),
        ModelVariant.GEMMA_4_E2B_IT: LLMModelConfig(
            pretrained_model_name="google/gemma-4-E2B-it",
            max_length=256,
        ),
        ModelVariant.GEMMA_4_E4B: LLMModelConfig(
            pretrained_model_name="google/gemma-4-E4B",
            max_length=256,
        ),
        ModelVariant.GEMMA_4_E4B_IT: LLMModelConfig(
            pretrained_model_name="google/gemma-4-E4B-it",
            max_length=256,
        ),
        ModelVariant.GEMMA_4_26B_A4B: LLMModelConfig(
            pretrained_model_name="google/gemma-4-26B-A4B",
            max_length=256,
        ),
        ModelVariant.GEMMA_4_26B_A4B_IT: LLMModelConfig(
            pretrained_model_name="google/gemma-4-26B-A4B-it",
            max_length=256,
        ),
        ModelVariant.GEMMA_4_31B: LLMModelConfig(
            pretrained_model_name="google/gemma-4-31B",
            max_length=256,
        ),
        ModelVariant.GEMMA_4_31B_IT: LLMModelConfig(
            pretrained_model_name="google/gemma-4-31B-it",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_4_E4B_IT

    sample_text = "What is your favorite city?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        if variant in _SMALL_VARIANTS:
            group = ModelGroup.GENERALITY
        else:
            group = ModelGroup.RED

        return ModelInfo(
            model="Gemma 4",
            variant=variant,
            group=group,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Gemma4 causal LM model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.text_config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(
        self,
        dtype_override=None,
        batch_size=1,
        max_new_tokens: int = 256,
        prompt: Optional[str] = None,
    ):
        """Load and return sample inputs for the Gemma4 model.

        Returns a dict with input_ids, attention_mask, and use_cache keys
        since the forward signature has pixel_values before attention_mask.
        """
        max_length = self._variant_config.max_length
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        if self._variant in _INSTRUCT_VARIANTS:
            input_prompt = [
                {
                    "role": "user",
                    "content": prompt or self.sample_text,
                }
            ]
            input_text = self.tokenizer.apply_chat_template(
                input_prompt,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            input_text = prompt or self.sample_text

        inputs = self.tokenizer(
            [input_text],
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        if dtype_override is not None:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attn_mask = cast_input_to_type(attn_mask, dtype_override)
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "use_cache": False,
        }

    def get_mesh_config(self, num_devices: int):
        """Get the mesh configuration for tensor parallel execution.

        Prefers a 2D mesh (2, N/2) over a 1D mesh.  When the model axis
        (second dimension) does not divide all head counts, the axes are
        swapped so that the first dimension becomes the model axis —
        equivalent to an (N/2, 2) mesh which the compiler rejects directly.
        """
        if self._variant in _SMALL_VARIANTS:
            return None, None

        tc = self.config.text_config
        global_kv_heads = (
            getattr(tc, "num_global_key_value_heads", None) or tc.num_key_value_heads
        )

        def heads_divide(model_axis):
            return (
                tc.num_attention_heads % model_axis == 0
                and tc.num_key_value_heads % model_axis == 0
                and global_kv_heads % model_axis == 0
            )

        half = num_devices // 2
        if half > 1:
            # Standard 2D: model axis = half (e.g. 4 for 8 devices)
            if heads_divide(half):
                return (2, half), ("batch", "model")
            # Swapped 2D: model axis = 2, equivalent to (half, 2)
            if heads_divide(2):
                return (2, half), ("model", "batch")

        # Fallback to 1D mesh
        return (1, num_devices), ("batch", "model")

    def load_shard_spec(self, model):
        """Load the sharding specification for tensor parallel execution.

        Gemma4 uses heterogeneous attention: sliding layers use head_dim with
        more KV heads, while global layers use global_head_dim with fewer KV
        heads. For global attention K/V projections where the number of KV heads
        doesn't divide the device count, we shard along the input dimension
        instead of the output dimension to avoid reshape propagation failures.
        """
        if self._variant in _SMALL_VARIANTS:
            return None

        # Determine if attention can be sharded based on head divisibility
        tc = self.config.text_config
        num_devices = 8  # llmbox
        global_kv_heads = (
            getattr(tc, "num_global_key_value_heads", None) or tc.num_key_value_heads
        )
        _, mesh_names = self.get_mesh_config(num_devices)
        # Find effective model axis size from mesh config
        mesh_shape_check = self.get_mesh_config(num_devices)[0]
        model_axis = mesh_shape_check[1] if mesh_names == ("batch", "model") else mesh_shape_check[0]
        can_shard_attn = (
            tc.num_attention_heads % model_axis == 0
            and tc.num_key_value_heads % model_axis == 0
            and global_kv_heads % model_axis == 0
        )

        shard_specs = {}
        for layer in model.model.language_model.layers:
            if can_shard_attn:
                attn = layer.self_attn
                shard_specs[attn.q_proj.weight] = ("model", "batch")
                if getattr(attn, "k_proj", None) is not None:
                    shard_specs[attn.k_proj.weight] = ("model", "batch")
                if getattr(attn, "v_proj", None) is not None:
                    shard_specs[attn.v_proj.weight] = ("model", "batch")
                shard_specs[attn.o_proj.weight] = ("batch", "model")

            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

        return shard_specs
