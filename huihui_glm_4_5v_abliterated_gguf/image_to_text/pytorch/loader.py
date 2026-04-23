# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui GLM-4.5V Abliterated GGUF model loader implementation for image to text.
"""

import inspect

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.modeling_utils as _modeling_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
from transformers import (
    AutoProcessor,
    Glm4vMoeConfig,
    Glm4vMoeForConditionalGeneration,
)
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


def _register_glm4moe_gguf_support():
    """Register glm4moe architecture in transformers GGUF support."""
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_TO_TRANSFORMERS_MAPPING,
        GGUF_SUPPORTED_ARCHITECTURES,
    )

    if "glm4moe" not in GGUF_TO_TRANSFORMERS_MAPPING.get("config", {}):
        GGUF_TO_TRANSFORMERS_MAPPING["config"]["glm4moe"] = {
            "context_length": "max_position_embeddings",
            "block_count": "num_hidden_layers",
            "feed_forward_length": "intermediate_size",
            "embedding_length": "hidden_size",
            "rope.dimension_count": None,
            "rope.freq_base": "rope_theta",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.layer_norm_rms_epsilon": "rms_norm_eps",
            "vocab_size": "vocab_size",
            "expert_count": "n_routed_experts",
            "expert_used_count": "num_experts_per_tok",
            "expert_feed_forward_length": "moe_intermediate_size",
            "attention.key_length": "head_dim",
            "leading_dense_block_count": "first_k_dense_replace",
            "expert_shared_count": "n_shared_experts",
        }

    if "glm4moe" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("glm4moe")


_register_glm4moe_gguf_support()


def _find_real_load_gguf_checkpoint(fn):
    """Traverse patch chain to find the real transformers load_gguf_checkpoint."""
    seen = set()
    current = fn
    while True:
        fn_id = id(current)
        if fn_id in seen or not callable(current) or not hasattr(current, "__code__"):
            return current
        seen.add(fn_id)
        if (
            getattr(current, "__module__", "")
            == "transformers.modeling_gguf_pytorch_utils"
        ):
            return current
        try:
            if "model_to_load" in inspect.signature(current).parameters:
                return current
        except (ValueError, TypeError):
            pass
        freevars = current.__code__.co_freevars
        cells = current.__closure__ or ()
        next_fn = None
        for i, varname in enumerate(freevars):
            if i >= len(cells):
                break
            if "load_gguf_checkpoint" in varname or "orig_load" in varname:
                try:
                    v = cells[i].cell_contents
                    if callable(v) and id(v) not in seen:
                        next_fn = v
                        break
                except ValueError:
                    pass
        if next_fn is None:
            v = getattr(current, "__globals__", {}).get("_orig_load_gguf_checkpoint")
            if v is not None and callable(v) and id(v) not in seen:
                next_fn = v
        if next_fn is None:
            return current
        current = next_fn


_real_load_gguf_checkpoint = _find_real_load_gguf_checkpoint(
    _gguf_utils.load_gguf_checkpoint
)


def _patched_load_gguf_checkpoint(
    gguf_checkpoint_path, return_tensors=False, model_to_load=None, **kwargs
):
    return _real_load_gguf_checkpoint(
        gguf_checkpoint_path,
        return_tensors=return_tensors,
        model_to_load=model_to_load,
        **kwargs,
    )


_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_modeling_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


# Patch get_gguf_hf_weights_map to handle glm4v_moe -> glm4moe and missing num_hidden_layers
_orig_get_gguf_hf_weights_map = _gguf_utils.get_gguf_hf_weights_map


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    if model_type is None:
        model_type = hf_model.config.model_type
    # Remap glm4v_moe variants to the gguf-py architecture name
    if model_type in ("glm4v_moe", "glm4v_moe_text"):
        model_type = "glm4moe"
    if num_layers is None:
        if hasattr(hf_model.config, "num_hidden_layers"):
            num_layers = hf_model.config.num_hidden_layers
        elif hasattr(hf_model.config, "text_config") and hasattr(
            hf_model.config.text_config, "num_hidden_layers"
        ):
            num_layers = hf_model.config.text_config.num_hidden_layers
        else:
            num_layers = 0
    return _orig_get_gguf_hf_weights_map(
        hf_model, processor, model_type, num_layers, qual_name
    )


_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


class ModelVariant(StrEnum):
    """Available Huihui GLM-4.5V Abliterated GGUF model variants for image to text."""

    HUIHUI_GLM_4_5V_ABLITERATED_MRADERMACHER_GGUF = (
        "huihui_glm_4_5v_abliterated_mradermacher_gguf"
    )


class ModelLoader(ForgeModel):
    """Huihui GLM-4.5V Abliterated GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_GLM_4_5V_ABLITERATED_MRADERMACHER_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-GLM-4.5V-abliterated-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_GLM_4_5V_ABLITERATED_MRADERMACHER_GGUF

    _GGUF_FILES = {
        ModelVariant.HUIHUI_GLM_4_5V_ABLITERATED_MRADERMACHER_GGUF: "Huihui-GLM-4.5V-abliterated.Q4_K_M.gguf",
    }

    sample_image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Huihui GLM-4.5V Abliterated GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @property
    def _gguf_file(self):
        """Get the GGUF filename for the current variant."""
        return self._GGUF_FILES.get(self._variant)

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["gguf_file"] = self._gguf_file
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor or config; use the base model for both
        self.processor = AutoProcessor.from_pretrained("zai-org/GLM-4.5V")
        config = Glm4vMoeConfig.from_pretrained("zai-org/GLM-4.5V")

        # Other loaders may have overwritten _gguf_utils.load_gguf_checkpoint with
        # versions that don't accept model_to_load. Restore the real transformers
        # function for our call and then put the chain head back.
        prev_load = _gguf_utils.load_gguf_checkpoint
        _gguf_utils.load_gguf_checkpoint = _real_load_gguf_checkpoint
        try:
            model = Glm4vMoeForConditionalGeneration.from_pretrained(
                pretrained_model_name, config=config, **model_kwargs
            )
        finally:
            _gguf_utils.load_gguf_checkpoint = prev_load

        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.sample_image,
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
