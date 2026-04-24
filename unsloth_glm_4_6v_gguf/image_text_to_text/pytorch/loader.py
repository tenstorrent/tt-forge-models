# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unsloth GLM-4.6V GGUF model loader implementation for image-text-to-text tasks.
"""
import torch
from transformers import AutoProcessor, AutoConfig, Glm4vMoeForConditionalGeneration
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
from ....tools.utils import get_file
from PIL import Image


def _patch_transformers_glm4v_moe_gguf():
    """Monkey-patch transformers to add glm4v_moe GGUF architecture support.

    The GLM-4.6V model uses the 'glm4moe' architecture in its GGUF metadata.
    Transformers 5.x has Glm4vMoeForConditionalGeneration but lacks GGUF loading
    support for glm4moe. We bridge the gap by registering the config mapping,
    remapping model_type to glm4v_moe, and teaching get_gguf_hf_weights_map to
    resolve the composite-config num_hidden_layers for tensor name lookup.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "glm4moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # 1. Register glm4moe as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("glm4moe")

    # 2. Add config field mapping: glm4moe GGUF keys -> Glm4vMoeTextConfig fields
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
        "attention.key_length": "head_dim",
        "attention.value_length": None,
        "vocab_size": "vocab_size",
        "expert_used_count": "num_experts_per_tok",
        "expert_group_count": "n_group",
        "expert_group_used_count": "topk_group",
        "expert_count": "n_routed_experts",
        "expert_feed_forward_length": "moe_intermediate_size",
        "expert_shared_count": "n_shared_experts",
        "leading_dense_block_count": "first_k_dense_replace",
    }

    # 3. Register tokenizer converter (BPE-based, same style as other GLM models)
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "glm4moe" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["glm4moe"] = GGUFQwen2Converter

    # 4. Patch load_gguf_checkpoint to remap model_type and build rope_parameters.
    # The GGUF config is flat (text fields only); we set model_type='glm4v_moe' so
    # that Glm4vMoeConfig is constructed with the text fields going to text_config
    # and a default vision_config. We also build rope_parameters from rope_theta
    # and partial_rotary_factor so the text backbone RoPE is correctly configured.
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "glm4moe":
            config["model_type"] = "glm4v_moe"
            head_dim = config.get("head_dim", 128)
            rope_theta = config.pop("rope_theta", 500000.0)
            partial_rotary_factor = config.get("partial_rotary_factor")
            if partial_rotary_factor is None:
                try:
                    from gguf import GGUFReader
                    from transformers.modeling_gguf_pytorch_utils import (
                        _gguf_parse_value,
                    )

                    reader = GGUFReader(args[0])
                    for key, field in reader.fields.items():
                        if "rope.dimension_count" in key:
                            rope_dim = _gguf_parse_value(
                                field.parts[field.data[0]], field.types
                            )
                            partial_rotary_factor = rope_dim / head_dim
                            break
                except Exception:
                    pass
                if partial_rotary_factor is None:
                    partial_rotary_factor = 0.5
            config["partial_rotary_factor"] = partial_rotary_factor
            # mrope_section matches the zai-org/GLM-4.6V reference config
            config["rope_parameters"] = {
                "rope_type": "default",
                "rope_theta": rope_theta,
                "partial_rotary_factor": partial_rotary_factor,
                "mrope_section": [8, 12, 12],
            }
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # 5. Patch get_gguf_hf_weights_map so that glm4v_moe composite configs resolve
    # to the 'glm4moe' gguf-py arch.  The composite Glm4vMoeConfig does not expose
    # num_hidden_layers directly, so we read it from the nested text_config.
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type

        if model_type in ("glm4v_moe", "glm4v_moe_text"):
            if num_layers is None:
                text_cfg = getattr(hf_model.config, "text_config", None)
                if text_cfg is not None:
                    num_layers = getattr(text_cfg, "num_hidden_layers", None)
                if num_layers is None:
                    num_layers = getattr(hf_model.config, "num_hidden_layers", None)
            model_type = "glm4moe"

        return orig_get_map(
            hf_model,
            processor,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


_patch_transformers_glm4v_moe_gguf()


class ModelVariant(StrEnum):
    """Available Unsloth GLM-4.6V GGUF model variants for image-text-to-text tasks."""

    GLM_4_6V_GGUF_Q2_K = "glm_4_6v_gguf_q2_k"


class ModelLoader(ForgeModel):
    """Unsloth GLM-4.6V GGUF model loader implementation for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.GLM_4_6V_GGUF_Q2_K: LLMModelConfig(
            pretrained_model_name="unsloth/GLM-4.6V-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_4_6V_GGUF_Q2_K

    GGUF_FILE = "GLM-4.6V-Q2_K.gguf"

    # Processor is loaded from the original GLM-4.6V repo since the GGUF repo
    # only hosts quantized model weights without processor/tokenizer configs.
    PROCESSOR_MODEL = "zai-org/GLM-4.6V"

    sample_text = "What do you see in this image?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="unsloth_glm_4_6v_gguf",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(self.PROCESSOR_MODEL, **kwargs)

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        model = Glm4vMoeForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        image_file = get_file(self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.sample_text},
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

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self.GGUF_FILE,
        )
        return self.config
