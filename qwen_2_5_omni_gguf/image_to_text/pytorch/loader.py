# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 2.5 Omni GGUF model loader implementation for image to text.
"""
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
from typing import Optional


def _patch_transformers_qwen2vl_gguf():
    """Monkey-patch transformers to add qwen2vl GGUF architecture support.

    Transformers 5.x supports Qwen2VLForConditionalGeneration but lacks GGUF
    loading for the qwen2vl architecture. We bridge the gap by registering the
    architecture and mapping its GGUF config keys to HF config field names.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen2vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # Register qwen2vl as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")

    # Map GGUF config keys to HF config field names (text backbone only)
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen2vl"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.dimension_sections": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    }

    # Reuse qwen2 BPE tokenizer converter
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "qwen2vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen2vl"] = GGUFQwen2Converter

    # Patch load_gguf_checkpoint to remap model_type qwen2vl -> qwen2_vl and
    # inject mrope_section into rope_parameters (required by Qwen2VL M-RoPE).
    # mrope_section = [dim_t, dim_h, dim_w] where dim_t = rope.dimension_sections
    # and dim_h = dim_w = (head_dim/2 - dim_t) / 2.
    orig_load = gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") != "qwen2vl":
            return result
        config["model_type"] = "qwen2_vl"
        # Compute mrope_section from GGUF metadata
        try:
            from gguf import GGUFReader
            from transformers.modeling_gguf_pytorch_utils import _gguf_parse_value

            gguf_path = args[0] if args else kwargs.get("gguf_checkpoint_path")
            reader = GGUFReader(gguf_path)
            dim_sections = None
            for key, field in reader.fields.items():
                if "rope.dimension_sections" in key:
                    dim_sections = _gguf_parse_value(
                        field.parts[field.data[0]], field.types
                    )
                    break
            if dim_sections is not None:
                n_heads = config.get("num_attention_heads", 28)
                hidden_size = config.get("hidden_size", 3584)
                head_dim = hidden_size // n_heads
                rope_dims = head_dim // 2
                dim_spatial = (rope_dims - dim_sections) // 2
                mrope_section = [int(dim_sections), dim_spatial, dim_spatial]
                config.setdefault("rope_parameters", {})
                config["rope_parameters"]["mrope_section"] = mrope_section
                config["rope_parameters"].setdefault("rope_type", "default")
        except Exception:
            pass
        return result

    gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    # Patch modules that imported load_gguf_checkpoint directly
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    # Patch get_gguf_hf_weights_map to handle qwen2_vl composite config
    # (num_hidden_layers lives in text_config, and model_type must be qwen2vl for gguf-py lookup)
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        cfg = getattr(hf_model, "config", None)
        if model_type is None and cfg is not None:
            model_type = getattr(cfg, "model_type", None)
        if model_type == "qwen2_vl":
            model_type = "qwen2vl"
        if num_layers is None and cfg is not None:
            # Qwen2VLConfig nests num_hidden_layers inside text_config
            num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(
                getattr(cfg, "text_config", None), "num_hidden_layers", None
            )
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map
    if hasattr(modeling_utils, "get_gguf_hf_weights_map"):
        modeling_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_transformers_qwen2vl_gguf()

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
    """Available Qwen 2.5 Omni GGUF model variants for image to text."""

    QWEN_2_5_OMNI_7B_Q4_K_M = "7b_q4_k_m"


class ModelLoader(ForgeModel):
    """Qwen 2.5 Omni GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_2_5_OMNI_7B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="ggml-org/Qwen2.5-Omni-7B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_2_5_OMNI_7B_Q4_K_M

    # Processor configs are not in the GGUF repo; load from the base model.
    _BASE_MODEL_NAMES = {
        ModelVariant.QWEN_2_5_OMNI_7B_Q4_K_M: "Qwen/Qwen2.5-Omni-7B",
    }

    @property
    def _base_model_name(self):
        return self._BASE_MODEL_NAMES[self._variant]

    _GGUF_FILES = {
        ModelVariant.QWEN_2_5_OMNI_7B_Q4_K_M: "Qwen2.5-Omni-7B-Q4_K_M.gguf",
    }

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    @property
    def _gguf_file(self):
        return self._GGUF_FILES[self._variant]

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use.
        """
        super().__init__(variant)
        self.processor = None
        self.config = None
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
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 2.5 Omni GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 2.5 Omni GGUF model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Qwen 2.5 Omni GGUF model instance for image to text.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self._gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self._gguf_file
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        self.processor = AutoProcessor.from_pretrained(self._base_model_name)

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen 2.5 Omni GGUF model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file,
            trust_remote_code=True,
        )
        return self.config
