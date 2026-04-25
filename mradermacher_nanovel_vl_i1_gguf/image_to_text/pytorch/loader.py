# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mradermacher NaNovel VL i1 GGUF model loader implementation for image to text.
"""

from contextlib import contextmanager

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
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


def _patch_transformers_qwen3vl_gguf():
    """Monkey-patch transformers to add qwen3vl GGUF architecture support.

    The gguf library already knows about qwen3vl tensor names, but
    transformers lacks the config mapping and architecture registration
    needed to load qwen3vl GGUF checkpoints.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen3vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vl"] = {
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
    }

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "qwen3vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3vl"] = GGUFQwen2Converter

    _QWEN3VL_TEXT_CONFIG_KEYS = {
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "max_position_embeddings",
        "vocab_size",
    }

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen3vl":
            config["model_type"] = "qwen3_vl"
            # Move text backbone values into nested text_config so Qwen3VLConfig
            # properly initializes the text model with the GGUF dimensions.
            text_config = {}
            for key in list(config.keys()):
                if key in _QWEN3VL_TEXT_CONFIG_KEYS:
                    text_config[key] = config.pop(key)
            if text_config:
                # Qwen3VLTextRotaryEmbedding calls rope_scaling.get(...) so it
                # must be a dict with rope_type. GGUF files omit rope_scaling;
                # supply the 32B defaults (mrope_section matches head_dim/2=64).
                if "rope_scaling" not in text_config:
                    text_config["rope_scaling"] = {
                        "rope_type": "default",
                        "mrope_section": [24, 20, 20],
                        "mrope_interleaved": True,
                    }
                config["text_config"] = text_config
                # The GGUF file doesn't include vision config params so
                # vision_config.out_hidden_size defaults to 3584. We must
                # override it to match the text hidden_size so the merger
                # output matches the text embedding dimension.
                if "hidden_size" in text_config:
                    config["vision_config"] = {
                        "out_hidden_size": text_config["hidden_size"]
                    }
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_transformers_qwen3vl_gguf()


@contextmanager
def _qwen3vl_weights_map_patch():
    """Transiently install a get_gguf_hf_weights_map patch as the outermost wrapper.

    Applied inside load_model() so it is always the outermost wrapper regardless
    of other loaders that patch the same function at import time. For Qwen3VL
    models the top-level config stores num_hidden_layers inside text_config.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    prev = gguf_utils.get_gguf_hf_weights_map

    def patched(hf_model, model_type=None, num_layers=None, qual_name=""):
        if model_type is None:
            model_type = getattr(getattr(hf_model, "config", None), "model_type", None)
        if model_type == "qwen3_vl":
            model_type = "qwen3vl"
            if num_layers is None:
                try:
                    num_layers = hf_model.config.text_config.num_hidden_layers
                except AttributeError:
                    pass
        return prev(
            hf_model,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    gguf_utils.get_gguf_hf_weights_map = patched
    try:
        yield
    finally:
        gguf_utils.get_gguf_hf_weights_map = prev


class ModelVariant(StrEnum):
    """Available Mradermacher NaNovel VL i1 GGUF model variants for image to text."""

    NANOVEL_VL_I1_GGUF = "nanovel_vl_i1_gguf"


class ModelLoader(ForgeModel):
    """Mradermacher NaNovel VL i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.NANOVEL_VL_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/NaNovel-VL-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NANOVEL_VL_I1_GGUF

    GGUF_FILE = "NaNovel-VL.i1-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Mradermacher NaNovel VL i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-32B-Instruct")

        with _qwen3vl_weights_map_patch():
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
            )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
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
