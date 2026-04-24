# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ASID Captioner 3B i1 GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen2VLForConditionalGeneration,
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

_TRANSFORMERS_LOAD_GGUF = None


def _find_real_load_gguf_checkpoint():
    """Traverse the patch chain to find the original transformers function.

    Other GGUF loaders monkey-patch load_gguf_checkpoint with wrappers that
    have fixed signatures (gguf_path, return_tensors=False) and cannot handle
    the model_to_load kwarg that transformers passes on the second call.
    We bypass the entire chain and call the real function directly.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    target_module = "transformers.modeling_gguf_pytorch_utils"
    fn = gguf_utils.load_gguf_checkpoint
    seen: set = set()

    while fn is not None:
        fn_id = id(fn)
        if fn_id in seen:
            break
        seen.add(fn_id)

        if getattr(fn, "__module__", None) == target_module:
            return fn

        closure = getattr(fn, "__closure__", None)
        if not closure:
            break

        next_fn = None
        for cell in closure:
            try:
                val = cell.cell_contents
                if callable(val) and id(val) not in seen:
                    next_fn = val
                    break
            except ValueError:
                pass

        fn = next_fn

    return gguf_utils.load_gguf_checkpoint


def _register_qwen2vl_gguf_architecture():
    """Register qwen2vl in transformers GGUF config mapping and save the real load function."""
    global _TRANSFORMERS_LOAD_GGUF

    from transformers.integrations.ggml import (
        GGUF_CONFIG_MAPPING,
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
    )

    if "qwen2vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")

    if "qwen2vl" not in GGUF_CONFIG_MAPPING:
        GGUF_CONFIG_MAPPING["qwen2vl"] = {
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

    if "qwen2vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen2vl"] = GGUFQwen2Converter

    # Save real transformers function now, before other loaders patch it further.
    # We use it later to bypass broken intermediate patch chains.
    if _TRANSFORMERS_LOAD_GGUF is None:
        _TRANSFORMERS_LOAD_GGUF = _find_real_load_gguf_checkpoint()


_register_qwen2vl_gguf_architecture()


def _apply_qwen2vl_load_patch():
    """Install a load_gguf_checkpoint wrapper that handles qwen2vl model_type remapping.

    Installs immediately before from_pretrained to be the outermost wrapper.
    Calls the real transformers function directly (bypassing other models'
    broken intermediate patches) then remaps qwen2vl -> qwen2_vl.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    real_fn = _TRANSFORMERS_LOAD_GGUF or _find_real_load_gguf_checkpoint()

    def _qwen2vl_patched(*args, **kwargs):
        result = real_fn(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen2vl":
            config["model_type"] = "qwen2_vl"
        return result

    gguf_utils.load_gguf_checkpoint = _qwen2vl_patched
    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = _qwen2vl_patched


class ModelVariant(StrEnum):
    """Available ASID Captioner 3B i1 GGUF model variants for image to text."""

    ASID_CAPTIONER_3B_I1_Q4_K_M_GGUF = "3b_i1_Q4_K_M_gguf"


class ModelLoader(ForgeModel):
    """ASID Captioner 3B i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.ASID_CAPTIONER_3B_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/ASID-Captioner-3B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ASID_CAPTIONER_3B_I1_Q4_K_M_GGUF

    GGUF_FILE = "ASID-Captioner-3B.i1-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ASID Captioner 3B i1 GGUF",
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
        self.processor = AutoProcessor.from_pretrained(
            "AudioVisual-Caption/ASID-Captioner-3B"
        )

        # Re-apply patch right before from_pretrained so our wrapper is the
        # outermost and calls the real transformers function directly.
        _apply_qwen2vl_load_patch()

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
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
