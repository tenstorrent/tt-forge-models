# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui Qwen3 VL 4B Abliterated GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen3VLConfig,
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


def _find_real_load_gguf():
    """Walk the monkey-patch chain to find the real transformers load_gguf_checkpoint.

    Several loaders install broken wrappers with fixed signatures
    (gguf_path, return_tensors=False) that drop the model_to_load kwarg added
    in transformers 5.2.0. Some wrappers chain via __globals__['_orig_load_gguf_checkpoint'];
    others use a closure variable (e.g. orig_load). We BFS both paths to find the
    first function whose signature explicitly accepts model_to_load.
    """
    import inspect
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils

    seen_ids: set = set()
    queue = [_gguf_utils.load_gguf_checkpoint]

    while queue:
        fn = queue.pop(0)
        if fn is None or not callable(fn):
            continue
        fn_id = id(fn)
        if fn_id in seen_ids:
            continue
        seen_ids.add(fn_id)

        try:
            sig = inspect.signature(fn)
            if "model_to_load" in sig.parameters:
                return fn
        except (ValueError, TypeError):
            pass

        # Path 1: module-global _orig_load_gguf_checkpoint (most loaders)
        next_g = getattr(fn, "__globals__", {}).get("_orig_load_gguf_checkpoint")
        if next_g is not None and callable(next_g):
            queue.append(next_g)

        # Path 2: closure cells whose names suggest an "orig" or "load" function
        code = getattr(fn, "__code__", None)
        closure = getattr(fn, "__closure__", None)
        if code is not None and closure is not None:
            for varname, cell in zip(code.co_freevars, closure):
                if "orig" in varname or "load" in varname:
                    try:
                        val = cell.cell_contents
                        if callable(val):
                            queue.append(val)
                    except ValueError:
                        pass

    return _gguf_utils.load_gguf_checkpoint


def _register_qwen3vl_gguf_architecture():
    """Register qwen3vl GGUF architecture in transformers.

    GGUF files for Qwen3-VL models store general.architecture = "qwen3vl"
    (no underscore). transformers does not have qwen3vl in
    GGUF_SUPPORTED_ARCHITECTURES, so load_gguf_checkpoint raises ValueError.
    Also, get_gguf_hf_weights_map uses hf_model.config.model_type = "qwen3_vl"
    (with underscore), which gguf-py cannot find in MODEL_ARCH_NAMES since it
    expects "qwen3vl".

    Fix:
    1. Add qwen3vl to GGUF_CONFIG_MAPPING (and GGUF_SUPPORTED_ARCHITECTURES).
    2. Patch get_gguf_hf_weights_map to remap "qwen3_vl" -> "qwen3vl".
    3. Install a top-level load_gguf_checkpoint wrapper that properly forwards
       model_to_load (bypassing broken fixed-signature wrappers from other
       loaders that dropped this kwarg added in transformers 5.2.0).
    """
    from transformers.integrations.ggml import GGUF_CONFIG_MAPPING
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils

    if "qwen3vl" in GGUF_CONFIG_MAPPING:
        return

    GGUF_CONFIG_MAPPING["qwen3vl"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.key_length": "head_dim",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    }
    _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    # Fix (2): patch get_gguf_hf_weights_map for model_type underscore mismatch
    _orig_get_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_get_map(hf_model, processor, model_type=None, num_layers=None, **kwargs):
        if model_type is None and hasattr(hf_model, "config"):
            model_type = hf_model.config.model_type
        if model_type == "qwen3_vl":
            model_type = "qwen3vl"
            if num_layers is None and hasattr(hf_model, "config"):
                cfg = hf_model.config
                num_layers = getattr(
                    getattr(cfg, "text_config", cfg), "num_hidden_layers", None
                )
        return _orig_get_map(
            hf_model, processor, model_type=model_type, num_layers=num_layers, **kwargs
        )

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_map

    # Fix (3): install a properly-signed top wrapper that routes model_to_load
    # to the real transformers function while keeping config-loading compat.
    _real_fn = _find_real_load_gguf()
    _broken_chain = _gguf_utils.load_gguf_checkpoint

    def _compat_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
        if kwargs:
            return _real_fn(gguf_path, return_tensors=return_tensors, **kwargs)
        return _broken_chain(gguf_path, return_tensors=return_tensors)

    _gguf_utils.load_gguf_checkpoint = _compat_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available Huihui Qwen3 VL 4B Abliterated GGUF model variants for image to text."""

    HUIHUI_QWEN3_VL_4B_INSTRUCT_ABLITERATED_GGUF = "4b_instruct_abliterated_gguf"
    HUIHUI_QWEN3_VL_4B_INSTRUCT_ABLITERATED_MRADERMACHER_GGUF = (
        "4b_instruct_abliterated_mradermacher_gguf"
    )


class ModelLoader(ForgeModel):
    """Huihui Qwen3 VL 4B Abliterated GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_QWEN3_VL_4B_INSTRUCT_ABLITERATED_GGUF: LLMModelConfig(
            pretrained_model_name="noctrex/Huihui-Qwen3-VL-4B-Instruct-abliterated-GGUF",
            max_length=128,
        ),
        ModelVariant.HUIHUI_QWEN3_VL_4B_INSTRUCT_ABLITERATED_MRADERMACHER_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-Qwen3-VL-4B-Instruct-abliterated-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_QWEN3_VL_4B_INSTRUCT_ABLITERATED_GGUF

    _GGUF_FILES = {
        ModelVariant.HUIHUI_QWEN3_VL_4B_INSTRUCT_ABLITERATED_GGUF: "Huihui-Qwen3-VL-4B-Instruct-abliterated-Q4_K_M.gguf",
        ModelVariant.HUIHUI_QWEN3_VL_4B_INSTRUCT_ABLITERATED_MRADERMACHER_GGUF: "Huihui-Qwen3-VL-4B-Instruct-abliterated.Q4_K_M.gguf",
    }

    BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"

    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Huihui Qwen3 VL 4B Abliterated GGUF",
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
        model_kwargs["gguf_file"] = self._GGUF_FILES[self._variant]
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained(self.BASE_MODEL)
        self.processor.image_processor.min_pixels = self.min_pixels
        self.processor.image_processor.max_pixels = self.max_pixels

        # Register qwen3vl GGUF architecture before loading
        _register_qwen3vl_gguf_architecture()

        # Load config from the base model (has config.json with correct nested
        # text_config / vision_config). Passing config explicitly skips GGUF
        # config parsing which would map flat fields to the wrong struct level.
        config = Qwen3VLConfig.from_pretrained(self.BASE_MODEL)

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name,
            config=config,
            ignore_mismatched_sizes=True,
            **model_kwargs,
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
