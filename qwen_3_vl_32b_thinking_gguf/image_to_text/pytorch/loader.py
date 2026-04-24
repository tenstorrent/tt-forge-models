# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 VL 32B Thinking GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLConfig,
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


class ModelVariant(StrEnum):
    """Available Qwen 3 VL 32B Thinking GGUF model variants for image to text."""

    QWEN_3_VL_32B_THINKING_1M_GGUF = "32b_thinking_1m_gguf"
    QWEN_3_VL_32B_THINKING_GGUF = "32b_thinking_gguf"


class ModelLoader(ForgeModel):
    """Qwen 3 VL 32B Thinking GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_VL_32B_THINKING_1M_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/Qwen3-VL-32B-Thinking-1M-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_32B_THINKING_GGUF: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-32B-Thinking-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_VL_32B_THINKING_1M_GGUF

    _GGUF_FILES = {
        ModelVariant.QWEN_3_VL_32B_THINKING_1M_GGUF: "Qwen3-VL-32B-Thinking-1M-Q4_K_M.gguf",
        ModelVariant.QWEN_3_VL_32B_THINKING_GGUF: "Qwen3VL-32B-Thinking-Q4_K_M.gguf",
    }

    BASE_MODEL = "Qwen/Qwen3-VL-32B-Thinking"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 3 VL 32B Thinking GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _find_true_load_gguf_checkpoint(fn):
        """Traverse patch-chain to find the true transformers load_gguf_checkpoint."""
        visited = set()
        while id(fn) not in visited:
            visited.add(id(fn))
            if getattr(fn, "__qualname__", "") == "load_gguf_checkpoint":
                return fn
            nxt = None
            if hasattr(fn, "__code__") and fn.__code__.co_freevars and fn.__closure__:
                clos = dict(
                    zip(
                        fn.__code__.co_freevars,
                        [c.cell_contents for c in fn.__closure__],
                    )
                )
                nxt = next(
                    (
                        v
                        for k, v in clos.items()
                        if callable(v)
                        and any(x in k.lower() for x in ("orig_load", "orig_gguf"))
                    ),
                    None,
                )
            if nxt is None and hasattr(fn, "__code__") and hasattr(fn, "__globals__"):
                for name in fn.__code__.co_names:
                    if any(x in name.lower() for x in ("orig_load", "orig_gguf")):
                        v = fn.__globals__.get(name)
                        if callable(v):
                            nxt = v
                            break
            if nxt:
                fn = nxt
                continue
            break
        return fn

    @staticmethod
    def _make_qwen3vl_weight_map(hf_model):
        """Build GGUF→HF weight name mapping for Qwen3VL models.

        GGUF qwen3vl tensors use the standard language-model naming
        (blk.N.*, token_embd.weight, output.weight, output_norm.weight).
        The HF Qwen3VL model wraps the LM under model.language_model.*, so
        this function strips that prefix for the gguf-py name lookup and then
        maps each GGUF tensor name back to the full HF path.
        """
        from gguf import MODEL_ARCH, get_tensor_name_map

        n_layers = hf_model.config.text_config.num_hidden_layers
        name_map = get_tensor_name_map(MODEL_ARCH.QWEN3VL, n_layers)

        gguf_to_hf = {}
        LM_PREFIX = "model.language_model."
        for hf_name in hf_model.state_dict():
            if hf_name.startswith(LM_PREFIX):
                # 'model.language_model.layers.0.X' -> 'model.layers.0.X'
                lookup = "model." + hf_name[len(LM_PREFIX) :]
            elif hf_name == "lm_head.weight":
                lookup = hf_name
            else:
                continue  # vision weights are absent from the GGUF file

            if lookup.endswith(".weight"):
                base, suffix = lookup[:-7], ".weight"
            elif lookup.endswith(".bias"):
                base, suffix = lookup[:-5], ".bias"
            else:
                base, suffix = lookup, ""

            gguf_name = name_map.get_name(base)
            if gguf_name is not None:
                gguf_to_hf[gguf_name + suffix] = hf_name

        return gguf_to_hf

    def load_model(self, *, dtype_override=None, **kwargs):
        import transformers.modeling_gguf_pytorch_utils as _gguf_utils

        # transformers does not support qwen3vl GGUF architecture natively;
        # add it so load_gguf_checkpoint passes the architecture check and
        # can read the model config from the GGUF metadata.
        _qwen3_cfg = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"].get("qwen3", {})
        if "qwen3vl" not in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
            _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")
        _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"].setdefault(
            "qwen3vl", _qwen3_cfg
        )

        # Other loaders monkey-patch load_gguf_checkpoint with old signatures
        # that drop model_to_load. Restore the true function for this call.
        _saved_load = _gguf_utils.load_gguf_checkpoint
        _true_load = self._find_true_load_gguf_checkpoint(_saved_load)
        _gguf_utils.load_gguf_checkpoint = _true_load

        # get_gguf_hf_weights_map does not handle Qwen3VL's nested config
        # (no top-level num_hidden_layers) or the model.language_model.* prefix.
        # Replace it with a version that builds the correct GGUF→HF mapping.
        _saved_map_fn = _gguf_utils.get_gguf_hf_weights_map

        def _patched_map_fn(hf_model, processor, model_type=None, **kw):
            if (
                model_type is None and hasattr(hf_model, "config")
            ) or model_type == "qwen3_vl":
                mt = model_type or getattr(hf_model.config, "model_type", None)
                if mt == "qwen3_vl":
                    return self._make_qwen3vl_weight_map(hf_model)
            return _saved_map_fn(hf_model, processor, model_type=model_type, **kw)

        _gguf_utils.get_gguf_hf_weights_map = _patched_map_fn

        try:
            pretrained_model_name = self._variant_config.pretrained_model_name

            model_kwargs = {}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs["gguf_file"] = self._GGUF_FILES[self._variant]
            model_kwargs |= kwargs

            self.processor = AutoProcessor.from_pretrained(self.BASE_MODEL)

            # Supply the base-model config so Qwen3VLConfig is properly nested
            # (the GGUF metadata only provides flat LM parameters and would
            # leave vision dimensions at their defaults, causing size mismatches).
            model_kwargs["config"] = Qwen3VLConfig.from_pretrained(self.BASE_MODEL)

            model = Qwen3VLForConditionalGeneration.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
            model.eval()
        finally:
            _gguf_utils.load_gguf_checkpoint = _saved_load
            _gguf_utils.get_gguf_hf_weights_map = _saved_map_fn

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
