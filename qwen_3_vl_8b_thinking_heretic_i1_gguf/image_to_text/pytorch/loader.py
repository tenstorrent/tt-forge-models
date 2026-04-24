# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 VL 8B Thinking Heretic i1 GGUF model loader implementation for
image to text.
"""

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
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
    """Available Qwen 3 VL 8B Thinking Heretic i1 GGUF variants for image to text."""

    QWEN_3_VL_8B_THINKING_HERETIC_I1_Q4_K_M_GGUF = "8b_thinking_heretic_i1_q4_k_m_gguf"


class ModelLoader(ForgeModel):
    """Qwen 3 VL 8B Thinking Heretic i1 GGUF loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_VL_8B_THINKING_HERETIC_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3-VL-8B-Thinking-heretic-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_VL_8B_THINKING_HERETIC_I1_Q4_K_M_GGUF

    GGUF_FILE = "Qwen3-VL-8B-Thinking-heretic.i1-Q4_K_M.gguf"

    # Base model for config and processor (GGUF repo has no config.json or processor files)
    BASE_MODEL = "Qwen/Qwen3-VL-8B-Thinking"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 3 VL 8B Thinking Heretic i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _apply_gguf_compat_patches():
        """Patch transformers GGUF utilities to support qwen3vl architecture.

        transformers 5.2 does not include qwen3vl in its GGUF architecture
        registry, but the gguf-py package (>=0.10.0) already has full tensor
        name mappings for it. This patch:
          1. Registers qwen3vl in GGUF_TO_TRANSFORMERS_MAPPING so the
             architecture check in load_gguf_checkpoint passes.
          2. Wraps get_gguf_hf_weights_map to translate the HF model_type
             "qwen3_vl" → "qwen3vl" (the gguf-py arch name) and to fix
             num_layers for VL models whose top-level config nests the layer
             count under text_config.
          3. Wraps is_gguf_available to handle InvalidVersion when gguf is
             installed at runtime and importlib.metadata cache is stale."""
        import importlib.util

        import transformers.modeling_gguf_pytorch_utils as _gguf_mod
        import transformers.utils.import_utils as _import_utils

        # --- patch 1: safe is_gguf_available ---
        _orig_is_gguf_available = _import_utils.is_gguf_available

        def _safe_is_gguf_available(min_version=None):
            try:
                if min_version is not None:
                    return _orig_is_gguf_available(min_version=min_version)
                return _orig_is_gguf_available()
            except Exception:
                return importlib.util.find_spec("gguf") is not None

        _import_utils.is_gguf_available = _safe_is_gguf_available
        _gguf_mod.is_gguf_available = _safe_is_gguf_available

        # --- patch 2: register qwen3vl in GGUF config mapping ---
        if "qwen3vl" not in _gguf_mod.GGUF_TO_TRANSFORMERS_MAPPING["config"]:
            qwen3_cfg = dict(
                _gguf_mod.GGUF_TO_TRANSFORMERS_MAPPING["config"].get("qwen3", {})
            )
            _gguf_mod.GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vl"] = qwen3_cfg
            _gguf_mod.GGUF_SUPPORTED_ARCHITECTURES = list(
                _gguf_mod.GGUF_TO_TRANSFORMERS_MAPPING["config"].keys()
            )

        # --- patch 3: fix get_gguf_hf_weights_map for qwen3_vl model type ---
        orig_get_map = _gguf_mod.get_gguf_hf_weights_map

        def _compat_get_map(
            hf_model, processor, model_type=None, num_layers=None, qual_name=""
        ):
            if hf_model is None:
                return {}
            # Translate HF model_type to gguf-py arch name for VL models
            if model_type is None and hasattr(hf_model, "config"):
                mt = getattr(hf_model.config, "model_type", None)
                if mt == "qwen3_vl":
                    model_type = "qwen3vl"
            # Fix num_layers for VL configs where it lives under text_config
            if num_layers is None and hasattr(hf_model, "config"):
                cfg = hf_model.config
                if not hasattr(cfg, "num_hidden_layers") and hasattr(
                    cfg, "text_config"
                ):
                    num_layers = getattr(cfg.text_config, "num_hidden_layers", None)
            return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

        _gguf_mod.get_gguf_hf_weights_map = _compat_get_map

    def load_model(self, *, dtype_override=None, **kwargs):
        self._apply_gguf_compat_patches()
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repos do not ship processor or config.json; load both from base model.
        self.processor = AutoProcessor.from_pretrained(self.BASE_MODEL)
        # Pass config explicitly so from_pretrained skips GGUF config parsing
        # (qwen3vl config keys map to Qwen3Config fields, not Qwen3VLConfig).
        config = AutoConfig.from_pretrained(self.BASE_MODEL)

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, config=config, **model_kwargs
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
