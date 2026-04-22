# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mradermacher Qwen3-VL 4B Instruct Abliterated GGUF model loader
implementation for image to text.

The GGUF file uses the "qwen3vl" architecture, which transformers does not
support natively.  We bridge this by:
1. Adding "qwen3vl" to GGUF_SUPPORTED_ARCHITECTURES with qwen3 config mappings.
2. Patching load_gguf_checkpoint so that:
   - During config loading (return_tensors=False): model_type "qwen3vl" is
     converted to "qwen3" so AutoConfig returns a flat Qwen3Config we can
     use as text_config.
   - During weight loading (return_tensors=True): the Qwen3VLConfig is
     temporarily made to look like a qwen3 config so get_gguf_hf_weights_map
     can map GGUF tensor names (blk.N.*, token_embd.*) to the HF model
     parameters under model.language_model.*.
3. Building a full Qwen3VLConfig in load_model by combining the GGUF text
   config with the vision config from the base Qwen/Qwen3-VL-4B-Instruct
   model, then passing it to from_pretrained.
"""

import importlib.metadata

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
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

_BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def _refresh_gguf_detection():
    """Refresh transformers' gguf package detection if installed after import."""
    from transformers.utils import import_utils

    if "gguf" not in import_utils.PACKAGE_DISTRIBUTION_MAPPING:
        import_utils.PACKAGE_DISTRIBUTION_MAPPING = (
            importlib.metadata.packages_distributions()
        )
        import_utils.is_gguf_available.cache_clear()


def _patch_qwen3vl_gguf():
    """Patch transformers to add qwen3vl GGUF architecture support.

    The Qwen3-VL GGUF files report architecture "qwen3vl", which transformers
    does not recognise.  This patch registers "qwen3vl" with the same config
    field mappings as "qwen3" (the text backbone is identical) and intercepts
    load_gguf_checkpoint to:
      - Convert model_type "qwen3vl" → "qwen3" during config loading so that
        AutoConfig can return a Qwen3Config with the correct text-backbone
        parameters.
      - Temporarily masquerade Qwen3VLConfig as qwen3 during weight loading so
        that get_gguf_hf_weights_map resolves the qwen3 tensor names
        (token_embd.*, blk.N.*) to the model.language_model.* parameters via
        the recursive sub-module scan.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )

    if getattr(gguf_utils, "_qwen3vl_patched", False):
        return

    if "qwen3vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    if "qwen3vl" not in GGUF_TO_TRANSFORMERS_MAPPING.get("config", {}):
        GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vl"] = dict(
            GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3"]
        )

    _orig = gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(*args, **kwargs):
        hf_model = kwargs.get("model_to_load")
        if hf_model is not None:
            cfg = hf_model.config
            if getattr(cfg, "model_type", None) == "qwen3_vl":
                # Weight-loading phase: temporarily pretend this is a plain
                # qwen3 config so get_gguf_hf_weights_map can use the qwen3
                # tensor name map for the text-backbone parameters.
                orig_cls_model_type = type(cfg).model_type
                num_hidden_layers = cfg.text_config.num_hidden_layers
                type(cfg).model_type = "qwen3"
                cfg.num_hidden_layers = num_hidden_layers
                try:
                    return _orig(*args, **kwargs)
                finally:
                    type(cfg).model_type = orig_cls_model_type
                    del cfg.num_hidden_layers
        result = _orig(*args, **kwargs)
        # Config-loading phase: convert model_type "qwen3vl" → "qwen3" so
        # AutoConfig builds a Qwen3Config with the correct text-backbone fields.
        if result.get("config", {}).get("model_type") == "qwen3vl":
            result["config"]["model_type"] = "qwen3"
        return result

    gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    gguf_utils._qwen3vl_patched = True

    import transformers.modeling_utils as modeling_utils
    import transformers.configuration_utils as config_utils

    for mod in (modeling_utils, config_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = _patched_load_gguf_checkpoint


_patch_qwen3vl_gguf()


class ModelVariant(StrEnum):
    """Available Mradermacher Qwen3-VL 4B Abliterated GGUF variants for image to text."""

    QWEN3_VL_4B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF = "4B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Mradermacher Qwen3-VL 4B Abliterated GGUF loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_VL_4B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3-VL-4B-Instruct-abliterated-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_VL_4B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF

    _GGUF_FILES = {
        ModelVariant.QWEN3_VL_4B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF: "Qwen3-VL-4B-Instruct-abliterated.Q4_K_M.gguf",
    }

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Mradermacher Qwen3-VL 4B Abliterated GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @property
    def _gguf_file(self):
        return self._GGUF_FILES.get(self._variant)

    def _build_full_config(self):
        """Build a Qwen3VLConfig by combining GGUF text config with base vision config."""
        _refresh_gguf_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name
        # After our patch, model_type "qwen3vl" → "qwen3", so AutoConfig returns Qwen3Config
        text_config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=self._gguf_file
        )
        base_config = AutoConfig.from_pretrained(_BASE_MODEL)
        return Qwen3VLConfig(
            text_config=text_config.to_dict(),
            vision_config=base_config.vision_config.to_dict(),
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        _refresh_gguf_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self._gguf_file
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(_BASE_MODEL)

        config = self._build_full_config()
        model_kwargs["config"] = config

        model = Qwen3VLForConditionalGeneration.from_pretrained(
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
