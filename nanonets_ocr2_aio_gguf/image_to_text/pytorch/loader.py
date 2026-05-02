# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
prithivMLmods/Nanonets-OCR2-3B-AIO-GGUF model loader implementation for image to text.
"""
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
)
from transformers import modeling_gguf_pytorch_utils as _gguf_utils
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS
from typing import Optional
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _qwen2_5_vl_module

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

# ── Bug 1: qwen2vl GGUF architecture not registered in transformers 5.x ───────
# The GGUF file has general.architecture = "qwen2vl" but GGUF_SUPPORTED_ARCHITECTURES
# only contains qwen2/qwen2_moe/qwen3/qwen3_moe.  Register it using qwen2 as template.
def _register_qwen2vl_gguf_arch():
    if "qwen2vl" not in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
        _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        mapping = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]
        if isinstance(mapping, dict) and "qwen2" in mapping:
            mapping.setdefault("qwen2vl", mapping["qwen2"])
    GGUF_TO_FAST_CONVERTERS.setdefault("qwen2vl", GGUF_TO_FAST_CONVERTERS["qwen2"])


_register_qwen2vl_gguf_arch()

# ── Bug 2: qwen2_5_vl model_type not translated to qwen2vl for gguf-py lookup ─
# get_gguf_hf_weights_map reads model.config.model_type = "qwen2_5_vl" but
# gguf-py MODEL_ARCH_NAMES only has "qwen2vl".  Add the missing translation.
_orig_get_gguf_hf_weights_map = _gguf_utils.get_gguf_hf_weights_map


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    if model_type is None and hf_model is not None:
        cfg = getattr(hf_model, "config", None)
        if cfg is not None:
            model_type = getattr(cfg, "model_type", None)
    if model_type in ("qwen2_5_vl", "qwen2_vl"):
        model_type = "qwen2vl"
    return _orig_get_gguf_hf_weights_map(hf_model, processor, model_type, num_layers, qual_name)


_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


# ── Bug 3: .tolist() on TT device tensors ─────────────────────────────────────
# The test runner moves all inputs including image_grid_thw to TT device.
# VisionEncoder methods (rot_pos_emb, get_window_index) call .tolist() on grid_thw
# for Python control flow. TT device does not support eager tensor reads and
# .tolist() fails with Error code: 13. Also get_rope_index calls input_ids.tolist()
# and get_image_features calls image_grid_thw.prod(-1).tolist().
# Fix: move metadata tensors to CPU before .tolist() calls; main tensor computations
# (pixel_values, hidden_states, etc.) remain on TT device.
_orig_rot_pos_emb = _qwen2_5_vl_module.Qwen2_5_VLVisionTransformer.rot_pos_emb
_orig_get_window_index = _qwen2_5_vl_module.Qwen2_5_VLVisionTransformer.get_window_index
_orig_get_rope_index = _qwen2_5_vl_module.Qwen2_5_VLForConditionalGeneration.get_rope_index
_orig_get_image_features = _qwen2_5_vl_module.Qwen2_5_VLForConditionalGeneration.get_image_features


def _patched_rot_pos_emb(self, grid_thw):
    return _orig_rot_pos_emb(self, grid_thw.cpu())


def _patched_get_window_index(self, grid_thw):
    return _orig_get_window_index(self, grid_thw.cpu())


def _patched_get_rope_index(
    self,
    input_ids=None,
    image_grid_thw=None,
    video_grid_thw=None,
    second_per_grid_ts=None,
    attention_mask=None,
    **kwargs,
):
    return _orig_get_rope_index(
        self,
        input_ids=input_ids.cpu() if input_ids is not None else None,
        image_grid_thw=image_grid_thw.cpu() if image_grid_thw is not None else None,
        video_grid_thw=video_grid_thw.cpu() if video_grid_thw is not None else None,
        second_per_grid_ts=second_per_grid_ts.cpu() if second_per_grid_ts is not None else None,
        attention_mask=attention_mask.cpu() if attention_mask is not None else None,
        **kwargs,
    )


def _patched_get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
    return _orig_get_image_features(
        self,
        pixel_values,
        image_grid_thw=image_grid_thw.cpu() if image_grid_thw is not None else None,
        **kwargs,
    )


_qwen2_5_vl_module.Qwen2_5_VLVisionTransformer.rot_pos_emb = _patched_rot_pos_emb
_qwen2_5_vl_module.Qwen2_5_VLVisionTransformer.get_window_index = _patched_get_window_index
_qwen2_5_vl_module.Qwen2_5_VLForConditionalGeneration.get_rope_index = _patched_get_rope_index
_qwen2_5_vl_module.Qwen2_5_VLForConditionalGeneration.get_image_features = _patched_get_image_features


class ModelVariant(StrEnum):
    """Available Nanonets OCR2 AIO GGUF model variants for image to text."""

    NANONETS_OCR2_3B_AIO_GGUF = "3B_AIO_GGUF"


class ModelLoader(ForgeModel):
    """Nanonets OCR2 AIO GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.NANONETS_OCR2_3B_AIO_GGUF: LLMModelConfig(
            pretrained_model_name="prithivMLmods/Nanonets-OCR2-3B-AIO-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NANONETS_OCR2_3B_AIO_GGUF

    _GGUF_FILES = {
        ModelVariant.NANONETS_OCR2_3B_AIO_GGUF: "Nanonets-OCR2-3B.Q4_K_M.gguf",
    }

    # Base model provides processor config and full model config (vision + text).
    # The GGUF repo ships only text weights; vision encoder is randomly initialized.
    _BASE_MODEL = "nanonets/Nanonets-OCR2-3B"

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
            model="Nanonets OCR2 AIO GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Nanonets OCR2 AIO GGUF model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Nanonets OCR2 AIO GGUF model instance for image to text.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Load processor from the base model: the GGUF repo has no preprocessor_config.json.
        # use_fast=False avoids the transformers 5.x fast-processor default breaking change.
        self.processor = AutoProcessor.from_pretrained(
            self._BASE_MODEL, use_fast=False
        )

        # Load config from the base model so from_pretrained gets a complete qwen2_5_vl
        # config rather than the incomplete config the GGUF would produce.
        config = AutoConfig.from_pretrained(self._BASE_MODEL)
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
            config.text_config.num_hidden_layers = self.num_layers

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self._gguf_file
        model_kwargs["config"] = config

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Nanonets OCR2 AIO GGUF model.

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
                    {"type": "text", "text": "Convert the document to markdown."},
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
        self.config = AutoConfig.from_pretrained(self._BASE_MODEL)
        return self.config
