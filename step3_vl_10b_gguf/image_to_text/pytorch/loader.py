# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Step3 VL 10B GGUF model loader implementation for image to text.
"""
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
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


def _patch_step3_gguf_loading():
    """Patch transformers GGUF weight mapper to handle step_robotics model type.

    Step3-VL uses Qwen3 as its LM backbone, so its GGUF weight names follow
    the qwen3 convention. The mapper only needs to know to use qwen3 arch.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if getattr(gguf_utils, "_step3_patched", False):
        return

    _orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        effective_type = (
            model_type
            if model_type is not None
            else getattr(getattr(hf_model, "config", None), "model_type", None)
        )
        if effective_type == "step_robotics":
            model_type = "qwen3"
        return _orig_get_map(
            hf_model,
            processor,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    gguf_utils.get_gguf_hf_weights_map = patched_get_map
    import transformers.modeling_utils as modeling_utils

    if hasattr(modeling_utils, "get_gguf_hf_weights_map"):
        modeling_utils.get_gguf_hf_weights_map = patched_get_map
    gguf_utils._step3_patched = True


_patch_step3_gguf_loading()


class ModelVariant(StrEnum):
    """Available Step3 VL 10B GGUF model variants for image to text."""

    STEP3_VL_10B_GGUF_Q4_K_M = "10b_gguf_q4_k_m"


class ModelLoader(ForgeModel):
    """Step3 VL 10B GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.STEP3_VL_10B_GGUF_Q4_K_M: LLMModelConfig(
            pretrained_model_name="seanbailey518/Step3-VL-10B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.STEP3_VL_10B_GGUF_Q4_K_M

    _GGUF_FILES = {
        ModelVariant.STEP3_VL_10B_GGUF_Q4_K_M: "Step3-VL-10B-Q4_K_M.gguf",
    }

    # Processor is loaded from the original non-GGUF repo since the GGUF repo
    # only contains quantized model weights without tokenizer/processor configs.
    _PROCESSOR_SOURCE = "stepfun-ai/Step3-VL-10B"

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )
    sample_text = "Describe this image."

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
            model="Step3 VL 10B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._PROCESSOR_SOURCE, trust_remote_code=True
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Step3 VL 10B GGUF model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Step3 VL 10B GGUF model instance for image to text.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Load config and model class from the original non-GGUF repo.
        # The GGUF file metadata reports Qwen3ForCausalLM which doesn't match the
        # VL model class (Step3VL10BForCausalLM). The GGUF repo also lacks the
        # remote code files, so we import the model class from the source repo.
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        config = AutoConfig.from_pretrained(
            self._PROCESSOR_SOURCE,
            trust_remote_code=True,
        )

        if self.num_layers is not None:
            config.text_config.num_hidden_layers = self.num_layers

        # GGUF weight mapping expects num_hidden_layers on the top-level config.
        if not hasattr(config, "num_hidden_layers") and hasattr(config, "text_config"):
            config.num_hidden_layers = config.text_config.num_hidden_layers

        ModelClass = get_class_from_dynamic_module(
            "modeling_step_vl.Step3VL10BForCausalLM",
            self._PROCESSOR_SOURCE,
        )

        model_kwargs = {"config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self._gguf_file

        if self.processor is None:
            self._load_processor()

        model = ModelClass.from_pretrained(
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
        ).eval()

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Step3 VL 10B GGUF model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.sample_image,
                    },
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
        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._PROCESSOR_SOURCE,
            trust_remote_code=True,
        )
        return self.config
