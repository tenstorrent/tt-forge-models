# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UnifiedReward-Edit-qwen-72b i1 GGUF model loader implementation for image to text.

Note: The qwen2vl GGUF architecture is not yet supported by the transformers
GGUF loader. Since the text decoder of Qwen2.5-VL uses the same architecture
as Qwen2, we register qwen2vl as an alias and load the text decoder only.
"""

import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer
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


def _patch_qwen2vl_gguf_support():
    """Register qwen2vl GGUF architecture for transformers text model loading.

    Qwen2.5-VL GGUF files use the 'qwen2vl' architecture identifier.
    Transformers 5.x has no GGUF loading support for qwen2vl. The text decoder
    of Qwen2.5-VL is architecturally identical to Qwen2, so we register
    qwen2vl as an alias and remap the model_type to 'qwen2' after loading.
    """
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "qwen2vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen2vl"] = dict(
        GGUF_TO_TRANSFORMERS_MAPPING["config"].get("qwen2", {})
    )

    GGUF_TO_FAST_CONVERTERS.setdefault("qwen2vl", GGUF_TO_FAST_CONVERTERS["qwen2"])

    _orig_load_gguf_checkpoint = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = _orig_load_gguf_checkpoint(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen2vl":
            config["model_type"] = "qwen2"
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_qwen2vl_gguf_support()


class ModelVariant(StrEnum):
    """Available UnifiedReward-Edit-qwen-72b i1 GGUF model variants for image to text."""

    UNIFIED_REWARD_EDIT_QWEN_72B_I1_GGUF = "72b_i1_gguf"


class ModelLoader(ForgeModel):
    """UnifiedReward-Edit-qwen-72b i1 GGUF model loader implementation for image to text tasks.

    Loads the text decoder from the GGUF checkpoint using the qwen2 architecture
    mapping. Vision encoder tensors in the GGUF are not loaded since the text
    decoder alone is sufficient for compile-only testing.
    """

    _VARIANTS = {
        ModelVariant.UNIFIED_REWARD_EDIT_QWEN_72B_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/UnifiedReward-Edit-qwen-72b-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.UNIFIED_REWARD_EDIT_QWEN_72B_I1_GGUF

    GGUF_FILE = "UnifiedReward-Edit-qwen-72b.i1-Q4_K_M.gguf"

    BASE_MODEL = "CodeGoat24/UnifiedReward-Edit-qwen-72b"

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="UnifiedReward-Edit-qwen-72b i1 GGUF",
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
        model_kwargs["ignore_mismatched_sizes"] = True
        model_kwargs |= kwargs

        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = Qwen2ForCausalLM.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            [self.sample_text] * batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)[
                    :batch_size
                ]

        return inputs

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs
