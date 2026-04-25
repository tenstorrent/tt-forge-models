# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wizardeur Qwen3.5-9B GPTQ-marlin model loader implementation for image to text.
"""

import re
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    GPTQConfig,
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
    """Available Wizardeur Qwen3.5-9B GPTQ-marlin model variants for image to text."""

    QWEN3_5_9B_GPTQ_MARLIN = "9B_GPTQ_marlin"


class ModelLoader(ForgeModel):
    """Wizardeur Qwen3.5-9B GPTQ-marlin model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_5_9B_GPTQ_MARLIN: LLMModelConfig(
            pretrained_model_name="wizardeur/Qwen3.5-9B-GPTQ-marlin",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_5_9B_GPTQ_MARLIN

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Wizardeur Qwen3.5-9B GPTQ-marlin",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import Qwen3_5ForConditionalGeneration

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"device_map": "cpu"}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        # Load config and override GPTQ backend to "torch" to avoid the marlin
        # CPU transform bug in gptqmodel when running on CPU-only systems.
        config = AutoConfig.from_pretrained(pretrained_model_name)
        if hasattr(config, "quantization_config"):
            quant_dict = config.quantization_config
            if hasattr(quant_dict, "to_dict"):
                quant_dict = quant_dict.to_dict()
            quant_dict = dict(quant_dict)
            quant_dict["backend"] = "torch"
            model_kwargs["quantization_config"] = GPTQConfig.from_dict(quant_dict)

        # gptqmodel's TorchQuantLinear calls torch.compile(backend="inductor") in
        # post_init, which crashes on CPU-only torch builds. Patch optimize() to be
        # a no-op so the model uses the uncompiled dequantize_weight path.
        try:
            from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

            def _patched_optimize(self, backend=None, mode=None, fullgraph=False):
                self.optimized = True

            TorchQuantLinear.optimize = _patched_optimize
        except ImportError:
            pass

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        # Post-load surgery: gptqmodel 6.x wraps layers that were NOT quantized in
        # the checkpoint (e.g. in_proj_a, in_proj_b, lm_head) with TorchQuantLinear
        # or TorchFusedQuantLinear and randomly initializes their g_idx, causing
        # out-of-bounds index errors in transform_cpu/_build_ret_idx at compile time.
        # Use duck-typing to find all GPTQ-style layers (any class with g_idx,
        # qweight, scales) so we don't depend on a successful gptqmodel import here.
        self._fix_bad_quantized_layers(model, pretrained_model_name)

        return model

    def _fix_bad_quantized_layers(self, model, pretrained_model_name):
        """Replace GPTQ layers with invalid g_idx with plain nn.Linear.

        Uses duck-typing (checks for g_idx/qweight/scales attributes) so this
        works with any gptqmodel backend class (TorchQuantLinear,
        TorchFusedQuantLinear, etc.) without a class-specific import.
        """
        # Find all GPTQ-style layers with invalid (randomly initialized) g_idx.
        # A layer is "bad" if g_idx.max() >= number of quantization groups
        # (scales.shape[0]), which indicates the g_idx was never set from actual
        # quantization data.
        bad_layer_names = []
        for name, module in model.named_modules():
            if not (
                hasattr(module, "g_idx")
                and hasattr(module, "scales")
                and hasattr(module, "qweight")
            ):
                continue
            try:
                if module.g_idx.max().item() >= module.scales.shape[0]:
                    bad_layer_names.append(name)
            except Exception:
                bad_layer_names.append(name)

        if not bad_layer_names:
            return

        # Load all checkpoint shards to find full-precision weights for bad layers
        repo_files = list(list_repo_files(pretrained_model_name))
        shard_files = [f for f in repo_files if re.match(r".*\.safetensors$", f)]

        # Build a map from weight key suffix to tensor for all bad layers
        needed_suffixes = {name + ".weight" for name in bad_layer_names}
        # Also try bias
        needed_suffixes |= {name + ".bias" for name in bad_layer_names}

        weight_map = {}
        for shard_file in shard_files:
            local_path = hf_hub_download(pretrained_model_name, shard_file)
            shard = load_file(local_path, device="cpu")
            for key, tensor in shard.items():
                # Strip leading "model." prefix if present for matching
                for suffix in list(needed_suffixes):
                    if key.endswith(suffix) or key == suffix:
                        weight_map[suffix] = tensor
                        needed_suffixes.discard(suffix)
            if not needed_suffixes:
                break

        # Determine the model's floating-point activation dtype by finding a
        # known non-quantized parameter (e.g. embedding or layer-norm weight).
        act_dtype = torch.bfloat16
        for _, p in model.named_parameters():
            if p.dtype in (torch.float16, torch.bfloat16, torch.float32):
                act_dtype = p.dtype
                break

        # Replace bad GPTQ layers with nn.Linear
        for layer_name in bad_layer_names:
            weight_key = layer_name + ".weight"
            if weight_key not in weight_map:
                # Try with "model." prefix stripped from layer_name
                stripped = re.sub(r"^model\.", "", layer_name)
                weight_key_alt = stripped + ".weight"
                if weight_key_alt in weight_map:
                    weight_key = weight_key_alt
                else:
                    continue

            weight = weight_map[weight_key]
            bias_key = layer_name + ".bias"
            bias = weight_map.get(bias_key, None)

            # Get the parent module and attribute name
            parts = layer_name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr = parts[-1]

            old_module = getattr(parent, attr)
            out_features, in_features = weight.shape
            new_linear = nn.Linear(in_features, out_features, bias=bias is not None)
            new_linear.weight = nn.Parameter(weight.to(act_dtype))
            if bias is not None:
                new_linear.bias = nn.Parameter(bias.to(act_dtype))
            setattr(parent, attr, new_linear)

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
