# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from gr00t n1.6-release for Python 3.12 / transformers 5.x compatibility.
import os

import torch
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

_EAGLE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "nvidia", "Eagle-Block2A-2B-v2"
)


class EagleBackboneN1d6(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "nvidia/Eagle-Block2A-2B-v2",
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = True,
        use_flash_attention: bool = False,
        projector_dim: int = -1,
        load_bf16: bool = False,
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,
        transformers_loading_kwargs: dict = {},
    ):
        super().__init__()

        extra_kwargs = {}
        # Use sdpa instead of flash_attention_2 if flash attention is not available
        try:
            import flash_attn  # noqa: F401

            if use_flash_attention:
                extra_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            extra_kwargs["attn_implementation"] = "sdpa"

        if load_bf16:
            extra_kwargs["torch_dtype"] = torch.bfloat16

        if model_name == "nvidia/Eagle-Block2A-2B-v2":
            config = AutoConfig.from_pretrained(
                _EAGLE_MODEL_PATH, trust_remote_code=True
            )
            # Explicitly override attn implementation on sub-configs so that
            # Siglip2VisionModel (which declares _supports_flash_attn_2=True)
            # doesn't auto-select flash_attention_2 in environments where
            # flash_attn is unavailable.
            attn_impl = extra_kwargs.get("attn_implementation", "sdpa")
            for sub_cfg_name in ("vision_config", "text_config"):
                sub_cfg = getattr(config, sub_cfg_name, None)
                if sub_cfg is not None:
                    sub_cfg._attn_implementation = attn_impl
                    sub_cfg._attn_implementation_autoset = False
            self.model = AutoModel.from_config(
                config, trust_remote_code=True, **extra_kwargs
            )
        else:
            raise ValueError(f"Model {model_name} not supported")

        while len(self.model.language_model.model.layers) > select_layer:
            self.model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual, tune_top_llm_layers)
        if load_bf16 and trainable_params_fp32:
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
            # lm_head must be fp32 to match hidden states produced by fp32 top layers
            lm_head = getattr(
                getattr(self.model, "language_model", None), "lm_head", None
            )
            if lm_head is not None:
                for p in lm_head.parameters():
                    p.data = p.data.to(torch.float32)

    def set_trainable_parameters(
        self, tune_llm: bool, tune_visual: bool, tune_top_llm_layers: int
    ):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.model.language_model.requires_grad_(False)
        if not tune_visual:
            self.model.vision_model.requires_grad_(False)
            self.model.mlp1.requires_grad_(False)
        if tune_top_llm_layers > 0:
            for layer in self.model.language_model.model.layers[-tune_top_llm_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def set_frozen_modules_to_eval_mode(self):
        if self.training:
            if self.model.language_model and not self.tune_llm:
                self.model.language_model.eval()
            if self.model.vision_model and not self.tune_visual:
                self.model.vision_model.eval()
                self.model.mlp1.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        keys_to_use = ["input_ids", "attention_mask", "pixel_values"]
        vl_input = {k: vl_input[k] for k in keys_to_use}
        # DEBUG: register hook on lm_head to capture hidden_states dtype
        lm_head = self.model.language_model.lm_head

        def _pre_hook(module, args):
            inp = args[0] if args else None
            print(
                f"[DEBUG lm_head] input={inp.dtype if inp is not None else None} "
                f"weight={module.weight.dtype}"
            )

        handle = lm_head.register_forward_pre_hook(_pre_hook)
        try:
            outputs = self.model(**vl_input, output_hidden_states=True)
        finally:
            handle.remove()

        outputs = outputs["hidden_states"][-1]
        image_mask = vl_input["input_ids"] == self.model.config.image_token_index
        attention_mask = vl_input["attention_mask"] == 1
        return BatchFeature(
            data={
                "backbone_features": outputs,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            }
        )
