# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Dolphin 2.7 Mixtral 8x7B GGUF model loader implementation for causal language modeling.
"""
import re

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
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
    """Available Dolphin 2.7 Mixtral 8x7B GGUF model variants for causal language modeling."""

    DOLPHIN_27_MIXTRAL_8X7B_GGUF = "2.7_Mixtral_8x7B_GGUF"


class ModelLoader(ForgeModel):
    """Dolphin 2.7 Mixtral 8x7B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DOLPHIN_27_MIXTRAL_8X7B_GGUF: LLMModelConfig(
            pretrained_model_name="TheBloke/dolphin-2.7-mixtral-8x7b-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DOLPHIN_27_MIXTRAL_8X7B_GGUF

    GGUF_FILE = "dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf"

    sample_text = "What is the meaning of life?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Dolphin 2.7 Mixtral 8x7B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # Transformers 5.2.0 changed MixtralExperts to use batched gate_up_proj /
        # down_proj, but the GGUF name map has no entry for per-expert tensors
        # (blk.N.ffn_gate.E.weight). Load them manually so expert weights are not
        # left at random initialization.
        gguf_path = self._resolve_gguf_cache_path()
        if gguf_path:
            self._load_mixtral_gguf_expert_weights(
                model, gguf_path, model.model.layers[0].mlp.experts.gate_up_proj.dtype
            )

        self.config = model.config
        self.model = model
        return model

    def _resolve_gguf_cache_path(self) -> str:
        try:
            from huggingface_hub import hf_hub_download

            return hf_hub_download(
                repo_id=self._variant_config.pretrained_model_name,
                filename=self.GGUF_FILE,
            )
        except Exception:
            return None

    @staticmethod
    def _load_mixtral_gguf_expert_weights(model, gguf_path: str, dtype: torch.dtype):
        """Populate MixtralExperts batched weights from per-expert GGUF tensors.

        The GGUF file stores one tensor per expert per layer (blk.N.ffn_gate.E.weight).
        Transformers 5.2.0 expects batched [E, 2*inter, H] and [E, H, inter] tensors,
        which the GGUF name map cannot assemble — so expert weights stay at their random
        init values.  This method reads and assembles them directly.
        """
        try:
            from gguf import GGUFReader, dequantize
        except ImportError:
            return

        reader = GGUFReader(gguf_path)
        tensor_by_name = {t.name: t for t in reader.tensors}

        # Only proceed if the GGUF uses the per-expert format
        per_expert_pattern = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)\.(\d+)\.weight")
        if not any(per_expert_pattern.match(name) for name in tensor_by_name):
            return

        num_layers = model.config.num_hidden_layers
        num_experts = model.config.num_local_experts

        for layer_idx in range(num_layers):
            gate_list, up_list, down_list = [], [], []
            for e in range(num_experts):
                gate_t = tensor_by_name.get(f"blk.{layer_idx}.ffn_gate.{e}.weight")
                up_t = tensor_by_name.get(f"blk.{layer_idx}.ffn_up.{e}.weight")
                down_t = tensor_by_name.get(f"blk.{layer_idx}.ffn_down.{e}.weight")
                if gate_t is None or up_t is None or down_t is None:
                    break
                gate_list.append(
                    torch.from_numpy(np.copy(dequantize(gate_t.data, gate_t.tensor_type)))
                )
                up_list.append(
                    torch.from_numpy(np.copy(dequantize(up_t.data, up_t.tensor_type)))
                )
                down_list.append(
                    torch.from_numpy(np.copy(dequantize(down_t.data, down_t.tensor_type)))
                )

            if len(gate_list) != num_experts:
                continue

            experts = model.model.layers[layer_idx].mlp.experts

            # gate_up_proj: [E, 2*inter, H] — gate in first half, up in second
            gate_up = torch.stack(
                [torch.cat([gate_list[e], up_list[e]], dim=0) for e in range(num_experts)]
            ).to(dtype)
            # down_proj: [E, H, inter]
            down = torch.stack(down_list).to(dtype)

            experts.gate_up_proj.data.copy_(gate_up)
            experts.down_proj.data.copy_(down)

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts = [text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
