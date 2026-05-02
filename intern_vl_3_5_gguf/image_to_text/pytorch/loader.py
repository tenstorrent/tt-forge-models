# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InternVL3.5 GGUF model loader implementation for image to text.

InternVL3.5 GGUFs are split into two files:
  1. Main GGUF (qwen3 arch): language model weights
  2. mmproj GGUF (clip arch): vision encoder + multimodal projector weights

transformers has no built-in InternVL GGUF support so we load each GGUF
manually: the LLM via AutoModelForCausalLM and the vision encoder via
GGUFReader, then transplant both into InternVLForConditionalGeneration.
"""

import torch
import numpy as np
import huggingface_hub
from gguf import GGUFReader
from accelerate import init_empty_weights
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    InternVLForConditionalGeneration,
    InternVLConfig,
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
    """Available InternVL3.5 GGUF model variants for image to text."""

    INTERN_VL3_5_4B_Q4_K_M = "4b_q4_k_m"
    INTERN_VL3_5_4B_Q8_0 = "4b_q8_0"
    INTERN_VL3_5_14B_Q4_K_M = "14b_q4_k_m"
    INTERN_VL3_5_14B_Q8_0 = "14b_q8_0"


class ModelLoader(ForgeModel):
    """InternVL3.5 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.INTERN_VL3_5_4B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="bartowski/OpenGVLab_InternVL3_5-4B-GGUF",
            max_length=128,
        ),
        ModelVariant.INTERN_VL3_5_4B_Q8_0: LLMModelConfig(
            pretrained_model_name="bartowski/OpenGVLab_InternVL3_5-4B-GGUF",
            max_length=128,
        ),
        ModelVariant.INTERN_VL3_5_14B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="bartowski/OpenGVLab_InternVL3_5-14B-GGUF",
            max_length=128,
        ),
        ModelVariant.INTERN_VL3_5_14B_Q8_0: LLMModelConfig(
            pretrained_model_name="bartowski/OpenGVLab_InternVL3_5-14B-GGUF",
            max_length=128,
        ),
    }

    _GGUF_FILES = {
        ModelVariant.INTERN_VL3_5_4B_Q4_K_M: "OpenGVLab_InternVL3_5-4B-Q4_K_M.gguf",
        ModelVariant.INTERN_VL3_5_4B_Q8_0: "OpenGVLab_InternVL3_5-4B-Q8_0.gguf",
        ModelVariant.INTERN_VL3_5_14B_Q4_K_M: "OpenGVLab_InternVL3_5-14B-Q4_K_M.gguf",
        ModelVariant.INTERN_VL3_5_14B_Q8_0: "OpenGVLab_InternVL3_5-14B-Q8_0.gguf",
    }

    # Vision encoder + projector weights (stored as CLIP-format GGUF)
    _MMPROJ_FILES = {
        ModelVariant.INTERN_VL3_5_4B_Q4_K_M: "mmproj-OpenGVLab_InternVL3_5-4B-f16.gguf",
        ModelVariant.INTERN_VL3_5_4B_Q8_0: "mmproj-OpenGVLab_InternVL3_5-4B-f16.gguf",
        ModelVariant.INTERN_VL3_5_14B_Q4_K_M: "mmproj-OpenGVLab_InternVL3_5-14B-f16.gguf",
        ModelVariant.INTERN_VL3_5_14B_Q8_0: "mmproj-OpenGVLab_InternVL3_5-14B-f16.gguf",
    }

    _HF_PROCESSORS = {
        ModelVariant.INTERN_VL3_5_4B_Q4_K_M: "OpenGVLab/InternVL3_5-4B-HF",
        ModelVariant.INTERN_VL3_5_4B_Q8_0: "OpenGVLab/InternVL3_5-4B-HF",
        ModelVariant.INTERN_VL3_5_14B_Q4_K_M: "OpenGVLab/InternVL3_5-14B-HF",
        ModelVariant.INTERN_VL3_5_14B_Q8_0: "OpenGVLab/InternVL3_5-14B-HF",
    }

    DEFAULT_VARIANT = ModelVariant.INTERN_VL3_5_4B_Q4_K_M

    sample_image = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="InternVL3.5 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override or torch.bfloat16
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]
        mmproj_file = self._MMPROJ_FILES[self._variant]
        hf_ref = self._HF_PROCESSORS[self._variant]

        self.processor = AutoProcessor.from_pretrained(
            hf_ref,
            trust_remote_code=True,
        )

        # Load InternVL config from HF reference; disable KV cache for
        # single forward pass compilation.
        config = InternVLConfig.from_pretrained(hf_ref)
        config.text_config.use_cache = False

        # Load the LLM (Qwen3) from the main GGUF. The GGUF metadata
        # declares arch=qwen3 so from_pretrained resolves to Qwen3ForCausalLM.
        qwen3 = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            gguf_file=gguf_file,
            dtype=dtype,
        )

        # Remap Qwen3 state dict keys into InternVL language model namespace:
        #   model.<x>  →  model.language_model.<x>
        #   lm_head.*  →  lm_head.*  (unchanged)
        state_dict = {}
        for k, v in qwen3.state_dict().items():
            if k.startswith("model."):
                new_k = "model.language_model." + k[len("model."):]
            else:
                new_k = k
            state_dict[new_k] = v
        del qwen3

        # Download and parse the mmproj GGUF (F16 vision encoder + projector).
        mmproj_path = huggingface_hub.hf_hub_download(
            pretrained_model_name, mmproj_file
        )
        reader = GGUFReader(mmproj_path)
        gguf_tensors = {t.name: t for t in reader.tensors}

        def _to_tensor(name):
            arr = np.array(gguf_tensors[name].data)
            return torch.from_numpy(arr).to(dtype).contiguous()

        # Vision tower embeddings.
        # GGUFReader.tensor.data already returns numpy arrays in PyTorch
        # dimension order (t.shape is the GGUF-format reversed shape).
        state_dict["model.vision_tower.embeddings.cls_token"] = _to_tensor("v.class_embd")
        state_dict["model.vision_tower.embeddings.position_embeddings"] = _to_tensor(
            "v.position_embd.weight"
        )
        state_dict["model.vision_tower.embeddings.patch_embeddings.projection.weight"] = _to_tensor(
            "v.patch_embd.weight"
        )
        state_dict["model.vision_tower.embeddings.patch_embeddings.projection.bias"] = _to_tensor(
            "v.patch_embd.bias"
        )

        # Vision encoder transformer blocks
        blk_map = {
            "attn_q": "attention.q_proj",
            "attn_k": "attention.k_proj",
            "attn_v": "attention.v_proj",
            "attn_out": "attention.projection_layer",
            "ffn_up": "mlp.fc1",
            "ffn_down": "mlp.fc2",
            "ln1": "layernorm_before",
            "ln2": "layernorm_after",
        }
        # lambda_1/lambda_2 have no .weight suffix in HF but do in GGUF
        lambda_map = {
            "ls1": "lambda_1",
            "ls2": "lambda_2",
        }

        num_vis_layers = config.vision_config.num_hidden_layers
        for i in range(num_vis_layers):
            gguf_blk = f"v.blk.{i}."
            hf_layer = f"model.vision_tower.encoder.layer.{i}."
            for gguf_part, hf_part in blk_map.items():
                for suffix in (".weight", ".bias"):
                    g = f"{gguf_blk}{gguf_part}{suffix}"
                    if g in gguf_tensors:
                        state_dict[f"{hf_layer}{hf_part}{suffix}"] = _to_tensor(g)
            for gguf_part, hf_part in lambda_map.items():
                g = f"{gguf_blk}{gguf_part}.weight"
                if g in gguf_tensors:
                    state_dict[f"{hf_layer}{hf_part}"] = _to_tensor(g)

        # Multimodal projector  (MLP indices: 0=layer_norm, 1=linear_1, 3=linear_2)
        proj_map = {
            "mm.model.mlp.0.weight": "model.multi_modal_projector.layer_norm.weight",
            "mm.model.mlp.0.bias":   "model.multi_modal_projector.layer_norm.bias",
            "mm.model.mlp.1.weight": "model.multi_modal_projector.linear_1.weight",
            "mm.model.mlp.1.bias":   "model.multi_modal_projector.linear_1.bias",
            "mm.model.mlp.3.weight": "model.multi_modal_projector.linear_2.weight",
            "mm.model.mlp.3.bias":   "model.multi_modal_projector.linear_2.bias",
        }
        for g, h in proj_map.items():
            state_dict[h] = _to_tensor(g)

        # Build the full InternVL model and load the combined state dict.
        with init_empty_weights():
            model = InternVLForConditionalGeneration(config)
        model.load_state_dict(state_dict, strict=True, assign=True)
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
                    {"type": "text", "text": "What is shown in this image?"},
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
