# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Magistral Small MLX model loader implementation for causal language modeling.

lmstudio-community/Magistral-Small-2509-MLX-5bit stores weights as MLX affine
5-bit quantized tensors (uint32-packed, LSB-first, with per-group bf16 scales
and biases).  The top-level config has model_type='mistral3' (Mistral3Config),
which is not in AutoModelForCausalLM's registry, so we load the full
Mistral3ForConditionalGeneration model and dequantize weights manually.

Key-prefix remapping from safetensors → transformers state dict:
  language_model.model.*  → model.language_model.*
  language_model.lm_head.* → lm_head.*
  vision_tower.vision_model.* → model.vision_tower.*
  multi_modal_projector.*  → model.multi_modal_projector.*
"""
import json
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers.models.mistral3.modeling_mistral3 import (
    Mistral3ForConditionalGeneration,
)
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open
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
    """Available Magistral Small MLX model variants for causal language modeling."""

    SMALL_2509_MLX_5BIT = "Small_2509_MLX_5bit"


class ModelLoader(ForgeModel):
    """Magistral Small MLX model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.SMALL_2509_MLX_5BIT: LLMModelConfig(
            pretrained_model_name="lmstudio-community/Magistral-Small-2509-MLX-5bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMALL_2509_MLX_5BIT

    sample_text = "Give me a short introduction to large language model."

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
            model="Magistral Small MLX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    # ------------------------------------------------------------------
    # 5-bit MLX affine dequantization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_mlx_5bit(packed: torch.Tensor, in_features: int) -> torch.Tensor:
        """Unpack 5-bit MLX affine integers from a uint32 tensor.

        packed  : [out_f, ceil(in_features*5/32)] torch.uint32
        returns : [out_f, in_features] torch.int32

        MLX stores elements LSB-first, bit-packed consecutively across uint32
        boundaries: element i occupies bits [5i, 5i+5).
        """
        out_f = packed.shape[0]
        packed_i64 = packed.to(torch.int64)

        chunk = 128
        shifts = torch.arange(32, dtype=torch.int64, device=packed.device)
        weights = torch.tensor([1, 2, 4, 8, 16], dtype=torch.int32, device=packed.device)
        parts = []
        for s in range(0, out_f, chunk):
            e = min(s + chunk, out_f)
            c = packed_i64[s:e]                    # [c, pack_c]
            bits = (c.unsqueeze(-1) >> shifts) & 1  # [c, pack_c, 32]
            bits = bits.reshape(e - s, -1)          # [c, pack_c*32]
            bits = bits[:, : in_features * 5].reshape(e - s, in_features, 5)
            parts.append((bits.to(torch.int32) * weights).sum(-1))
        return torch.cat(parts, dim=0)

    @classmethod
    def _dequantize_shard(
        cls,
        raw: dict,
        group_size: int,
        target_dtype: torch.dtype,
    ) -> dict:
        """Return a new state dict with quantized weights dequantized to target_dtype."""
        # Pre-identify which base keys are quantized so we can skip their auxiliaries.
        quant_bases = {
            key[: -len(".weight")]
            for key, t in raw.items()
            if key.endswith(".weight") and t.dtype == torch.uint32
        }
        aux_keys = {
            base + suffix
            for base in quant_bases
            for suffix in (".scales", ".biases")
        }

        out = {}
        for key, tensor in raw.items():
            if key in aux_keys:
                continue
            if key.endswith(".weight") and tensor.dtype == torch.uint32:
                base = key[: -len(".weight")]
                skey = base + ".scales"
                bkey = base + ".biases"
                scales = raw[skey].to(torch.float32)  # [out_f, G]
                biases = raw[bkey].to(torch.float32)  # [out_f, G]
                out_f = tensor.shape[0]
                in_f = scales.shape[1] * group_size

                w_int = cls._unpack_mlx_5bit(tensor, in_f)  # [out_f, in_f]
                sc = scales.unsqueeze(-1).expand(-1, -1, group_size).reshape(out_f, in_f)
                bi = biases.unsqueeze(-1).expand(-1, -1, group_size).reshape(out_f, in_f)
                w = w_int.to(torch.float32) * sc + bi
                out[key] = w.to(target_dtype)
            else:
                if tensor.is_floating_point():
                    out[key] = tensor.to(target_dtype)
                else:
                    out[key] = tensor
        return out

    @staticmethod
    def _remap_key(key: str) -> str:
        """Remap an MLX safetensors key to the transformers model state dict key."""
        if key.startswith("language_model.model."):
            return "model.language_model." + key[len("language_model.model."):]
        if key.startswith("language_model.lm_head."):
            return "lm_head." + key[len("language_model.lm_head."):]
        if key.startswith("vision_tower.vision_model."):
            return "model.vision_tower." + key[len("vision_tower.vision_model."):]
        if key.startswith("multi_modal_projector."):
            return "model.multi_modal_projector." + key[len("multi_modal_projector."):]
        return key

    # ------------------------------------------------------------------

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        target_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.tokenizer is None:
            self._load_tokenizer()

        # Load config, strip quantization_config so transformers doesn't
        # try to parse the {bits:5, mode:affine} entry.
        cfg = AutoConfig.from_pretrained(pretrained_model_name)
        cfg.quantization_config = None

        if self.num_layers is not None:
            cfg.text_config.num_hidden_layers = self.num_layers

        model = Mistral3ForConditionalGeneration(cfg).to(target_dtype)

        quant_cfg = {"group_size": 64, "bits": 5}

        # Discover actual shard filenames (the index may reference a different
        # shard count than what is actually uploaded to the repo).
        shard_files = sorted(
            f
            for f in list_repo_files(pretrained_model_name)
            if f.endswith(".safetensors") and not f.endswith(".index.json")
        )

        for shard_name in shard_files:
            path = hf_hub_download(pretrained_model_name, shard_name)
            raw = {}
            with safe_open(path, framework="pt") as f:
                for k in f.keys():
                    raw[k] = f.get_tensor(k)

            deq = self._dequantize_shard(raw, quant_cfg["group_size"], target_dtype)
            remapped = {self._remap_key(k): v for k, v in deq.items()}
            model.load_state_dict(remapped, strict=False)

        model.tie_weights()
        model.eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        if self.tokenizer.chat_template is not None:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = self.sample_text
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
            self._variant_config.pretrained_model_name
        )
        return self.config
