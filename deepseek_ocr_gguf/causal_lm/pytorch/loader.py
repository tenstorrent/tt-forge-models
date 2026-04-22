# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek OCR GGUF model loader implementation for causal language modeling.

NexaAI's GGUF uses a non-standard layout (transformers-format tensor names,
no tokenizer data in the GGUF, deepseek_vl_v2 architecture unsupported by
transformers' GGUF loader).  We bypass the GGUF loader and instead:
  1. Load the tokenizer directly from the HuggingFace repo (tokenizer.json).
  2. Build the model from the config fields embedded in the GGUF.
  3. Load and dequantize weights from the GGUF via the `gguf` library.
"""
import torch
from transformers import AutoTokenizer
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
from .src.model import DeepseekOCRConfig, DeepseekOCRForCausalLM


class ModelVariant(StrEnum):
    """Available DeepSeek OCR GGUF model variants for causal language modeling."""

    DEEPSEEK_OCR_Q4_0 = "Q4_0"
    DEEPSEEK_OCR_CUDA_Q4_0 = "CUDA_Q4_0"


class ModelLoader(ForgeModel):
    """DeepSeek OCR GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_OCR_Q4_0: LLMModelConfig(
            pretrained_model_name="NexaAI/DeepSeek-OCR-GGUF",
            max_length=128,
        ),
        ModelVariant.DEEPSEEK_OCR_CUDA_Q4_0: LLMModelConfig(
            pretrained_model_name="NexaAI/DeepSeek-OCR-GGUF-CUDA",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_OCR_Q4_0

    GGUF_FILE = "DeepSeek-OCR.Q4_0.gguf"

    sample_text = "What is your favorite city?"

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
            model="DeepSeek OCR GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        # NexaAI repos store tokenizer.json separately from the GGUF file.
        # Loading via gguf_file= fails because the GGUF has no tokenizer data
        # and the deepseek_vl_v2 architecture is unsupported by transformers.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def _build_config_from_gguf(self) -> DeepseekOCRConfig:
        """Read scalar config fields directly from the GGUF metadata."""
        from huggingface_hub import hf_hub_download
        from gguf import GGUFReader
        from transformers.modeling_gguf_pytorch_utils import _gguf_parse_value

        gguf_path = hf_hub_download(
            self._variant_config.pretrained_model_name, self.GGUF_FILE
        )
        reader = GGUFReader(gguf_path)
        raw = {}
        for key, field in reader.fields.items():
            try:
                raw[key] = _gguf_parse_value(field.parts[field.data[0]], field.types)
            except Exception:
                pass

        num_layers = (
            self.num_layers
            if self.num_layers is not None
            else int(raw.get("num_hidden_layers", 12))
        )
        return DeepseekOCRConfig(
            vocab_size=int(raw.get("vocab_size", 129280)),
            hidden_size=int(raw.get("hidden_size", 1280)),
            intermediate_size=int(raw.get("intermediate_size", 6848)),
            moe_intermediate_size=int(raw.get("moe_intermediate_size", 896)),
            num_hidden_layers=num_layers,
            num_attention_heads=int(raw.get("num_attention_heads", 10)),
            num_key_value_heads=int(raw.get("num_key_value_heads", 10)),
            head_dim=int(raw.get("head_dim", 128)),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            max_position_embeddings=int(raw.get("max_position_embeddings", 8192)),
            n_shared_experts=int(raw.get("n_shared_experts", 2)),
            n_routed_experts=int(raw.get("n_routed_experts", 64)),
            num_experts_per_tok=int(raw.get("num_experts_per_tok", 6)),
            moe_layer_freq=int(raw.get("moe_layer_freq", 1)),
            first_k_dense_replace=int(raw.get("first_k_dense_replace", 1)),
        )

    def _load_weights_from_gguf(
        self, model: DeepseekOCRForCausalLM, dtype: torch.dtype
    ):
        """Dequantize GGUF tensors and load them into the model."""
        import numpy as np
        from huggingface_hub import hf_hub_download
        from gguf import GGUFReader, dequantize

        gguf_path = hf_hub_download(
            self._variant_config.pretrained_model_name, self.GGUF_FILE
        )
        reader = GGUFReader(gguf_path)

        state = model.state_dict()
        loaded = set()
        for tensor in reader.tensors:
            name = tensor.name
            if name not in state:
                continue
            weights = dequantize(tensor.data, tensor.tensor_type)
            t = torch.from_numpy(np.copy(weights))
            if dtype is not None:
                t = t.to(dtype)
            state[name] = t
            loaded.add(name)

        model.load_state_dict(state, strict=False)

    def load_model(self, *, dtype_override=None, **kwargs):
        cfg = self._build_config_from_gguf()
        self.config = cfg

        model = DeepseekOCRForCausalLM(cfg)

        self._load_weights_from_gguf(model, dtype_override)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        if self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": self.sample_text}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = self.sample_text
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        return inputs

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.layers:
            if hasattr(layer.mlp, "gate_proj"):
                shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
            if hasattr(layer.mlp, "gate_proj_experts"):
                shard_specs[layer.mlp.gate_proj_experts] = (None, "model", "batch")
                shard_specs[layer.mlp.up_proj_experts] = (None, "model", "batch")
                shard_specs[layer.mlp.down_proj_experts] = (None, "batch", "model")
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        return shard_specs

    def load_config(self):
        if self.config is None:
            self.config = self._build_config_from_gguf()
        return self.config
