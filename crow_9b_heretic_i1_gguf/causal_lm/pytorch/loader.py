# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Crow 9B HERETIC GGUF model loader implementation for causal language modeling.

The Crow-9B-HERETIC-i1 uses the qwen35 GGUF architecture: a hybrid GatedDeltaNet
(linear attention / SSM) + full-attention model. Every 4th layer is full attention;
the rest are GatedDeltaNet layers. This corresponds to Qwen3Next in transformers.
"""
import os
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, Qwen3NextConfig, Qwen3NextForCausalLM
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
    """Available Crow 9B HERETIC GGUF model variants for causal language modeling."""

    CROW_9B_HERETIC_I1_GGUF = "9B_HERETIC_I1_GGUF"


class ModelLoader(ForgeModel):
    """Crow 9B HERETIC GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.CROW_9B_HERETIC_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Crow-9B-HERETIC-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CROW_9B_HERETIC_I1_GGUF

    GGUF_FILE = "Crow-9B-HERETIC.i1-Q4_K_M.gguf"

    # Qwen35 GGUF uses full attention every 4 layers (layers 3,7,11,...,31 are full)
    FULL_ATTENTION_INTERVAL = 4

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
            model="Crow 9B HERETIC GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_gguf_path(self):
        """Download and return path to the GGUF file."""
        return hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=self.GGUF_FILE,
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

    @staticmethod
    def _read_gguf_field(fields, name, default=None):
        """Read a scalar or list value from a GGUF fields dict."""
        if name not in fields:
            return default
        field = fields[name]
        values = [field.parts[idx].tolist() for idx in field.data]
        if len(values) == 1:
            v = values[0]
            return v[0] if isinstance(v, list) and len(v) == 1 else v
        return values

    @classmethod
    def _build_qwen3next_config(cls, gguf_path, dtype=None):
        """Build Qwen3NextConfig from a qwen35 GGUF file."""
        from gguf import GGUFReader

        reader = GGUFReader(gguf_path)
        fields = reader.fields

        def rf(name, default=None):
            return cls._read_gguf_field(fields, name, default)

        num_layers = rf("qwen35.block_count")
        full_attn_interval = rf("qwen35.full_attention_interval", cls.FULL_ATTENTION_INTERVAL)

        layer_types = [
            "full_attention" if (i + 1) % full_attn_interval == 0 else "linear_attention"
            for i in range(num_layers)
        ]

        # Derive SSM geometry from tensor shapes:
        #   ssm_a: (num_v_heads,) = (32,)
        #   ssm_norm: (head_v_dim,) = (128,)
        #   ssm_conv1d: (conv_dim, conv_kernel) → conv_dim = key_dim*2 + value_dim
        #     value_dim = num_v_heads * head_v_dim = 32*128 = 4096
        #     key_dim = (conv_dim - value_dim) / 2 = (8192-4096)/2 = 2048
        #   linear_num_k_heads from ssm.group_count = 16
        #   linear_key_head_dim = key_dim / num_k_heads = 2048/16 = 128
        ssm_group_count = rf("qwen35.ssm.group_count", 16)
        ssm_conv_kernel = rf("qwen35.ssm.conv_kernel", 4)
        num_v_heads = rf("qwen35.ssm.time_step_rank", 32)
        head_v_dim = 128   # from ssm_norm.weight shape
        value_dim = num_v_heads * head_v_dim  # 32 * 128 = 4096
        conv_dim = 8192  # from ssm_conv1d.weight numpy shape[0]
        key_dim = (conv_dim - value_dim) // 2  # 2048
        head_k_dim = key_dim // ssm_group_count  # 128

        # Read vocab_size from the actual embedding tensor shape; the GGUF metadata
        # field may report the base Qwen3 vocab (151936) even when the model has
        # an extended vocabulary (e.g. 248320 for Crow-9B-HERETIC).
        vocab_size = None
        for tensor in reader.tensors:
            if tensor.name == "token_embd.weight":
                vocab_size = tensor.shape[-1]  # stored as (hidden_size, vocab_size) in GGUF
                break
        if vocab_size is None:
            vocab_size = rf("qwen35.vocab_size") or rf("tokenizer.ggml.token_count") or 151936

        cfg = Qwen3NextConfig(
            vocab_size=vocab_size,
            hidden_size=rf("qwen35.embedding_length"),
            intermediate_size=rf("qwen35.feed_forward_length"),
            num_hidden_layers=num_layers,
            num_attention_heads=rf("qwen35.attention.head_count"),
            num_key_value_heads=rf("qwen35.attention.head_count_kv"),
            head_dim=rf("qwen35.attention.key_length"),
            rms_norm_eps=rf("qwen35.attention.layer_norm_rms_epsilon"),
            rope_theta=rf("qwen35.rope.freq_base"),
            max_position_embeddings=rf("qwen35.context_length"),
            layer_types=layer_types,
            linear_conv_kernel_dim=ssm_conv_kernel,
            linear_num_key_heads=ssm_group_count,
            linear_num_value_heads=num_v_heads,
            linear_key_head_dim=head_k_dim,
            linear_value_head_dim=head_v_dim,
            num_experts=0,          # dense FFN, not MoE
            decoder_sparse_step=1,  # unused when num_experts=0
            torch_dtype=dtype,
        )
        return cfg

    @classmethod
    def _load_state_dict_from_gguf(cls, gguf_path, layer_types, dtype=None):
        """Read GGUF tensors and map them to Qwen3NextForCausalLM parameter names.

        GGUF stores weights in (n_in, n_out) order in the shape metadata but
        dequantize() returns numpy arrays in (n_out, n_in) order, matching
        PyTorch's nn.Linear weight convention. No transposing needed for linear
        weights. Conv1d weights need unsqueeze(1) to add the groups dimension.
        """
        from gguf import GGUFReader, dequantize

        reader = GGUFReader(gguf_path)

        raw = {}
        for tensor in reader.tensors:
            weights = dequantize(tensor.data, tensor.tensor_type)
            t = torch.from_numpy(np.copy(weights))
            if dtype is not None:
                t = t.to(dtype)
            raw[tensor.name] = t

        sd = {}

        # Global tensors
        for gguf_name, hf_name in [
            ("token_embd.weight", "model.embed_tokens.weight"),
            ("output_norm.weight", "model.norm.weight"),
            ("output.weight", "lm_head.weight"),
        ]:
            if gguf_name in raw:
                sd[hf_name] = raw[gguf_name]

        for layer_idx, layer_type in enumerate(layer_types):
            p = f"blk.{layer_idx}"
            h = f"model.layers.{layer_idx}"

            # Layer norms shared by both layer types
            if f"{p}.attn_norm.weight" in raw:
                sd[f"{h}.input_layernorm.weight"] = raw[f"{p}.attn_norm.weight"]
            if f"{p}.post_attention_norm.weight" in raw:
                sd[f"{h}.post_attention_layernorm.weight"] = raw[f"{p}.post_attention_norm.weight"]

            # Dense FFN shared by both layer types
            for gguf_sfx, hf_sfx in [
                ("ffn_gate.weight", "mlp.gate_proj.weight"),
                ("ffn_up.weight",   "mlp.up_proj.weight"),
                ("ffn_down.weight", "mlp.down_proj.weight"),
            ]:
                if f"{p}.{gguf_sfx}" in raw:
                    sd[f"{h}.{hf_sfx}"] = raw[f"{p}.{gguf_sfx}"]

            if layer_type == "linear_attention":
                la = f"{h}.linear_attn"

                # in_proj_qkvz = cat([attn_qkv (QKV), attn_gate (Z)], dim=0)
                # attn_qkv numpy: (key_dim*2+value_dim, hidden) = (8192, 4096)
                # attn_gate numpy: (value_dim, hidden) = (4096, 4096)
                # in_proj_qkvz: (key_dim*2+value_dim*2, hidden) = (12288, 4096)
                qkv = raw.get(f"{p}.attn_qkv.weight")
                z = raw.get(f"{p}.attn_gate.weight")
                if qkv is not None and z is not None:
                    sd[f"{la}.in_proj_qkvz.weight"] = torch.cat([qkv, z], dim=0)

                # in_proj_ba = cat([ssm_beta (b), ssm_alpha (a)], dim=0)
                # Each has numpy shape: (num_v_heads, hidden) = (32, 4096)
                # in_proj_ba: (num_v_heads*2, hidden) = (64, 4096)
                alpha = raw.get(f"{p}.ssm_alpha.weight")
                beta = raw.get(f"{p}.ssm_beta.weight")
                if alpha is not None and beta is not None:
                    sd[f"{la}.in_proj_ba.weight"] = torch.cat([beta, alpha], dim=0)

                # Scalar / 1-D SSM state tensors
                if f"{p}.ssm_a" in raw:
                    sd[f"{la}.A_log"] = raw[f"{p}.ssm_a"]
                if f"{p}.ssm_dt.bias" in raw:
                    sd[f"{la}.dt_bias"] = raw[f"{p}.ssm_dt.bias"]
                if f"{p}.ssm_norm.weight" in raw:
                    sd[f"{la}.norm.weight"] = raw[f"{p}.ssm_norm.weight"]

                # conv1d: numpy (conv_dim, conv_kernel) → (conv_dim, 1, conv_kernel)
                if f"{p}.ssm_conv1d.weight" in raw:
                    sd[f"{la}.conv1d.weight"] = raw[f"{p}.ssm_conv1d.weight"].unsqueeze(1)

                # out_proj
                if f"{p}.ssm_out.weight" in raw:
                    sd[f"{la}.out_proj.weight"] = raw[f"{p}.ssm_out.weight"]

            elif layer_type == "full_attention":
                sa = f"{h}.self_attn"

                for gguf_sfx, hf_sfx in [
                    ("attn_q.weight",      "q_proj.weight"),
                    ("attn_k.weight",      "k_proj.weight"),
                    ("attn_v.weight",      "v_proj.weight"),
                    ("attn_output.weight", "o_proj.weight"),
                    ("attn_q_norm.weight", "q_norm.weight"),
                    ("attn_k_norm.weight", "k_norm.weight"),
                ]:
                    if f"{p}.{gguf_sfx}" in raw:
                        sd[f"{sa}.{hf_sfx}"] = raw[f"{p}.{gguf_sfx}"]

        return sd

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        gguf_path = self._get_gguf_path()

        cfg = self._build_qwen3next_config(gguf_path, dtype=dtype_override)
        self.config = cfg

        layer_types = cfg.layer_types
        state_dict = self._load_state_dict_from_gguf(gguf_path, layer_types, dtype=dtype_override)

        model = Qwen3NextForCausalLM(cfg)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            import warnings
            warnings.warn(f"Missing keys when loading Crow-9B-HERETIC GGUF: {missing[:5]}...")
        model.eval()

        self.model = model
        return model

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

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            if hasattr(layer, "self_attn"):
                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            if hasattr(layer, "linear_attn"):
                shard_specs[layer.linear_attn.out_proj.weight] = ("batch", "model")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
        return shard_specs

    def load_config(self):
        gguf_path = self._get_gguf_path()
        self.config = self._build_qwen3next_config(gguf_path)
        return self.config
