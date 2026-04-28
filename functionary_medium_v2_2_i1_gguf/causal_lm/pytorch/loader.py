# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Functionary Medium v2.2 i1 GGUF model loader implementation for causal language modeling.

functionary-medium-v2.2 is based on Mixtral-8x7B (8 experts, top-2 routing).
The GGUF file uses architecture="llama" with llama.expert_count=8, so transformers'
AutoModelForCausalLM.from_pretrained does not recognise it as a MoE model and
leaves all expert MLP weights randomly initialised.  We bypass AutoModel and load
the weights manually into MixtralForCausalLM.
"""
import re
import numpy as np
import torch
from transformers import AutoTokenizer, MixtralConfig, MixtralForCausalLM
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
    """Available Functionary Medium v2.2 i1 GGUF model variants for causal language modeling."""

    FUNCTIONARY_MEDIUM_V2_2_I1_Q4_K_M_GGUF = "v2.2_i1_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Functionary Medium v2.2 i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.FUNCTIONARY_MEDIUM_V2_2_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/functionary-medium-v2.2-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FUNCTIONARY_MEDIUM_V2_2_I1_Q4_K_M_GGUF

    GGUF_FILE = "functionary-medium-v2.2.i1-Q4_K_M.gguf"

    sample_text = "What is your favorite city?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Functionary Medium v2.2 i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {"gguf_file": self.GGUF_FILE}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    @staticmethod
    def _read_gguf_field(reader, name):
        """Return a scalar value from a GGUFReader field, or None if absent."""
        if name not in reader.fields:
            return None
        field = reader.fields[name]
        val = field.parts[-1].tolist()
        return val[0] if isinstance(val, list) and len(val) == 1 else val

    def _build_mixtral_config(self, reader, dtype):
        """Construct MixtralConfig directly from GGUF metadata."""
        rf = lambda name: self._read_gguf_field(reader, name)
        return MixtralConfig(
            hidden_size=rf("llama.embedding_length"),
            intermediate_size=rf("llama.feed_forward_length"),
            num_hidden_layers=rf("llama.block_count"),
            num_attention_heads=rf("llama.attention.head_count"),
            num_key_value_heads=rf("llama.attention.head_count_kv"),
            max_position_embeddings=rf("llama.context_length"),
            rms_norm_eps=rf("llama.attention.layer_norm_rms_epsilon"),
            rope_theta=rf("llama.rope.freq_base"),
            vocab_size=rf("llama.vocab_size"),
            num_local_experts=rf("llama.expert_count"),
            num_experts_per_tok=rf("llama.expert_used_count"),
            # Disable sliding-window attention; the GGUF does not specify one.
            sliding_window=None,
            torch_dtype=dtype,
        )

    def _load_weights_from_gguf(self, reader, model, dtype):
        """Read and dequantize GGUF tensors and populate model state dict.

        GGUF tensor naming conventions for Mixtral-style (llama arch + experts):
          - Global:  token_embd, output_norm, output
          - Per-layer scalar: blk.N.{attn_norm, ffn_norm, attn_q, attn_k,
                               attn_v, attn_output, ffn_gate_inp}
          - Expert stacks (shape [n_exp, d_out, d_in]):
              blk.N.ffn_gate_exps  → gate half of mlp.experts.gate_up_proj
              blk.N.ffn_up_exps   → up   half of mlp.experts.gate_up_proj
              blk.N.ffn_down_exps → mlp.experts.down_proj

        Transformers 5.x MixtralForCausalLM state-dict layout:
          model.layers.N.mlp.gate.weight           [n_exp, hidden]
          model.layers.N.mlp.experts.gate_up_proj  [n_exp, ffn*2, hidden]
          model.layers.N.mlp.experts.down_proj     [n_exp, hidden, ffn]
        """
        from gguf import dequantize

        SIMPLE = {
            "token_embd.weight": "model.embed_tokens.weight",
            "output_norm.weight": "model.norm.weight",
            "output.weight": "lm_head.weight",
        }
        PER_LAYER = {
            "attn_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
            "attn_q.weight": "self_attn.q_proj.weight",
            "attn_k.weight": "self_attn.k_proj.weight",
            "attn_v.weight": "self_attn.v_proj.weight",
            "attn_output.weight": "self_attn.o_proj.weight",
            "ffn_gate_inp.weight": "mlp.gate.weight",
        }

        # First pass: collect per-layer expert tensors keyed by layer index
        gate_exps: dict[str, torch.Tensor] = {}
        up_exps: dict[str, torch.Tensor] = {}
        down_exps: dict[str, torch.Tensor] = {}
        state_dict = {}

        for tensor in reader.tensors:
            w = torch.from_numpy(
                np.copy(dequantize(tensor.data, tensor.tensor_type))
            ).to(dtype)
            name = tensor.name

            if name in SIMPLE:
                state_dict[SIMPLE[name]] = w
                continue

            m = re.match(r"blk\.(\d+)\.(.+)", name)
            if not m:
                continue
            layer_n, suffix = m.group(1), m.group(2)
            prefix = f"model.layers.{layer_n}."

            if suffix in PER_LAYER:
                state_dict[prefix + PER_LAYER[suffix]] = w
            elif suffix == "ffn_gate_exps.weight":
                gate_exps[layer_n] = w
            elif suffix == "ffn_up_exps.weight":
                up_exps[layer_n] = w
            elif suffix == "ffn_down_exps.weight":
                down_exps[layer_n] = w

        # Second pass: combine gate+up and assign expert weights
        for layer_n in gate_exps:
            prefix = f"model.layers.{layer_n}."
            # gate_exps and up_exps are each [n_exp, ffn, hidden]; concatenate on dim=1
            state_dict[prefix + "mlp.experts.gate_up_proj"] = torch.cat(
                [gate_exps[layer_n], up_exps[layer_n]], dim=1
            )
            state_dict[prefix + "mlp.experts.down_proj"] = down_exps[layer_n]

        model.load_state_dict(state_dict, strict=True)

    def load_model(self, *, dtype_override=None, **kwargs):
        from gguf import GGUFReader
        from huggingface_hub import hf_hub_download

        pretrained_model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else torch.float32

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        gguf_path = hf_hub_download(pretrained_model_name, self.GGUF_FILE)
        reader = GGUFReader(gguf_path)

        config = self._build_mixtral_config(reader, dtype)
        model = MixtralForCausalLM(config).to(dtype)
        # Use dense batched expert dispatch to avoid nonzero()/for-loop that crashes XLA
        model.config._experts_implementation = "batched_mm"

        self._load_weights_from_gguf(reader, model, dtype)

        model = model.eval()
        self.config = model.config
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

    def load_config(self):
        from gguf import GGUFReader
        from huggingface_hub import hf_hub_download

        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_path = hf_hub_download(pretrained_model_name, self.GGUF_FILE)
        reader = GGUFReader(gguf_path)
        self.config = self._build_mixtral_config(reader, dtype=None)
        return self.config
