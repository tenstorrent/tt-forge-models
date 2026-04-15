# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nomic Embed Text v2 GGUF model loader implementation for sentence embedding generation.

The nomic-bert-moe GGUF architecture is not supported by transformers' GGUF
loader, so this module manually reads the GGUF file and maps tensors to the
HuggingFace NomicBert model.
"""
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Nomic Embed Text v2 GGUF model variants for embedding generation."""

    NOMIC_EMBED_TEXT_V2_GGUF = "nomic-embed-text-v2-gguf"


class ModelLoader(ForgeModel):
    """Nomic Embed Text v2 GGUF model loader implementation for sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.NOMIC_EMBED_TEXT_V2_GGUF: ModelConfig(
            pretrained_model_name="ggml-org/Nomic-Embed-Text-V2-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NOMIC_EMBED_TEXT_V2_GGUF

    GGUF_FILE = "nomic-embed-text-v2-moe-q8_0.gguf"

    sample_sentences = [
        "search_document: TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten"
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Nomic-Embed-Text-v2-GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    @staticmethod
    def _load_gguf_weights(gguf_path):
        """Read and dequantize all tensors from a GGUF file.

        Returns a dict mapping GGUF tensor names to dequantized numpy arrays.
        Quantized tensors (e.g. Q8_0) are converted to float32.
        """
        import gguf

        reader = gguf.GGUFReader(gguf_path, "r")
        tensors = {}
        for tensor in reader.tensors:
            tensors[tensor.name] = gguf.dequantize(tensor.data, tensor.tensor_type)
        return tensors

    @staticmethod
    def _map_gguf_to_hf(gguf_tensors, num_layers=12, moe_every_n=2):
        """Map dequantized GGUF tensors to HuggingFace NomicBert parameter names.

        After dequantization the numpy arrays are already in PyTorch-compatible
        layout (row-major), so 2-D weights do not need transposing.  MoE expert
        weights arrive as (num_experts, out, in) and are reshaped to the
        megablocks stacked format (num_experts * intermediate, hidden).  The
        down-projection experts need a per-expert transpose because GGUF stores
        them as (E, hidden, intermediate) while megablocks expects
        (E * intermediate, hidden).
        """
        state_dict = {}

        # Embeddings
        state_dict["embeddings.word_embeddings.weight"] = torch.from_numpy(
            gguf_tensors["token_embd.weight"].copy()
        )
        state_dict["embeddings.token_type_embeddings.weight"] = torch.from_numpy(
            gguf_tensors["token_types.weight"].reshape(1, -1).copy()
        )
        state_dict["emb_ln.weight"] = torch.from_numpy(
            gguf_tensors["token_embd_norm.weight"].copy()
        )
        state_dict["emb_ln.bias"] = torch.from_numpy(
            gguf_tensors["token_embd_norm.bias"].copy()
        )

        for i in range(num_layers):
            prefix = f"blk.{i}"
            hf_prefix = f"encoder.layers.{i}"
            is_moe = (i % moe_every_n) != 0

            # Attention (dequantized shapes already match HF convention)
            state_dict[f"{hf_prefix}.attn.Wqkv.weight"] = torch.from_numpy(
                gguf_tensors[f"{prefix}.attn_qkv.weight"].copy()
            )
            state_dict[f"{hf_prefix}.attn.Wqkv.bias"] = torch.from_numpy(
                gguf_tensors[f"{prefix}.attn_qkv.bias"].copy()
            )
            state_dict[f"{hf_prefix}.attn.out_proj.weight"] = torch.from_numpy(
                gguf_tensors[f"{prefix}.attn_output.weight"].copy()
            )
            state_dict[f"{hf_prefix}.attn.out_proj.bias"] = torch.from_numpy(
                gguf_tensors[f"{prefix}.attn_output.bias"].copy()
            )

            # Layer norms
            state_dict[f"{hf_prefix}.norm1.weight"] = torch.from_numpy(
                gguf_tensors[f"{prefix}.attn_output_norm.weight"].copy()
            )
            state_dict[f"{hf_prefix}.norm1.bias"] = torch.from_numpy(
                gguf_tensors[f"{prefix}.attn_output_norm.bias"].copy()
            )
            state_dict[f"{hf_prefix}.norm2.weight"] = torch.from_numpy(
                gguf_tensors[f"{prefix}.layer_output_norm.weight"].copy()
            )
            state_dict[f"{hf_prefix}.norm2.bias"] = torch.from_numpy(
                gguf_tensors[f"{prefix}.layer_output_norm.bias"].copy()
            )

            if is_moe:
                # Router: dequantized (E, hidden) matches HF (E, hidden)
                state_dict[f"{hf_prefix}.mlp.router.layer.weight"] = torch.from_numpy(
                    gguf_tensors[f"{prefix}.ffn_gate_inp.weight"].copy()
                )

                # Up experts: dequantized (E, dff, hidden) -> reshape (E*dff, hidden)
                up = gguf_tensors[f"{prefix}.ffn_up_exps.weight"]
                state_dict[f"{hf_prefix}.mlp.experts.mlp.w1"] = torch.from_numpy(
                    up.reshape(-1, up.shape[-1]).copy()
                )

                # Down experts: dequantized (E, hidden, dff) -> transpose to
                # (E, dff, hidden) -> reshape (E*dff, hidden)
                down = gguf_tensors[f"{prefix}.ffn_down_exps.weight"]
                down_t = np.transpose(down, (0, 2, 1))
                state_dict[f"{hf_prefix}.mlp.experts.mlp.w2"] = torch.from_numpy(
                    down_t.reshape(-1, down_t.shape[-1]).copy()
                )

                # Shared bias (not in GGUF, initialize to zeros)
                hidden_size = gguf_tensors[f"{prefix}.attn_output.bias"].shape[0]
                state_dict[f"{hf_prefix}.mlp.experts.bias"] = torch.zeros(hidden_size)
            else:
                # Dense FFN (dequantized shapes already match HF convention)
                state_dict[f"{hf_prefix}.mlp.fc1.weight"] = torch.from_numpy(
                    gguf_tensors[f"{prefix}.ffn_up.weight"].copy()
                )
                state_dict[f"{hf_prefix}.mlp.fc1.bias"] = torch.from_numpy(
                    gguf_tensors[f"{prefix}.ffn_up.bias"].copy()
                )
                state_dict[f"{hf_prefix}.mlp.fc2.weight"] = torch.from_numpy(
                    gguf_tensors[f"{prefix}.ffn_down.weight"].copy()
                )
                state_dict[f"{hf_prefix}.mlp.fc2.bias"] = torch.from_numpy(
                    gguf_tensors[f"{prefix}.ffn_down.bias"].copy()
                )

        return state_dict

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        # The nomic-bert-moe GGUF architecture is not supported by transformers,
        # so we load the model structure from the original (non-GGUF) config and
        # populate weights manually from the GGUF file.
        base_model = "nomic-ai/nomic-embed-text-v2-moe-unsupervised"
        config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)

        model = AutoModel.from_config(config, trust_remote_code=True)

        gguf_path = hf_hub_download(pretrained_model_name, filename=self.GGUF_FILE)
        gguf_tensors = self._load_gguf_weights(gguf_path)
        state_dict = self._map_gguf_to_hf(
            gguf_tensors,
            num_layers=config.n_layer,
            moe_every_n=config.moe_every_n_layers,
        )

        # strict=False because the GGUF file does not contain pooler weights
        model.load_state_dict(state_dict, strict=False)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_sentences,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs

    def output_postprocess(self, output, inputs=None):
        if inputs is None:
            inputs = self.load_inputs()

        attention_mask = inputs["attention_mask"]

        if isinstance(output, (tuple, list)):
            token_embeddings = output[0]
        elif hasattr(output, "last_hidden_state"):
            token_embeddings = output.last_hidden_state
        else:
            token_embeddings = output

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sentence_embeddings

    def decode_output(self, outputs, inputs=None):
        return self.output_postprocess(outputs, inputs=inputs)

    def unpack_forward_output(self, fwd_output):
        tensors = []

        if hasattr(fwd_output, "last_hidden_state"):
            tensors.append(fwd_output.last_hidden_state.flatten())
        if (
            hasattr(fwd_output, "pooler_output")
            and fwd_output.pooler_output is not None
        ):
            tensors.append(fwd_output.pooler_output.flatten())

        if tensors:
            return torch.cat(tensors, dim=0)
        return fwd_output
