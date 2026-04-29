# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Deepseek-Coder-V2-Lite-13B-Instruct-sft-s1K i1 GGUF model loader implementation for causal language modeling.
"""
from contextlib import contextmanager
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def _find_real_load_gguf_checkpoint():
    """Return the true load_gguf_checkpoint from modeling_gguf_pytorch_utils.

    Multiple GGUF loaders wrap load_gguf_checkpoint at import time, some using
    module-level globals and others using closure cells.  Some wrappers (e.g.
    qwen_3_5_imatrix_gguf, dmind_3_mini_i1_gguf) drop the model_to_load kwarg,
    which causes a TypeError when transformers 5.x calls load_gguf_checkpoint(…,
    model_to_load=…).

    Each patcher saves the previous function under a name like
    `_orig_load_gguf_checkpoint` (in globals) or `orig_load` (in a closure).
    The saved function's __name__ is always 'load_gguf_checkpoint' (the def-name).
    We traverse the wrapper chain by following that name — not by checking
    __module__, because later patchers import an already-patched version and the
    __module__ of the saved reference is not transformers.modeling_gguf_pytorch_utils.
    We stop when we reach a function whose own __module__ IS the target module.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    target_module = "transformers.modeling_gguf_pytorch_utils"
    fn = gguf_utils.load_gguf_checkpoint
    seen = set()

    for _ in range(30):
        if id(fn) in seen:
            break
        seen.add(id(fn))
        if getattr(fn, "__module__", "") == target_module:
            return fn

        found = None

        # Scan globals for any callable related to load_gguf_checkpoint.
        # Patchers store the previous function under names like
        # `_orig_load_gguf_checkpoint` (def-name: "load_gguf_checkpoint") or
        # by capturing another patcher (def-name: "_patched_load_gguf_checkpoint").
        # We use substring matching on def-name so we follow through all levels.
        exact = None
        fuzzy = None
        try:
            for val in list(fn.__globals__.values()):
                if not callable(val) or val is fn:
                    continue
                vmod = getattr(val, "__module__", "")
                vname = getattr(val, "__name__", "")
                if "load_gguf_checkpoint" in vname:
                    if vmod == target_module:
                        exact = val
                        break
                    if fuzzy is None:
                        fuzzy = val
        except RuntimeError:
            pass
        found = exact or fuzzy

        if found is not None:
            fn = found
            continue

        # Scan closure cells for the same pattern.
        for cell in getattr(fn, "__closure__", None) or ():
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if not callable(val):
                continue
            vmod = getattr(val, "__module__", "")
            vname = getattr(val, "__name__", "")
            if vmod == target_module:
                found = val
                break
            if "load_gguf_checkpoint" in vname:
                found = val
                break
        if found is not None:
            fn = found
            continue

        break  # No more candidates found

    return fn  # Best effort if traversal fails


def _register_deepseek2_gguf():
    """Register deepseek2 GGUF architecture support and return the correct loader.

    Transformers 5.x does not ship deepseek2 in GGUF_CONFIG_MAPPING or
    GGUF_TO_FAST_CONVERTERS.  The GGUF file uses "deepseek2" as its raw
    architecture key while the HF model_type is "deepseek_v2"; both keys
    must exist in GGUF_TO_FAST_CONVERTERS for convert_gguf_tokenizer to work.

    Also patches get_gguf_hf_weights_map to translate "deepseek_v2" →
    "deepseek2" so gguf-py's MODEL_ARCH_NAMES lookup succeeds during tensor
    weight mapping (gguf-py 0.18 only knows the key "deepseek2", not
    "deepseek_v2").

    Returns a load_gguf_checkpoint wrapper that:
      - accepts the model_to_load kwarg (transformers 5.x requirement),
      - remaps model_type deepseek2 → deepseek_v2 in the config dict.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFLlamaConverter,
    )

    if "deepseek2" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("deepseek2")
        GGUF_TO_TRANSFORMERS_MAPPING["config"]["deepseek2"] = {
            "context_length": "max_position_embeddings",
            "block_count": "num_hidden_layers",
            "feed_forward_length": "intermediate_size",
            "embedding_length": "hidden_size",
            "rope.freq_base": "rope_theta",
            "rope.dimension_count": "qk_rope_head_dim",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.layer_norm_rms_epsilon": "rms_norm_eps",
            "attention.key_length": None,
            "attention.value_length": None,
            "attention.key_length_mla": "qk_nope_head_dim",
            "attention.value_length_mla": "v_head_dim",
            "attention.q_lora_rank": "q_lora_rank",
            "attention.kv_lora_rank": "kv_lora_rank",
            "vocab_size": "vocab_size",
            "expert_count": "n_routed_experts",
            "expert_used_count": "num_experts_per_tok",
            "expert_shared_count": "n_shared_experts",
            "expert_group_count": "n_group",
            "expert_group_used_count": "topk_group",
            "expert_weights_scale": "routed_scaling_factor",
            "expert_weights_norm": "norm_topk_prob",
            "leading_dense_block_count": "first_k_dense_replace",
            "expert_feed_forward_length": "moe_intermediate_size",
        }

    # Register converters for both the raw GGUF key and the HF model_type.
    # The deepseek2 GGUF tokenizer uses tokenizer.ggml.model=gpt2 but stores
    # special-token IDs (bos=100000, eos=100001) within the vocabulary.
    # GGUFLlamaConverter correctly preserves those in-vocab IDs without adding
    # extra tokens beyond the 102400-entry embedding table.
    GGUF_TO_FAST_CONVERTERS.setdefault("deepseek2", GGUFLlamaConverter)
    GGUF_TO_FAST_CONVERTERS.setdefault("deepseek_v2", GGUFLlamaConverter)

    # Patch get_gguf_hf_weights_map to translate "deepseek_v2" → "deepseek2".
    # gguf-py's MODEL_ARCH_NAMES only contains "deepseek2" (the raw GGUF arch
    # key), not "deepseek_v2" (the HF model_type).  Without this remap the
    # weight-name mapping lookup raises NotImplementedError.
    _orig_get_weights_map = gguf_utils.get_gguf_hf_weights_map

    def _deepseek_get_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type == "deepseek_v2":
            model_type = "deepseek2"
        return _orig_get_weights_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = _deepseek_get_weights_map

    # Find the real unpatched load_gguf_checkpoint before building our wrapper.
    _real_load = _find_real_load_gguf_checkpoint()

    def _deepseek_load(gguf_path, return_tensors=False, model_to_load=None):
        result = _real_load(
            gguf_path, return_tensors=return_tensors, model_to_load=model_to_load
        )
        cfg = result.get("config", {})
        if cfg.get("model_type") == "deepseek2":
            cfg["model_type"] = "deepseek_v2"
            # head_count_kv=1 in GGUF is the MLA compressed rank (1 token per KV),
            # not the expanded KV head count.  MLA expands to num_attention_heads.
            cfg["num_key_value_heads"] = cfg.get("num_attention_heads", 16)
            # key_length_mla stores qk_nope_head_dim + qk_rope_head_dim (total key
            # head dim after MLA expansion).  Subtract the RoPE portion to get the
            # true qk_nope_head_dim that the model weights were built with.
            rope_dim = cfg.get("qk_rope_head_dim", 64)
            if "qk_nope_head_dim" in cfg:
                cfg["qk_nope_head_dim"] = cfg["qk_nope_head_dim"] - rope_dim
            # This GGUF stores a single fused Q matrix (attn_q.weight) rather than
            # the two LoRA factors (attn_q_a + attn_q_b).  Signal that by setting
            # q_lora_rank=None so transformers uses q_proj instead of q_a/q_b.
            cfg["q_lora_rank"] = None
        return result

    return _deepseek_load


_DEEPSEEK_LOAD_FN = _register_deepseek2_gguf()


@contextmanager
def _deepseek_gguf_ctx():
    """Temporarily install _DEEPSEEK_LOAD_FN at all binding sites.

    Other loaders (e.g. qwen_3_5_imatrix_gguf) overwrite the module-level
    load_gguf_checkpoint with versions that drop model_to_load.  This context
    manager ensures the correct function is active for the duration of a
    from_pretrained call.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.tokenization_utils_tokenizers as tok_utils

    modules = (gguf_utils, tok_auto, config_utils, tok_utils)
    saved = {
        mod: mod.load_gguf_checkpoint
        for mod in modules
        if hasattr(mod, "load_gguf_checkpoint")
    }
    for mod in saved:
        mod.load_gguf_checkpoint = _DEEPSEEK_LOAD_FN
    try:
        yield
    finally:
        for mod, fn in saved.items():
            mod.load_gguf_checkpoint = fn


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
    """Available Deepseek-Coder-V2-Lite-13B-Instruct-sft-s1K i1 GGUF model variants for causal language modeling."""

    DEEPSEEK_CODER_V2_LITE_13B_INSTRUCT_SFT_S1K_I1_Q4_K_M_GGUF = (
        "DEEPSEEK_CODER_V2_LITE_13B_INSTRUCT_SFT_S1K_I1_Q4_K_M_GGUF"
    )


class ModelLoader(ForgeModel):
    """Deepseek-Coder-V2-Lite-13B-Instruct-sft-s1K i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_CODER_V2_LITE_13B_INSTRUCT_SFT_S1K_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Deepseek-Coder-V2-Lite-13B-Instruct-sft-s1K-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = (
        ModelVariant.DEEPSEEK_CODER_V2_LITE_13B_INSTRUCT_SFT_S1K_I1_Q4_K_M_GGUF
    )

    GGUF_FILE = "Deepseek-Coder-V2-Lite-13B-Instruct-sft-s1K.i1-Q4_K_M.gguf"

    sample_text = "Write a bubble sort algorithm in Python."

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
            model="Deepseek-Coder-V2-Lite-13B-Instruct-sft-s1K i1 GGUF",
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

        with _deepseek_gguf_ctx():
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
            with _deepseek_gguf_ctx():
                config = AutoConfig.from_pretrained(
                    pretrained_model_name, gguf_file=self.GGUF_FILE
                )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        with _deepseek_gguf_ctx():
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()

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
        with _deepseek_gguf_ctx():
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
            )
        return self.config
