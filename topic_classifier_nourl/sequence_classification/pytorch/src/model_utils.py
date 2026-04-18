# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import glob
import shutil
import sys
from pathlib import Path


def patch_sdpa_attention_source(pretrained_model_name):
    """Patch cached HuggingFace modeling.py for SDPA compat with TT compiler.

    Fixes two issues in NewSdpaAttention._attention:
    1. Casts attn_mask to query dtype (float64 -> bfloat16 mismatch on XLA)
    2. Expands attn_mask dim 2 to match query sequence length (TT-MLIR requires it)

    torch.compile/dynamo traces source bytecode, so must patch on disk.
    """
    from transformers.dynamic_module_utils import HF_MODULES_CACHE

    org, model = pretrained_model_name.split("/")
    safe_model = model.replace("-", "_hyphen_")
    pattern = str(
        Path(HF_MODULES_CACHE)
        / "transformers_modules"
        / org
        / safe_model
        / "*"
        / "modeling.py"
    )
    matches = glob.glob(pattern)
    if not matches:
        return

    old = (
        "    def _attention(self, query_states, key_states, value_states, attention_bias, head_mask):\n"
        "        attn_output = torch.nn.functional.scaled_dot_product_attention(\n"
    )
    new = (
        "    def _attention(self, query_states, key_states, value_states, attention_bias, head_mask):\n"
        "        if attention_bias is not None and torch.is_tensor(attention_bias):\n"
        "            attention_bias = attention_bias.to(query_states.dtype)\n"
        "            seq_len = query_states.size(1)\n"
        "            if attention_bias.size(-2) == 1 and seq_len > 1:\n"
        "                attention_bias = attention_bias.expand(-1, -1, seq_len, -1)\n"
        "        attn_output = torch.nn.functional.scaled_dot_product_attention(\n"
    )

    patched = False
    for source_file in matches:
        source_path = Path(source_file)
        content = source_path.read_text()
        if old in content:
            content = content.replace(old, new)
            source_path.write_text(content)
            pycache = source_path.parent / "__pycache__"
            if pycache.exists():
                shutil.rmtree(pycache)
            patched = True

    if patched:
        modules_to_remove = [
            key
            for key in sys.modules
            if "transformers_modules" in key and safe_model in key
        ]
        for key in modules_to_remove:
            del sys.modules[key]
