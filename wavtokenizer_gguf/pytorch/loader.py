# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
WavTokenizer GGUF model loader for discrete audio codec tokenization.

Loads the GGUF-converted WavTokenizer decoder hosted at ggml-org/WavTokenizer,
which is a GGUF packaging of novateur/WavTokenizer-large-speech-75token (a
VQ-VAE style audio codec operating at 24 kHz with 75 tokens per second).

Requires the WavTokenizer repository to be cloned at /tmp/wavtokenizer_repo.
"""

import os
import re
import sys
from typing import Optional

import torch

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

WAVTOKENIZER_REPO_PATH = "/tmp/wavtokenizer_repo"


def _ensure_wavtokenizer_importable():
    """Ensure the WavTokenizer repo is cloned and importable."""
    if not os.path.isdir(WAVTOKENIZER_REPO_PATH):
        import subprocess

        subprocess.check_call(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "https://github.com/jishengpeng/WavTokenizer.git",
                WAVTOKENIZER_REPO_PATH,
            ]
        )

    if WAVTOKENIZER_REPO_PATH not in sys.path:
        sys.path.insert(0, WAVTOKENIZER_REPO_PATH)


def _map_gguf_tensor(name, data):
    """Map a GGUF tensor name and data to a PyTorch state dict (key, tensor) pair.

    GGUFReader already reverses the GGUF axis convention so tensors arrive in
    PyTorch order.  The only adjustments needed are:
      - squeeze the trailing size-1 dim from biases stored as (C, 1)
      - broadcast AdaLayerNorm scale/shift from (C,) to (4, C)
    """

    def _sq(t):
        return t.squeeze(-1)

    def _ada(t):
        return t.unsqueeze(0).expand(4, -1).contiguous()

    # ConvNeXt blocks: convnext.N.*
    m = re.match(r"^convnext\.(\d+)\.(.+)$", name)
    if m:
        n, sub = m.group(1), m.group(2)
        if sub == "gamma.weight":
            return f"backbone.convnext.{n}.gamma", data
        if sub == "dw.weight":
            return f"backbone.convnext.{n}.dwconv.weight", data
        if sub == "dw.bias":
            return f"backbone.convnext.{n}.dwconv.bias", _sq(data)
        if sub == "norm.weight":
            return f"backbone.convnext.{n}.norm.scale.weight", _ada(data)
        if sub == "norm.bias":
            return f"backbone.convnext.{n}.norm.shift.weight", _ada(data)
        if sub == "pw1.weight":
            return f"backbone.convnext.{n}.pwconv1.weight", data
        if sub == "pw1.bias":
            return f"backbone.convnext.{n}.pwconv1.bias", data
        if sub == "pw2.weight":
            return f"backbone.convnext.{n}.pwconv2.weight", data
        if sub == "pw2.bias":
            return f"backbone.convnext.{n}.pwconv2.bias", data

    # Embed Conv1d
    if name == "conv1d.weight":
        return "backbone.embed.weight", data
    if name == "conv1d.bias":
        return "backbone.embed.bias", _sq(data)

    # VQ codebook (feature_extractor loaded separately; only embed is in GGUF)
    if name == "token_embd.weight":
        return "feature_extractor.encodec.quantizer.vq.layers.0._codebook.embed", data

    # Final layer norm
    if name == "output_norm.weight":
        return "backbone.final_layer_norm.weight", data
    if name == "output_norm.bias":
        return "backbone.final_layer_norm.bias", data

    # backbone.norm AdaLayerNorm (token_embd_norm in GGUF)
    if name == "token_embd_norm.weight":
        return "backbone.norm.scale.weight", _ada(data)
    if name == "token_embd_norm.bias":
        return "backbone.norm.shift.weight", _ada(data)

    # pos_net blocks
    m = re.match(r"^posnet\.(\d+)\.(.+)$", name)
    if m:
        n, sub = int(m.group(1)), m.group(2)
        if n in (0, 1, 3, 4):  # ResnetBlock
            if sub == "conv1.weight":
                return f"backbone.pos_net.{n}.conv1.weight", data
            if sub == "conv1.bias":
                return f"backbone.pos_net.{n}.conv1.bias", _sq(data)
            if sub == "conv2.weight":
                return f"backbone.pos_net.{n}.conv2.weight", data
            if sub == "conv2.bias":
                return f"backbone.pos_net.{n}.conv2.bias", _sq(data)
            if sub == "norm1.weight":
                return f"backbone.pos_net.{n}.norm1.weight", _sq(data)
            if sub == "norm1.bias":
                return f"backbone.pos_net.{n}.norm1.bias", _sq(data)
            if sub == "norm2.weight":
                return f"backbone.pos_net.{n}.norm2.weight", _sq(data)
            if sub == "norm2.bias":
                return f"backbone.pos_net.{n}.norm2.bias", _sq(data)
        elif n == 2:  # AttnBlock (Conv1d-based q/k/v/proj_out + GroupNorm)
            _conv_attn_map = {
                "attn_q": "q",
                "attn_k": "k",
                "attn_v": "v",
                "attn_output": "proj_out",
            }
            for gguf_prefix, pt_prefix in _conv_attn_map.items():
                if sub == f"{gguf_prefix}.weight":
                    return f"backbone.pos_net.2.{pt_prefix}.weight", data
                if sub == f"{gguf_prefix}.bias":
                    return f"backbone.pos_net.2.{pt_prefix}.bias", _sq(data)
            if sub == "attn_norm.weight":
                return "backbone.pos_net.2.norm.weight", _sq(data)
            if sub == "attn_norm.bias":
                return "backbone.pos_net.2.norm.bias", _sq(data)
        elif n == 5:  # GroupNorm (final Normalize)
            if sub == "attn_norm.weight":
                return "backbone.pos_net.5.weight", _sq(data)
            if sub == "attn_norm.bias":
                return "backbone.pos_net.5.bias", _sq(data)

    # Head output projection
    if name == "output.weight":
        return "head.out.weight", data
    if name == "output.bias":
        return "head.out.bias", data

    return None, None


def _load_from_gguf(config_path, model_path):
    """Load a WavTokenizer model from a GGUF checkpoint.

    Creates the model from config (which initialises pretrained encodec
    weights), then overlays backbone and head weights from the GGUF file.
    """
    import gguf

    _ensure_wavtokenizer_importable()
    from decoder.pretrained import WavTokenizer

    model = WavTokenizer.from_hparams0802(config_path)

    reader = gguf.GGUFReader(model_path)
    state_dict = {}
    for t in reader.tensors:
        tensor = torch.from_numpy(t.data.copy())
        mapped_name, mapped_tensor = _map_gguf_tensor(t.name, tensor)
        if mapped_name is not None:
            state_dict[mapped_name] = mapped_tensor

    # strict=False: keeps pretrained encodec keys and istft.window from model init
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


class ModelVariant(StrEnum):
    """Available WavTokenizer GGUF model variants."""

    LARGE_75_F16 = "Large_75_F16"
    LARGE_75_Q5_1 = "Large_75_Q5_1"


class ModelLoader(ForgeModel):
    """WavTokenizer GGUF model loader for discrete audio codec tokenization.

    Downloads a GGUF-packaged WavTokenizer decoder checkpoint from
    ggml-org/WavTokenizer and the companion config YAML from the base
    novateur/WavTokenizer repo, then loads them via the upstream WavTokenizer
    library.
    """

    _VARIANTS = {
        ModelVariant.LARGE_75_F16: ModelConfig(
            pretrained_model_name="ggml-org/WavTokenizer",
        ),
        ModelVariant.LARGE_75_Q5_1: ModelConfig(
            pretrained_model_name="ggml-org/WavTokenizer",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_75_Q5_1

    # Config YAML is hosted in the base novateur/WavTokenizer repo
    _CONFIG_REPO = "novateur/WavTokenizer"
    _CONFIG_FILENAME = (
        "wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    )

    _GGUF_FILES = {
        ModelVariant.LARGE_75_F16: "WavTokenizer-Large-75-F16.gguf",
        ModelVariant.LARGE_75_Q5_1: "WavTokenizer-Large-75-Q5_1.gguf",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WavTokenizer GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GGUF-packaged WavTokenizer decoder model."""
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(
            repo_id=self._CONFIG_REPO, filename=self._CONFIG_FILENAME
        )

        model_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=self._GGUF_FILES[self._variant],
        )

        model = _load_from_gguf(config_path, model_path)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample audio inputs for the WavTokenizer model.

        Returns:
            dict: Dictionary with 'audio_input' tensor (1-second mono audio at
                24kHz, shape [B, T]) and 'bandwidth_id' tensor.
        """
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)
        sample_rate = 24000
        audio_input = torch.randn(1, sample_rate, dtype=dtype)

        bandwidth_id = torch.tensor([0])

        return {"audio_input": audio_input, "bandwidth_id": bandwidth_id}
