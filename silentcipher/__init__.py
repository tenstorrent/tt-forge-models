# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SilentCipher deep audio watermarking model implementations.
"""
import argparse
import os

import torch
import yaml


# HuggingFace repo paths for each model type
_HF_REPO = "sony/silentcipher"
_CKPT_SUBDIRS = {
    "44.1k": "44_1_khz/73999_iteration",
    "16k": "16_khz/97561_iteration",
}


class _Pipeline:
    """Minimal SilentCipher pipeline exposing dec_m, config, and stft."""

    def __init__(self, config, ckpt_dir, device="cpu"):
        from .model import MsgDecoder
        from .stft import STFT

        self.config = config
        self.stft = STFT(config.N_FFT, config.HOP_LENGTH)
        self.stft.to(device)

        self.dec_m = [
            MsgDecoder(
                message_dim=config.message_dim,
                message_band_size=config.message_band_size,
            )
            for _ in range(config.n_messages)
        ]
        self._load_checkpoints(ckpt_dir, device)

    def _load_checkpoints(self, ckpt_dir, device):
        for i, m in enumerate(self.dec_m):
            ckpt_path = os.path.join(ckpt_dir, f"dec_m_{i}.ckpt")
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            # Strip DataParallel "module." prefix if present
            state = {
                (k[7:] if k.startswith("module.") else k): v for k, v in state.items()
            }
            m.load_state_dict(state)


def get_model(model_type="44.1k", device="cpu", **kwargs):
    """Download SilentCipher from HuggingFace Hub and return a pipeline."""
    from huggingface_hub import snapshot_download

    ckpt_subdir = _CKPT_SUBDIRS[model_type]
    folder_dir = snapshot_download(repo_id=_HF_REPO)
    ckpt_dir = os.path.join(folder_dir, ckpt_subdir)
    config_path = os.path.join(ckpt_dir, "hparams.yaml")

    with open(config_path) as f:
        cfg = argparse.Namespace(**yaml.safe_load(f))
    cfg.load_ckpt = ckpt_dir

    return _Pipeline(cfg, ckpt_dir, device)
