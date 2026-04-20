# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RhoFold+ model wrapper.

Handles cloning the RhoFold repository, loading the pretrained weights from
HuggingFace, and building the RhoFold RNA 3D structure prediction model using
its custom code.

Reference: https://github.com/ml4bio/RhoFold
HuggingFace: https://huggingface.co/cuhkaih/rhofold
"""

import sys
import subprocess
import tempfile
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

RHOFOLD_REPO_URL = "https://github.com/ml4bio/RhoFold.git"
RHOFOLD_CACHE_DIR = Path.home() / ".cache" / "rhofold"


def _ensure_repo_cloned():
    """Clone the RhoFold repo to cache if not already present."""
    repo_dir = RHOFOLD_CACHE_DIR / "RhoFold"

    if not repo_dir.exists():
        RHOFOLD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", RHOFOLD_REPO_URL, str(repo_dir)],
            check=True,
        )

    return repo_dir


def _add_repo_to_path(repo_dir):
    """Add the RhoFold repo to sys.path so its modules can be imported."""
    repo_str = str(repo_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def build_rhofold_model(repo_id, weights_filename):
    """Build and return the RhoFold model with pretrained weights loaded."""
    repo_dir = _ensure_repo_cloned()
    _add_repo_to_path(repo_dir)

    from rhofold.config import rhofold_config
    from rhofold.rhofold import RhoFold

    model = RhoFold(rhofold_config)

    weights_path = hf_hub_download(repo_id=repo_id, filename=weights_filename)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    return model


def build_rhofold_inputs(sequence):
    """Prepare RhoFold input features from a raw RNA sequence.

    Uses single-sequence mode where the FASTA file also serves as the MSA,
    matching the ``--single_seq_pred`` path in the upstream inference script.
    """
    repo_dir = _ensure_repo_cloned()
    _add_repo_to_path(repo_dir)

    from rhofold.utils.alphabet import get_features

    with tempfile.NamedTemporaryFile(
        suffix=".fasta", mode="w", delete=False
    ) as fasta_file:
        fasta_file.write(f">sample\n{sequence}\n")
        fasta_path = fasta_file.name

    return get_features(fasta_path, fasta_path)
