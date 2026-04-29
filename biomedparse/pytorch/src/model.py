# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BiomedParse model wrapper.

Handles cloning the BiomedParse repository and loading the model
using its custom code. BiomedParse is a biomedical foundation model
for joint segmentation, detection, and recognition across 9 imaging modalities.

Reference: https://github.com/microsoft/BiomedParse
HuggingFace: https://huggingface.co/microsoft/BiomedParse
"""

import sys
import subprocess
from pathlib import Path

# Repository details
BIOMEDPARSE_REPO_URL = "https://github.com/microsoft/BiomedParse.git"
BIOMEDPARSE_CACHE_DIR = Path.home() / ".cache" / "biomedparse"


def _ensure_repo_cloned():
    """Clone the BiomedParse repo (main branch) to cache if not already present.

    Returns:
        Path: Path to the cloned repository.
    """
    repo_dir = BIOMEDPARSE_CACHE_DIR / "BiomedParse"

    if repo_dir.exists():
        # Verify it's on the main branch (not v2 which lacks modeling/)
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "symbolic-ref", "--short", "HEAD"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or result.stdout.strip() != "main":
            import shutil

            shutil.rmtree(repo_dir)

    if not repo_dir.exists():
        BIOMEDPARSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "main",
                BIOMEDPARSE_REPO_URL,
                str(repo_dir),
            ],
            check=True,
        )

    return repo_dir


def _add_repo_to_path(repo_dir):
    """Add the BiomedParse repo to sys.path so its modules can be imported.

    Args:
        repo_dir: Path to the cloned BiomedParse repository.
    """
    repo_str = str(repo_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def build_biomedparse_model():
    """Build and return the BiomedParse model using the repository's code.

    Returns:
        torch.nn.Module: The BiomedParse model in eval mode.
    """
    repo_dir = _ensure_repo_cloned()
    _add_repo_to_path(repo_dir)

    # PIL.Image.LINEAR was removed in Pillow 10; detectron2 0.6 uses it.
    from PIL import Image as _Image

    if not hasattr(_Image, "LINEAR"):
        _Image.LINEAR = _Image.BILINEAR

    from modeling.BaseModel import BaseModel
    from modeling import build_model
    from utilities.arguments import load_opt_from_config_files
    from utilities.distributed import init_distributed

    config_path = str(repo_dir / "configs" / "biomedparse_inference.yaml")
    opt = load_opt_from_config_files([config_path])
    opt = init_distributed(opt)

    pretrained_pth = "hf_hub:microsoft/BiomedParse"
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval()

    return model
