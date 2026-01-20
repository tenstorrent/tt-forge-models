# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Example script demonstrating how to:
1. Download a model from HuggingFace
2. Upload it to your personal HuggingFace Hub
3. Load it from your personal Hub with caching

Requirements:
    pip install huggingface_hub transformers torch

Setup:
    1. Create a HuggingFace account at https://huggingface.co
    2. Create an access token at https://huggingface.co/settings/tokens (with write access)
    3. Login via: huggingface-cli login
       OR set: export HF_TOKEN="hf_your_token_here"
    4. Create a new model repo at https://huggingface.co/new

Usage:
    # Step 1: Upload model to your hub (one-time)
    python hf_upload_example.py --upload --repo-id "your-username/my-gpt2"

    # Step 2: Use your uploaded model (first run downloads, second run uses cache)
    python hf_upload_example.py --use --repo-id "your-username/my-gpt2"
"""

import argparse
import os
from pathlib import Path


def upload_model_to_hub(
    source_model: str,
    repo_id: str,
    private: bool = False,
):
    """Download a model and upload it to your HuggingFace Hub.

    Args:
        source_model: Source model to download (e.g., "gpt2", "microsoft/phi-1")
        repo_id: Your HuggingFace repo ID (e.g., "your-username/my-model")
        private: Whether to make the repo private
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import HfApi, create_repo

    print(f"[Step 1] Downloading model '{source_model}'...", flush=True)

    # Download model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(source_model)
    tokenizer = AutoTokenizer.from_pretrained(source_model)

    print(f"[Step 2] Creating repository '{repo_id}' on HuggingFace Hub...", flush=True)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"Repository '{repo_id}' ready.", flush=True)
    except Exception as e:
        print(f"Note: {e}", flush=True)

    print(f"[Step 3] Uploading model to '{repo_id}'...", flush=True)

    # Push model and tokenizer to Hub
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)

    print(f"[Success] Model uploaded to: https://huggingface.co/{repo_id}", flush=True)
    print(f"\nYou can now use this model with:", flush=True)
    print(f'  python hf_upload_example.py --use --repo-id "{repo_id}"', flush=True)


def use_model_from_hub(repo_id: str):
    """Load a model from your HuggingFace Hub with cache-first logic.

    First run: Downloads from HuggingFace Hub
    Second run: Uses cached version (no download needed)

    Args:
        repo_id: Your HuggingFace repo ID (e.g., "your-username/my-model")
    """
    from utils import load_huggingface_model, load_huggingface_tokenizer, is_huggingface_model_cached
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Check cache status
    is_cached = is_huggingface_model_cached(repo_id)
    print(f"\n{'='*60}", flush=True)
    print(f"Model: {repo_id}", flush=True)
    print(f"Cache status: {'CACHED ✓' if is_cached else 'NOT CACHED - will download'}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Load tokenizer with cache-first logic
    tokenizer = load_huggingface_tokenizer(
        AutoTokenizer,
        repo_id,
    )

    # Load model with cache-first logic
    model = load_huggingface_model(
        AutoModelForCausalLM,
        repo_id,
    )

    # Test the model
    print("\n[Testing model]", flush=True)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer("Hello, I am a language model", return_tensors="pt")

    # Generate some text
    model.eval()
    with __import__('torch').no_grad():
        outputs = model(**inputs)

    print(f"Model output shape: {outputs.logits.shape}", flush=True)
    print(f"\n[Success] Model loaded and tested!", flush=True)

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Upload and use models from your personal HuggingFace Hub"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload a model to your HuggingFace Hub",
    )
    parser.add_argument(
        "--use",
        action="store_true",
        help="Use a model from your HuggingFace Hub (with caching)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Your HuggingFace repo ID (e.g., 'your-username/my-model')",
    )
    parser.add_argument(
        "--source-model",
        type=str,
        default="gpt2",
        help="Source model to download and upload (default: gpt2)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )

    args = parser.parse_args()

    if not args.upload and not args.use:
        parser.error("Please specify --upload or --use")

    if args.upload:
        upload_model_to_hub(
            source_model=args.source_model,
            repo_id=args.repo_id,
            private=args.private,
        )

    if args.use:
        use_model_from_hub(repo_id=args.repo_id)


if __name__ == "__main__":
    main()

