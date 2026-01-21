# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Example script demonstrating how to:
1. Save a model's weights locally
2. Upload weights to your personal Google Drive
3. Load weights from Google Drive with caching

Requirements:
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib transformers torch

Setup:
    1. Go to https://console.cloud.google.com/
    2. Create a new project or select existing
    3. Enable the Google Drive API
    4. Create OAuth 2.0 credentials (Desktop app)
    5. Download credentials.json and save to ~/.config/gdrive/credentials.json
    
    OR use a simpler method:
    1. Upload file manually to Google Drive
    2. Right-click -> Share -> Anyone with link
    3. Copy the file ID from the URL

Usage:
    # Step 1: Save model weights locally
    python gdrive_upload_example.py --save --source-model "gpt2" --output "gpt2_weights.pt"

    # Step 2: Upload to Google Drive manually or use the upload function
    # After uploading, get the file ID from the shareable link
    
    # Step 3: Use your uploaded weights
    python gdrive_upload_example.py --use --file-id "YOUR_GDRIVE_FILE_ID" --filename "gpt2_weights.pt"
"""

import argparse
import os
import torch
from pathlib import Path


def save_model_weights(
    source_model: str,
    output_path: str,
):
    """Download a model and save its weights locally for uploading to Google Drive.

    Args:
        source_model: Source model to download (e.g., "gpt2", "microsoft/phi-1")
        output_path: Local path to save the weights
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Step 1] Downloading model '{source_model}'...", flush=True)

    # Download model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(source_model)
    tokenizer = AutoTokenizer.from_pretrained(source_model)

    print(f"[Step 2] Saving model weights to '{output_path}'...", flush=True)

    # Save the state dict
    torch.save(model.state_dict(), output_path)

    # Also save tokenizer for convenience
    tokenizer_dir = Path(output_path).parent / f"{Path(output_path).stem}_tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Success] Weights saved to: {output_path} ({file_size_mb:.1f} MB)", flush=True)
    print(f"[Success] Tokenizer saved to: {tokenizer_dir}", flush=True)
    print(f"\nNext steps:", flush=True)
    print(f"  1. Upload '{output_path}' to your Google Drive", flush=True)
    print(f"  2. Right-click -> Share -> Anyone with link -> Copy link", flush=True)
    print(f"  3. Extract the file ID from the URL", flush=True)
    print(f"     URL format: https://drive.google.com/file/d/FILE_ID/view", flush=True)
    print(f"  4. Use the file ID to load weights:", flush=True)
    print(f'     python gdrive_upload_example.py --use --file-id "YOUR_FILE_ID" --filename "gpt2_weights.pt"', flush=True)


def upload_to_gdrive_manual_instructions(file_path: str):
    """Print instructions for manually uploading to Google Drive."""
    print(f"\n{'='*60}", flush=True)
    print("Manual Upload Instructions:", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"1. Go to https://drive.google.com", flush=True)
    print(f"2. Click '+ New' -> 'File upload'", flush=True)
    print(f"3. Select: {file_path}", flush=True)
    print(f"4. Wait for upload to complete", flush=True)
    print(f"5. Right-click the uploaded file -> 'Share'", flush=True)
    print(f"6. Change access to 'Anyone with the link'", flush=True)
    print(f"7. Click 'Copy link'", flush=True)
    print(f"8. Extract the FILE_ID from the URL:", flush=True)
    print(f"   https://drive.google.com/file/d/FILE_ID/view?usp=sharing", flush=True)
    print(f"{'='*60}\n", flush=True)


def use_model_from_gdrive(file_id: str, filename: str):
    """Load a model from Google Drive with cache-first logic.

    Args:
        file_id: Google Drive file ID
        filename: Filename for the cached file
    """
    from utils import download_from_gdrive, is_gdrive_file_cached
    from transformers import GPT2LMHeadModel, GPT2Config

    # Check cache status
    is_cached = is_gdrive_file_cached(file_id, filename)
    print(f"\n{'='*60}", flush=True)
    print(f"Google Drive File ID: {file_id}", flush=True)
    print(f"Cache status: {'CACHED ✓' if is_cached else 'NOT CACHED - will download'}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Download/load from cache
    weights_path = download_from_gdrive(file_id, filename)

    # Load the weights
    print(f"[Loading] Loading weights from {weights_path}...", flush=True)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    # Create model and load weights
    # Note: You need to know the model architecture
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[Success] Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Save and load model weights from Google Drive"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save model weights locally for uploading to Google Drive",
    )
    parser.add_argument(
        "--use",
        action="store_true",
        help="Load model from Google Drive (with caching)",
    )
    parser.add_argument(
        "--source-model",
        type=str,
        default="gpt2",
        help="Source model to download (default: gpt2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_weights.pt",
        help="Output path for saved weights",
    )
    parser.add_argument(
        "--file-id",
        type=str,
        help="Google Drive file ID for loading",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="model_weights.pt",
        help="Filename for cached file",
    )

    args = parser.parse_args()

    if not args.save and not args.use:
        parser.error("Please specify --save or --use")

    if args.save:
        save_model_weights(
            source_model=args.source_model,
            output_path=args.output,
        )
        upload_to_gdrive_manual_instructions(args.output)

    if args.use:
        if not args.file_id:
            parser.error("--file-id is required when using --use")
        use_model_from_gdrive(
            file_id=args.file_id,
            filename=args.filename,
        )


if __name__ == "__main__":
    main()

