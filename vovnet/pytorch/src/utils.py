# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import hashlib
import os
import zipfile
from six.moves import urllib
import torch
from loguru import logger
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import requests
import shutil
import time
from datasets import load_dataset


def preprocess_steps(model_type):
    model = model_type(False, True).eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    dataset = load_dataset("huggingface/cats-image", split="test")
    img = dataset[0]["image"].convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    return model, img_tensor


def preprocess_timm_model(model_name):
    use_pretrained_weights = False
    if model_name == "ese_vovnet99b":
        use_pretrained_weights = False
    model = timm.create_model(model_name, pretrained=use_pretrained_weights)
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    dataset = load_dataset("huggingface/cats-image", split="test")
    img = dataset[0]["image"].convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    return model, img_tensor


def download_model(download_func, *args, num_retries=3, timeout=180, **kwargs):
    for _ in range(num_retries):
        try:
            return download_func(*args, **kwargs)
        except (
            requests.exceptions.HTTPError,
            urllib.error.HTTPError,
            requests.exceptions.ReadTimeout,
            urllib.error.URLError,
        ):
            logger.trace("HTTP error occurred. Retrying...")
            shutil.rmtree(os.path.expanduser("~") + "/.cache", ignore_errors=True)
            shutil.rmtree(
                os.path.expanduser("~") + "/.torch/models", ignore_errors=True
            )
            shutil.rmtree(
                os.path.expanduser("~") + "/.torchxrayvision/models_data",
                ignore_errors=True,
            )
            os.makedirs(os.path.expanduser("~") + "/.cache", exist_ok=True)
        time.sleep(timeout)

    logger.error("Failed to download the model after multiple retries.")
    assert False, "Failed to download the model after multiple retries."
