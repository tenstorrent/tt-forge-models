# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import torch
from torchvision import transforms
import requests
from PIL import Image
import PIL.Image as pil
import zipfile
import hashlib
import shutil
from .resnet_encoder import ResnetEncoder
from .depth_decoder import DepthDecoder
from ....tools.utils import get_file


def download_model(model_name):
    """If pretrained model doesn't exist, download and unzip it"""
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
            "a964b8356e08a02d009609d9e3928f7c",
        ),
        "stereo_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
            "3dfb76bcff0786e4ec07ac00f658dd07",
        ),
        "mono+stereo_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
            "c024d69012485ed05d7eaa9617a96b81",
        ),
        "mono_no_pt_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
            "9c2f071e35027c895a4728358ffc913a",
        ),
        "stereo_no_pt_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
            "41ec2de112905f85541ac33a854742d1",
        ),
        "mono+stereo_no_pt_640x192": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
            "46c3b824f541d143a45c37df65fbab0a",
        ),
        "mono_1024x320": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
            "0ab0766efdfeea89a0d9ea8ba90e1e63",
        ),
        "stereo_1024x320": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
            "afc2f2126d70cf3fdf26b550898b501a",
        ),
        "mono+stereo_1024x320": (
            "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
            "cdc5fc9b23513c07d5b19235d9ef08f7",
        ),
    }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    # Check if model already exists
    if os.path.exists(model_path):
        return

    if model_name not in download_paths:
        raise ValueError(f"Unknown model variant: {model_name}")

    url, expected_md5 = download_paths[model_name]
    zip_path = os.path.join("models", f"{model_name}.zip")

    print(f"Downloading {model_name} model...")

    # Download the zip file
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract the zip file to a temporary location first
    temp_extract_path = os.path.join("models", "temp_extract")
    os.makedirs(temp_extract_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_extract_path)

    # Create the proper model directory
    os.makedirs(model_path, exist_ok=True)

    # Move files from temp extraction to proper model directory
    for file_name in os.listdir(temp_extract_path):
        src = os.path.join(temp_extract_path, file_name)
        dst = os.path.join(model_path, file_name)
        if os.path.isfile(src):
            shutil.move(src, dst)

    # Clean up temp directory and zip file
    shutil.rmtree(temp_extract_path)
    os.remove(zip_path)


class MonoDepth2(torch.nn.Module):
    def __init__(self, encoder, depth_decoder):
        super().__init__()
        self.encoder = encoder
        self.depth_decoder = depth_decoder

    def forward(self, input):
        features = self.encoder(input)
        outputs = self.depth_decoder(features)
        return outputs[("disp", 0)]


def load_model(variant):
    download_model(variant)

    encoder_path = os.path.join("models", variant, "encoder.pth")
    depth_decoder_path = os.path.join("models", variant, "depth.pth")

    encoder = ResnetEncoder(18, False)
    depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict_enc = torch.load(encoder_path, map_location="cpu")
    filtered_dict_enc = {
        k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()
    }
    encoder.load_state_dict(filtered_dict_enc)

    loaded_dict = torch.load(depth_decoder_path, map_location="cpu")
    depth_decoder.load_state_dict(loaded_dict)

    model = MonoDepth2(encoder, depth_decoder)
    model.eval()

    feed_height = loaded_dict_enc["height"]
    feed_width = loaded_dict_enc["width"]

    return model, feed_height, feed_width


def load_input(feed_height, feed_width):

    image_file = get_file(
        "https://raw.githubusercontent.com/nianticlabs/monodepth2/master/assets/test_image.jpg"
    )
    input_image = Image.open(image_file).convert("RGB")
    original_width, original_height = input_image.size
    input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    input_tensor = transforms.ToTensor()(input_image_resized).unsqueeze(0)
    return input_tensor, original_width, original_height
