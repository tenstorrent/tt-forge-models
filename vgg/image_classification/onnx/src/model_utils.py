# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torchvision import transforms
from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file
from loguru import logger


def get_input_img_osmr():
    try:
        file_path = get_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
        input_image = Image.open(file_path).convert("RGB")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(
            0
        )  # create a mini-batch as expected by the model
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        input_batch = torch.rand(1, 3, 224, 224)
    return input_batch
