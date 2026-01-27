# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import cv2
import numpy as np
from .....tools.utils import get_file
import torch


def load_ssd512_inputs():
    image_path = get_file("test_images/ssd512_input.jpg")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    dataset_mean = (104, 117, 123)
    transform = BaseTransform(300, dataset_mean)
    img_t, _, _ = transform(img)
    img_t = img_t[:, :, (2, 1, 0)]
    x = torch.from_numpy(img_t).permute(2, 0, 1).unsqueeze(0)
    return x


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels
