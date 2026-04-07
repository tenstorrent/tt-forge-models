# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import cv2
import numpy as np
from .....tools.utils import get_file
import torch
import torch.nn.functional as F
from ..src.model import decode, nms


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


def postprocess_outputs(outputs):
    if (
        isinstance(outputs, torch.Tensor)
        and outputs.dim() == 4
        and outputs.size(-1) == 5
    ):
        return outputs

    # Expect tuple of (loc, conf, priors) from the model
    loc, conf, priors = outputs

    variance = [0.1, 0.2]
    top_k = 200
    conf_thresh = 0.01
    nms_thresh = 0.45
    num_classes = 21

    num = loc.size(0)
    num_priors = priors.size(0)
    device = loc.device
    dtype = loc.dtype

    # Softmax over classes
    conf_data = conf.view(num, num_priors, num_classes)
    conf_softmax = F.softmax(conf_data, dim=-1)
    # Shape to [num, num_classes, num_priors] to match original implementation
    conf_preds = conf_softmax.transpose(2, 1).contiguous()

    output = torch.zeros(num, num_classes, top_k, 5, device=device, dtype=dtype)

    # Per-image post-processing
    for i in range(num):
        decoded_boxes = decode(loc[i], priors, variance)
        conf_scores = conf_preds[i].clone()  # [num_classes, num_priors]

        for cl in range(1, num_classes):
            c_mask = conf_scores[cl].gt(conf_thresh)
            scores = conf_scores[cl][c_mask]
            if scores.numel() == 0:
                continue
            l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
            boxes = decoded_boxes[l_mask].view(-1, 4)
            ids, count = nms(boxes, scores, nms_thresh, top_k)
            output[i, cl, :count] = torch.cat(
                (scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1
            )

    # Global ranking and top-k pruning per image
    flt = output.contiguous().view(num, -1, 5)
    _, idx = flt[:, :, 0].sort(1, descending=True)
    _, rank = idx.sort(1)
    flt[(rank < top_k).unsqueeze(-1).expand_as(flt)].fill_(0)

    return output
