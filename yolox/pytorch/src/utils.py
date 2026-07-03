# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import cv2
import numpy as np
import torch


def get_detections(out_np, ratio, input_shape, score_thr=0.1, nms_thr=0.45):
    """Run NMS on raw model output and return a list of detection dicts.

    Args:
        out_np: numpy array of shape [batch, n_anchors, 85].
        ratio: scale ratio from preproc, used to map boxes back to original image size.
        input_shape: (h, w) tuple used during preprocessing.
        score_thr: confidence threshold for NMS. Defaults to 0.1 (demo/display).
            Lower it (e.g. 0.01) for COCO mAP so low-confidence detections are
            retained for recall/AP.
        nms_thr: IoU threshold for NMS. Defaults to 0.45.

    Returns:
        List of dicts with keys: class_name, score, cls_ind, bbox ([x1,y1,x2,y2]).
        Sorted by descending confidence.
    """
    from yolox.data.datasets import COCO_CLASSES
    from yolox.utils import demo_postprocess, multiclass_nms

    predictions = demo_postprocess(out_np, input_shape)[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=score_thr)

    result = []
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        for box, score, cls_ind in zip(final_boxes, final_scores, final_cls_inds):
            result.append(
                {
                    "class_name": COCO_CLASSES[int(cls_ind)],
                    "score": float(score),
                    "cls_ind": int(cls_ind),
                    "bbox": box.tolist(),
                }
            )
    result.sort(key=lambda d: d["score"], reverse=True)
    return result


def _forward_patch(self, xin, labels=None, imgs=None):
    outputs = []
    origin_preds = []
    x_shifts = []
    y_shifts = []
    expanded_strides = []

    for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
        zip(self.cls_convs, self.reg_convs, self.strides, xin)
    ):
        x = self.stems[k](x)
        cls_x = x
        reg_x = x

        cls_feat = cls_conv(cls_x)
        cls_output = self.cls_preds[k](cls_feat)

        reg_feat = reg_conv(reg_x)
        reg_output = self.reg_preds[k](reg_feat)
        obj_output = self.obj_preds[k](reg_feat)

        if self.training:
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self.get_output_and_grid(
                output, k, stride_this_level, xin[0].type()
            )
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(xin[0])
            )
            if self.use_l1:
                batch_size = reg_output.shape[0]
                hsize, wsize = reg_output.shape[-2:]
                reg_output = reg_output.view(
                    batch_size, self.n_anchors, 4, hsize, wsize
                )
                reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                    batch_size, -1, 4
                )
                origin_preds.append(reg_output.clone())

        else:
            output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )

        outputs.append(output)

    if self.training:
        return self.get_losses(
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            torch.cat(outputs, 1),
            origin_preds,
            dtype=xin[0].dtype,
        )
    else:
        self.hw = [x.shape[-2:] for x in outputs]
        # [batch, n_anchors_all, 85]
        outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(
            0, 2, 1
        )
        if self.decode_in_inference:
            return self.decode_outputs(outputs, dtype=xin[0].dtype)
        else:
            return outputs


def _decode_outputs(self, outputs, dtype):
    from yolox.utils import meshgrid

    grids = []
    strides = []
    for (hsize, wsize), stride in zip(self.hw, self.strides):
        yv, xv = meshgrid(
            [
                torch.arange(hsize, device=outputs.device),
                torch.arange(wsize, device=outputs.device),
            ]
        )
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        strides.append(torch.full((*shape, 1), stride, device=outputs.device))

    grids = torch.cat(grids, dim=1).to(dtype=dtype, device=outputs.device)
    strides = torch.cat(strides, dim=1).to(dtype=dtype, device=outputs.device)

    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    return outputs
