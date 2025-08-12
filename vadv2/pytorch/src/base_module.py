# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import copy
import torch.nn as nn
from abc import ABC, abstractmethod, ABCMeta


class BaseModule(nn.Module, metaclass=ABCMeta):
    """Base module for all modules in openmmlab.

    ``BaseModule`` is a wrapper of ``torch.nn.Module`` with additional
    functionality of parameter initialization. Compared with
    ``torch.nn.Module``, ``BaseModule`` mainly adds three attributes.

        - ``init_cfg``: the config to control the initialization.
        - ``init_weights``: The function of parameter
            initialization and recording initialization
            information.
        - ``_params_init_info``: Used to track the parameter
            initialization information. This attribute only
            exists during executing the ``init_weights``.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, init_cfg=None):
        """Initialize BaseModule, inherited from `torch.nn.Module`"""
        super(BaseModule, self).__init__()
        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f"\ninit_cfg={self.init_cfg}"
        return s


class Sequential(BaseModule, nn.Sequential):
    """Sequential module in openmmlab.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, *args, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)


class ModuleList(BaseModule, nn.ModuleList):
    """ModuleList in openmmlab.

    Args:
        modules (iterable, optional): an iterable of modules to add.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, modules=None, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleList.__init__(self, modules)


class BaseDetector(BaseModule, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self, init_cfg=None):
        super().__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, "neck") and self.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, "roi_head") and self.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return (hasattr(self, "roi_head") and self.roi_head.with_bbox) or (
            hasattr(self, "bbox_head") and self.bbox_head is not None
        )

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return (hasattr(self, "roi_head") and self.roi_head.with_mask) or (
            hasattr(self, "mask_head") and self.mask_head is not None
        )

    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass

    def extract_feats(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, "imgs"), (img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError(f"{name} must be a list, but got {type(var)}")

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                f"num of augmentations ({len(imgs)}) "
                f"!= num of image meta ({len(img_metas)})"
            )

        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]["batch_input_shape"] = tuple(img.size()[-2:])

        if num_augs == 1:
            if "proposals" in kwargs:
                kwargs["proposals"] = kwargs["proposals"][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, (
                "aug test does not support "
                "inference with batch size "
                f"{imgs[0].size(0)}"
            )
            # TODO: support test augmentation for predefined proposals
            assert "proposals" not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.
        """

        return self.forward_test(img, img_metas, **kwargs)


class Base3DDetector(BaseDetector):
    """Base class for detectors."""

    def forward_test(self, points, img_metas, img=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(points, "points"), (img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))

        num_augs = len(points)
        if num_augs != len(img_metas):
            raise ValueError(
                "num of augmentations ({}) != num of image meta ({})".format(
                    len(points), len(img_metas)
                )
            )

        if num_augs == 1:
            img = [img] if img is None else img
            return self.simple_test(points[0], img_metas[0], img[0], **kwargs)
        else:
            return self.aug_test(points, img_metas, img, **kwargs)

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
