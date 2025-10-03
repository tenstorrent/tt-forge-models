# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Panoptic-DeepLab model loader implementation based on CPU inference patterns
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import torch
import numpy as np

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


@dataclass
class PanopticDeepLabConfig(ModelConfig):
    """Configuration specific to Panoptic-DeepLab models"""

    config_file: str
    backbone: str
    num_classes: int
    image_size: tuple
    output_stride: int
    use_depthwise_separable_conv: bool = False


class ModelVariant(StrEnum):
    """Available Panoptic-DeepLab model variants."""

    # COCO variants
    RESNET_52_OS16_COCO = "resnet52_os16_coco"
    RESNET_101_OS16_COCO = "resnet101_os16_coco"
    RESNET_52_OS16_COCO_DSCONV = "resnet52_os16_coco_dsconv"

    # Cityscapes variants
    RESNET_52_OS16_CITYSCAPES = "resnet52_os16_cityscapes"
    RESNET_52_OS16_CITYSCAPES_DSCONV = "resnet52_os16_cityscapes_dsconv"


class ModelLoader(ForgeModel):
    """Panoptic-DeepLab model loader implementation based on CPU inference patterns."""

    _VARIANTS = {
        ModelVariant.RESNET_52_OS16_COCO: PanopticDeepLabConfig(
            pretrained_model_name="panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco",
            config_file="COCO-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco.yaml",
            backbone="resnet50",  # Use ResNet-50 instead of 52
            num_classes=133,  # COCO panoptic classes
            image_size=(640, 640),
            output_stride=16,
            use_depthwise_separable_conv=False,
        ),
        ModelVariant.RESNET_101_OS16_COCO: PanopticDeepLabConfig(
            pretrained_model_name="panoptic_deeplab_R_101_os16_mg124_poly_200k_bs64_crop_640_640_coco",
            config_file="COCO-PanopticSegmentation/panoptic_deeplab_R_101_os16_mg124_poly_200k_bs64_crop_640_640_coco.yaml",
            backbone="resnet101",
            num_classes=133,
            image_size=(640, 640),
            output_stride=16,
            use_depthwise_separable_conv=False,
        ),
        ModelVariant.RESNET_52_OS16_COCO_DSCONV: PanopticDeepLabConfig(
            pretrained_model_name="panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv",
            config_file="COCO-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml",
            backbone="resnet50",
            num_classes=133,
            image_size=(640, 640),
            output_stride=16,
            use_depthwise_separable_conv=True,
        ),
        ModelVariant.RESNET_52_OS16_CITYSCAPES: PanopticDeepLabConfig(
            pretrained_model_name="panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024",
            config_file="Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml",
            backbone="resnet50",
            num_classes=19,  # Cityscapes classes
            image_size=(1024, 2048),
            output_stride=16,
            use_depthwise_separable_conv=False,
        ),
        ModelVariant.RESNET_52_OS16_CITYSCAPES_DSCONV: PanopticDeepLabConfig(
            pretrained_model_name="panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv",
            config_file="Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml",
            backbone="resnet50",
            num_classes=19,
            image_size=(1024, 2048),
            output_stride=16,
            use_depthwise_separable_conv=True,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.RESNET_52_OS16_COCO

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.predictor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[StrEnum] = None) -> ModelInfo:
        """Get model information for the specified variant."""
        variant = variant or cls.DEFAULT_VARIANT

        return ModelInfo(
            model="panoptic_deeplab",
            variant=variant,
            framework=Framework.TORCH,
            task=ModelTask.CV_PANOPTIC_SEG,
            source=ModelSource.DETECTRON2,
            group=ModelGroup.GENERALITY,
        )

    def _setup_cfg(
        self, device: str = "cpu", dtype_override: Optional[torch.dtype] = None
    ) -> "CfgNode":
        """Setup detectron2 configuration based on the CPU inference script.

        Args:
            device: Device to run inference on (default: cpu)
            dtype_override: Override model dtype (currently not used by detectron2)

        Returns:
            CfgNode: Configured detectron2 config object
        """
        try:
            from detectron2.config import get_cfg
            from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
        except ImportError as e:
            raise ImportError(
                "detectron2 and panoptic-deeplab dependencies are required. "
                "Install with: pip install 'git+https://github.com/facebookresearch/detectron2.git'"
            ) from e

        config = self._variant_config

        # Create configs and perform basic setups (following CPU inference script)
        cfg = get_cfg()

        # Add Panoptic-DeepLab config
        add_panoptic_deeplab_config(cfg)

        # Set basic configuration without loading config file for now
        # (since we might not have access to the config files in the environment)
        self._configure_model_architecture(cfg, config)

        # Force specified device
        cfg.MODEL.DEVICE = device

        # Set model weights to empty for random initialization (testing purposes)
        cfg.MODEL.WEIGHTS = ""

        # Set confidence threshold (similar to CPU inference script)
        if hasattr(cfg.MODEL, "PANOPTIC_DEEPLAB"):
            cfg.MODEL.PANOPTIC_DEEPLAB.CENTER_THRESHOLD = 0.1

        cfg.freeze()
        return cfg

    def _configure_model_architecture(
        self, cfg: "CfgNode", config: PanopticDeepLabConfig
    ):
        """Configure the model architecture based on variant config.

        Args:
            cfg: Detectron2 config node
            config: PanopticDeepLabConfig for the current variant
        """
        # Set meta architecture
        cfg.MODEL.META_ARCHITECTURE = "PanopticDeepLab"

        # Configure backbone
        cfg.MODEL.BACKBONE.NAME = "build_resnet_deeplab_backbone"
        if config.backbone == "resnet50":
            cfg.MODEL.RESNETS.DEPTH = 50
        elif config.backbone == "resnet101":
            cfg.MODEL.RESNETS.DEPTH = 101
        else:
            cfg.MODEL.RESNETS.DEPTH = 50  # Default

        # Configure ResNet properties (following panoptic-deeplab configs)
        cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res5"]
        cfg.MODEL.RESNETS.RES5_DILATION = 2
        cfg.MODEL.RESNETS.NORM = "SyncBN"
        cfg.MODEL.RESNETS.STEM_TYPE = "deeplab"
        cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 128
        cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False

        # Configure semantic segmentation head
        cfg.MODEL.SEM_SEG_HEAD.NAME = "PanopticDeepLabSemSegHead"
        cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res5"]
        cfg.MODEL.SEM_SEG_HEAD.PROJECT_FEATURES = ["res2", "res3"]
        cfg.MODEL.SEM_SEG_HEAD.PROJECT_CHANNELS = [32, 64]
        cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS = 256
        cfg.MODEL.SEM_SEG_HEAD.ASPP_DILATIONS = [6, 12, 18]
        cfg.MODEL.SEM_SEG_HEAD.ASPP_DROPOUT = 0.1
        cfg.MODEL.SEM_SEG_HEAD.HEAD_CHANNELS = 256
        cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 256
        cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = config.num_classes
        cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE = "hard_pixel_mining"
        cfg.MODEL.SEM_SEG_HEAD.NORM = "SyncBN"
        cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV = (
            config.use_depthwise_separable_conv
        )

        # Configure instance embedding head
        cfg.MODEL.INS_EMBED_HEAD.NAME = "PanopticDeepLabInsEmbedHead"
        cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES = ["res2", "res3", "res5"]
        cfg.MODEL.INS_EMBED_HEAD.PROJECT_FEATURES = ["res2", "res3"]
        cfg.MODEL.INS_EMBED_HEAD.PROJECT_CHANNELS = [32, 64]
        cfg.MODEL.INS_EMBED_HEAD.ASPP_CHANNELS = 256
        cfg.MODEL.INS_EMBED_HEAD.ASPP_DILATIONS = [6, 12, 18]
        cfg.MODEL.INS_EMBED_HEAD.ASPP_DROPOUT = 0.1
        cfg.MODEL.INS_EMBED_HEAD.HEAD_CHANNELS = 32
        cfg.MODEL.INS_EMBED_HEAD.CONVS_DIM = 128
        cfg.MODEL.INS_EMBED_HEAD.COMMON_STRIDE = 4
        cfg.MODEL.INS_EMBED_HEAD.NORM = "SyncBN"
        cfg.MODEL.INS_EMBED_HEAD.CENTER_LOSS_WEIGHT = 200.0
        cfg.MODEL.INS_EMBED_HEAD.OFFSET_LOSS_WEIGHT = 0.01

        # Configure Panoptic-DeepLab post-processing
        cfg.MODEL.PANOPTIC_DEEPLAB.STUFF_AREA = (
            2048 if "cityscapes" in config.pretrained_model_name.lower() else 4096
        )
        cfg.MODEL.PANOPTIC_DEEPLAB.CENTER_THRESHOLD = 0.1
        cfg.MODEL.PANOPTIC_DEEPLAB.NMS_KERNEL = (
            7 if "cityscapes" in config.pretrained_model_name.lower() else 41
        )
        cfg.MODEL.PANOPTIC_DEEPLAB.TOP_K_INSTANCE = 200
        cfg.MODEL.PANOPTIC_DEEPLAB.PREDICT_INSTANCES = True
        cfg.MODEL.PANOPTIC_DEEPLAB.USE_DEPTHWISE_SEPARABLE_CONV = (
            config.use_depthwise_separable_conv
        )
        cfg.MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY = config.image_size[0]
        cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED = False

        # Set input format and size
        cfg.INPUT.FORMAT = "RGB"
        cfg.INPUT.MIN_SIZE_TEST = config.image_size[0]
        cfg.INPUT.MAX_SIZE_TEST = config.image_size[1]

        # Configure pixel mean and std (COCO defaults)
        cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
        cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]

        # Dataset configuration
        if "coco" in config.pretrained_model_name.lower():
            cfg.DATASETS.TRAIN = ("coco_2017_train_panoptic",)
            cfg.DATASETS.TEST = ("coco_2017_val_panoptic",)
        else:  # cityscapes
            cfg.DATASETS.TRAIN = ("cityscapes_fine_panoptic_train",)
            cfg.DATASETS.TEST = ("cityscapes_fine_panoptic_val",)

    def load_model(self, **kwargs) -> torch.nn.Module:
        """Load and return the Panoptic-DeepLab model instance.

        Args:
            **kwargs: Additional model-specific arguments.
                     - dtype_override: Override model dtype (e.g., torch.bfloat16)
                     - device: Device to load model on (default: cpu)
                     - force_cpu: Force CPU usage even if CUDA is available

        Returns:
            torch.nn.Module: The Panoptic-DeepLab model instance
        """
        dtype_override = kwargs.get("dtype_override", torch.float32)
        device = kwargs.get("device", "cpu")
        force_cpu = kwargs.get("force_cpu", True)

        # Force CPU usage (following CPU inference script pattern)
        if force_cpu:
            torch.cuda.is_available = lambda: False
            device = "cpu"

        # Setup configuration
        cfg = self._setup_cfg(device=device, dtype_override=dtype_override)

        # Create predictor (following CPU inference script)
        try:
            from detectron2.engine import DefaultPredictor

            self.predictor = DefaultPredictor(cfg)
            model = self.predictor.model
        except Exception as e:
            raise RuntimeError(
                f"Failed to create Panoptic-DeepLab model: {str(e)}"
            ) from e

        # Apply dtype override if specified and different from float32
        if dtype_override != torch.float32:
            model = model.to(dtype=dtype_override)

        model.eval()
        return model

    def load_inputs(self, **kwargs) -> List[Dict[str, torch.Tensor]]:
        """Load and return sample inputs for the model in detectron2 format.

        Args:
            **kwargs: Additional input-specific arguments.
                     - dtype_override: Override input dtype (e.g., torch.bfloat16)
                     - batch_size: Batch size for inputs (default: 1)
                     - image_size: Override image size (default: from config)

        Returns:
            List[Dict[str, torch.Tensor]]: Sample inputs in detectron2 format
        """
        config = self._variant_config
        dtype_override = kwargs.get("dtype_override", torch.float32)
        batch_size = kwargs.get("batch_size", 1)
        image_size = kwargs.get("image_size", config.image_size)

        inputs = []
        for i in range(batch_size):
            # Create random image tensor (C, H, W) - detectron2 format
            image = torch.randn(3, image_size[0], image_size[1], dtype=dtype_override)

            # Detectron2 expects a list of dicts, each with image, height, width
            input_dict = {
                "image": image,
                "height": image_size[0],
                "width": image_size[1],
            }
            inputs.append(input_dict)

        return inputs

    def predict(self, inputs: List[Dict[str, torch.Tensor]]) -> List[Dict[str, Any]]:
        """Run inference using the predictor (following CPU inference script pattern).

        Args:
            inputs: List of input dictionaries in detectron2 format

        Returns:
            List of prediction dictionaries
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = []
        for input_dict in inputs:
            # Convert to numpy array format expected by DefaultPredictor
            image = input_dict["image"]
            if isinstance(image, torch.Tensor):
                # Convert from (C, H, W) to (H, W, C) and to numpy
                image_np = image.permute(1, 2, 0).cpu().numpy()
                # Convert to BGR format (expected by DefaultPredictor)
                if image_np.shape[2] == 3:  # RGB
                    image_np = image_np[:, :, ::-1]  # RGB to BGR

                # Run inference
                predictions = self.predictor(image_np)
                results.append(predictions)
            else:
                raise ValueError("Input image must be a torch.Tensor")

        return results

    @classmethod
    def decode_output(cls, outputs: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Decode model outputs into human-readable format.

        Args:
            outputs: List of model output dictionaries from predict()
            **kwargs: Additional decoding arguments

        Returns:
            Decoded outputs with panoptic and semantic segmentation results
        """
        if not isinstance(outputs, list):
            outputs = [outputs]

        decoded_results = {"num_predictions": len(outputs), "predictions": []}

        for i, output in enumerate(outputs):
            decoded_pred = {"prediction_id": i}

            if isinstance(output, dict):
                for key, value in output.items():
                    if (
                        key == "panoptic_seg"
                        and isinstance(value, (tuple, list))
                        and len(value) == 2
                    ):
                        seg_map, segments_info = value
                        decoded_pred["panoptic_seg_shape"] = (
                            tuple(seg_map.shape) if hasattr(seg_map, "shape") else None
                        )
                        decoded_pred["num_segments"] = (
                            len(segments_info) if segments_info else 0
                        )
                        decoded_pred["segment_ids"] = (
                            [seg.get("id", -1) for seg in segments_info]
                            if segments_info
                            else []
                        )
                    elif key == "sem_seg" and hasattr(value, "shape"):
                        decoded_pred["sem_seg_shape"] = tuple(value.shape)
                        decoded_pred["num_semantic_classes"] = (
                            len(torch.unique(value))
                            if isinstance(value, torch.Tensor)
                            else None
                        )
                    elif key == "instances" and hasattr(value, "__len__"):
                        decoded_pred["num_instances"] = len(value)
                    else:
                        decoded_pred[key] = (
                            str(type(value))
                            if not isinstance(value, (int, float, str, bool))
                            else value
                        )

            decoded_results["predictions"].append(decoded_pred)

        return decoded_results

    def post_processing(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Post-process model outputs and print results."""
        decoded = self.decode_output(outputs)

        print("Panoptic-DeepLab Inference Results:")
        print(f"  Number of predictions: {decoded['num_predictions']}")

        for pred in decoded["predictions"]:
            pred_id = pred["prediction_id"]
            print(f"  Prediction {pred_id}:")
            for key, value in pred.items():
                if key != "prediction_id":
                    print(f"    {key}: {value}")

        return decoded
