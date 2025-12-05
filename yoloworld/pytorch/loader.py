# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLO-World model loader implementation
"""

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .src.utils import init_detector,Config,get_test_pipeline_cfg
from ...tools.utils import get_file

class ModelVariant(StrEnum):
    """Available YOLO WORLD model variants."""

    SMALL_1 = "small_640"
    SMALL_2 = "small_1280"
    MEDIUM_1 = "medium_640"
    MEDIUM_2 = "medium_1280"
    LARGE_1 = "large_640"
    LARGE_2 = "large_1280"
    XLARGE_1 = "xlarge_640"

class ModelLoader(ForgeModel):
    """YOLO WORLD model loader."""

    _VARIANTS = {
        ModelVariant.SMALL_1: ModelConfig(
            pretrained_model_name="small_640",
        ),
        ModelVariant.SMALL_2: ModelConfig(
            pretrained_model_name="small_1280",
        ),
        ModelVariant.MEDIUM_1: ModelConfig(
            pretrained_model_name="medium_640",
        ),
        ModelVariant.MEDIUM_2: ModelConfig(
            pretrained_model_name="medium_1280",
        ),
        ModelVariant.LARGE_1: ModelConfig(
            pretrained_model_name="large_640",
        ),
        ModelVariant.LARGE_2: ModelConfig(
            pretrained_model_name="large_1280",
        ),
        ModelVariant.XLARGE_1: ModelConfig(
            pretrained_model_name="xlarge_640",
        ),    
    }

    DEFAULT_VARIANT = ModelVariant.SMALL_1
    DEFAULT_TOPK = 100
    DEFAULT_THRESHOLD = 0.005
    DEFAULT_TEXTS = 'person,bus'

    # LOAD FROM - TO ADD

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        topk: Optional[int] = None,
        threshold: Optional[float] = None,
        texts: Optional[str] = None,
    ):
        super().__init__()
        self.variant = variant or self.DEFAULT_VARIANT
        self.topk = topk or self.DEFAULT_TOPK
        self.threshold = threshold or self.DEFAULT_THRESHOLD
        self.texts = texts or self.DEFAULT_TEXTS
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = self.DEFAULT_VARIANT
        return ModelInfo(
            model="yoloworld",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )
    
    def _load_data_config(self):
        config="/proj_sw/user_dev/akannan_new/3_dec_yoloworld_bgd/tt-forge-models/YOLO_World/reference/configs/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
        self.config = Config.fromfile(config)

        # to load the configs according to variants
    
    def load_model(self, dtype_override=None):
        checkpoint = get_file(f"test_files/pytorch/yoloworld/{self.variant}.pth")
        self.config.load_from = checkpoint
        self.model = init_detector(self.config,checkpoint,device="cpu")
        if dtype_override is not None:
            self.model = self.model.to(dtype_override)
        return self.model
    
    def load_inputs(self, dtype_override=None, batch_size=1):
        self.image_file = get_file(
            "https://ultralytics.com/images/bus.jpg"
        )
        test_pipeline_cfg = get_test_pipeline_cfg(cfg=self.config)
        data_info = dict(img_id=0, img_path=image_file, texts=self.texts)
        if dtype_override is not None:
            data_info['inputs'] = data_info['inputs'].to(dtype_override)
        data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),data_samples=[data_info['data_samples']])
        return data_batch
    

    def post_process(self, output, output_dir):
        pred_instances = output.pred_instances
        pred_instances = pred_instances[pred_instances.scores.float() >
                                        self.threshold]
        if len(pred_instances.scores) > self.topk:
            indices = pred_instances.scores.float().topk(self.topk)[1]
            pred_instances = pred_instances[indices]
        if 'masks' in pred_instances:
            masks = pred_instances['masks']
        else:
            masks = None
        detections = sv.Detections(xyxy=pred_instances['bboxes'],
                               class_id=pred_instances['labels'],
                               confidence=pred_instances['scores'],
                               mask=masks)
        image = cv2.imread(self.image_file)
        anno_image = image.copy()
        image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
        image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
        cv2.imwrite(osp.join(output_dir, osp.basename(self.image_file)), image)
        