# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
KEEP (Knowledge-Enhanced Evidence-based Pathology) model loader implementation for image-text similarity.
"""
import torch
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from typing import Optional
from PIL import Image

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


class ModelVariant(StrEnum):
    """Available KEEP model variants for pathology image-text similarity."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """KEEP model loader implementation for pathology image-text similarity tasks."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="Astaxanthin/KEEP",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="KEEP",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def _get_image_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the KEEP model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The KEEP model instance for pathology image-text similarity.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # Pre-import the remote module so its module-level code runs once.
        # modeling_keep.py does: timm.models.vision_transformer.LayerScale = RenameLayerScale
        # That class lacks device/dtype params that newer timm passes via Block(**dd).
        # After pre-import we replace the global with a compat subclass before from_pretrained
        # builds the model (module code won't re-execute on the second import).
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        import timm.models.vision_transformer as _timm_vit

        _KEEPModel = get_class_from_dynamic_module(
            "modeling_keep.KEEPModel",
            pretrained_model_name,
            trust_remote_code=True,
        )

        # Fix 1: RenameLayerScale doesn't accept device/dtype kwargs that newer
        # timm Block passes via dd = {'device': device, 'dtype': dtype}.
        _rename_cls = _timm_vit.LayerScale  # now == RenameLayerScale

        class _CompatLayerScale(_rename_cls):
            def __init__(
                self,
                dim: int,
                init_values: float = 1e-5,
                inplace: bool = False,
                device=None,
                dtype=None,
            ) -> None:
                super().__init__(dim, init_values=init_values, inplace=inplace)

        _timm_vit.LayerScale = _CompatLayerScale

        # Fix 2: KEEPModel.__init__ never calls post_init(), so transformers 5.x
        # _finalize_model_loading fails when accessing all_tied_weights_keys.
        _orig_keep_init = _KEEPModel.__init__

        def _keep_init_with_post_init(self, config):
            _orig_keep_init(self, config)
            self.post_init()

        _KEEPModel.__init__ = _keep_init_with_post_init

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the KEEP model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size (default 1).

        Returns:
            dict: Input tensors containing pixel values and token inputs.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        # Use a synthetic image to avoid load_dataset + dill/spacy namespace conflict
        image = Image.new("RGB", (224, 224))

        # Apply image preprocessing
        transform = self._get_image_transform()
        pixel_values = transform(image).unsqueeze(0)

        # Define text prompts for image-text similarity
        self.text_prompts = ["a photo of a cat", "a photo of a dog"]

        # Tokenize text inputs
        text_inputs = self.tokenizer(
            self.text_prompts,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # KEEP model forward() takes (image_inputs, text_inputs) where
        # text_inputs is a dict of tokenizer outputs
        inputs = {
            "image_inputs": pixel_values,
            "text_inputs": dict(text_inputs),
        }

        # Replicate tensors for batch size
        inputs["image_inputs"] = inputs["image_inputs"].repeat_interleave(
            batch_size, dim=0
        )
        for key in inputs["text_inputs"]:
            if torch.is_tensor(inputs["text_inputs"][key]):
                inputs["text_inputs"][key] = inputs["text_inputs"][
                    key
                ].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs["image_inputs"] = inputs["image_inputs"].to(dtype_override)

        return inputs

    def post_process(self, outputs):
        """Post-process KEEP model outputs to extract similarity scores.

        KEEP returns a dict with 'vision_features' and 'text_features' embeddings.
        Similarity is computed via dot product between normalized features.

        Args:
            outputs: Raw model output (dict with vision_features and text_features)
        """
        if self.text_prompts is None:
            self.text_prompts = ["a photo of a cat", "a photo of a dog"]

        vision_features = outputs["vision_features"]
        text_features = outputs["text_features"]

        # Compute cosine similarity (features are already L2-normalized by the model)
        similarity = vision_features @ text_features.T
        probs = similarity.softmax(dim=1)

        for i, text in enumerate(self.text_prompts):
            print(f"Probability of '{text}':", probs[0, i].item())

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        KEEP returns a dict with 'vision_features' and 'text_features'.

        Args:
            fwd_output: Output from the model's forward pass (dict)

        Returns:
            torch.Tensor: Concatenated flattened outputs for backward pass
        """
        if isinstance(fwd_output, dict):
            tensors = []
            for value in fwd_output.values():
                if isinstance(value, torch.Tensor):
                    tensors.append(value.flatten())
            if tensors:
                return torch.cat(tensors, dim=0)
        return fwd_output
