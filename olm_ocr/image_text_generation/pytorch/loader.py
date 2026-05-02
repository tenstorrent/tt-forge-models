# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
 OLM_OCR model loader implementation for image-to-text tasks.
"""
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


def _patch_qwen2_5_vl_tolist():
    """Patch Qwen2.5-VL methods to move device tensors to CPU before .tolist() calls.

    TT device does not support eager tensor readback (.tolist()). These methods use
    .tolist() only for Python-level control flow, not for main computation.
    """
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as m

    _orig_get_window_index = m.Qwen2_5_VisionTransformerPretrainedModel.get_window_index
    _orig_get_rope_index = m.Qwen2_5_VLModel.get_rope_index

    def _patched_rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw.cpu().tolist():
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids.to(grid_thw.device)].flatten(1)
        return rotary_pos_emb

    def _patched_get_window_index(self, grid_thw):
        device = grid_thw.device
        window_index, cu_window_seqlens = _orig_get_window_index(self, grid_thw.cpu())
        return window_index.to(device), cu_window_seqlens

    def _patched_get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        kwargs.pop("return_dict", None)
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).cpu().tolist()
        image_embeds = torch.split(vision_outputs.pooler_output, split_sizes)
        vision_outputs.pooler_output = image_embeds
        return vision_outputs

    def _patched_get_rope_index(
        self,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        attention_mask=None,
        **kwargs,
    ):
        device = input_ids.device if input_ids is not None else None
        position_ids, rope_deltas = _orig_get_rope_index(
            self,
            input_ids.cpu() if input_ids is not None else None,
            image_grid_thw.cpu() if image_grid_thw is not None else None,
            video_grid_thw.cpu() if video_grid_thw is not None else None,
            second_per_grid_ts,
            attention_mask.cpu() if attention_mask is not None else None,
            **kwargs,
        )
        if device is not None:
            position_ids = position_ids.to(device)
            rope_deltas = rope_deltas.to(device)
        return position_ids, rope_deltas

    m.Qwen2_5_VisionTransformerPretrainedModel.rot_pos_emb = _patched_rot_pos_emb
    m.Qwen2_5_VisionTransformerPretrainedModel.get_window_index = _patched_get_window_index
    m.Qwen2_5_VLModel.get_image_features = _patched_get_image_features
    m.Qwen2_5_VLModel.get_rope_index = _patched_get_rope_index


class ModelVariant(StrEnum):
    """Available OLM OCR model variants for image-to-text tasks."""

    OLM_OCR_7B_0225_Preview = "olmOCR-7B-0225-preview"
    OLM_OCR_7B_0725 = "olmOCR-7B-0725"
    OLM_OCR_7B_0825 = "olmOCR-7B-0825"
    OLM_OCR_2_7B_1025 = "olmOCR-2-7B-1025"
    OLM_OCR_7B_0825_FP8 = "olmOCR-7B-0825-FP8"


class ModelLoader(ForgeModel):
    """OLM_OCR model loader implementation for image-to-text tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.OLM_OCR_7B_0225_Preview: LLMModelConfig(
            pretrained_model_name="allenai/olmOCR-7B-0225-preview",
        ),
        ModelVariant.OLM_OCR_7B_0725: LLMModelConfig(
            pretrained_model_name="allenai/olmOCR-7B-0725",
        ),
        ModelVariant.OLM_OCR_7B_0825: LLMModelConfig(
            pretrained_model_name="allenai/olmOCR-7B-0825",
        ),
        ModelVariant.OLM_OCR_2_7B_1025: LLMModelConfig(
            pretrained_model_name="allenai/olmOCR-2-7B-1025",
        ),
        ModelVariant.OLM_OCR_7B_0825_FP8: LLMModelConfig(
            pretrained_model_name="allenai/olmOCR-7B-0825-FP8",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.OLM_OCR_7B_0225_Preview
    # Shared configuration parameters
    sample_text = "Give me a short introduction to large language model."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        group = ModelGroup.RED
        if variant in [
            ModelVariant.OLM_OCR_7B_0825_FP8,
        ]:
            group = ModelGroup.VULCAN

        return ModelInfo(
            model="olm_ocr",
            variant=variant,
            group=group,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load Processor for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the processor's default dtype.

        Returns:
            The loaded processor instance
        """
        # Initialize processor with dtype override if specified
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        # use_fast=False: Qwen2VLImageProcessor defaults to fast in transformers 5.x
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, use_fast=False, **kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the OLM_OCR model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The OLM_OCR model instance for image-to-text tasks.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name
        # Ensure tokenizer is loaded
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        # Load the model with dtype override if specified
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        _patch_qwen2_5_vl_tolist()

        # For FP8 compressed-tensors models: set run_compressed=False so weights are
        # dequantized to bfloat16 on load. TT does not support the FP8 matmul path.
        config = AutoConfig.from_pretrained(pretrained_model_name)
        qc = getattr(config, "quantization_config", None)
        if isinstance(qc, dict):
            qc["run_compressed"] = False
        elif qc is not None:
            qc.run_compressed = False
        model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # compressed-tensors leaves an instance-level forward on each quantized Linear
        # after decompression; remove it so TT-XLA's __torch_function__ is not shadowed.
        for m in model.modules():
            if "forward" in m.__dict__:
                del m.__dict__["forward"]

        self.config = model.config
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the OLM_OCR model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        # Use chat template for input text
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        return inputs

    def get_mesh_config(self, num_devices: int):

        # Prefer (1, N) when heads divide N, otherwise try (2, N/2)
        if self.config.num_attention_heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif (
            self.config.num_attention_heads % (num_devices // 2) == 0
            and num_devices % 2 == 0
        ):
            mesh_shape = (2, num_devices // 2)
        else:
            raise ValueError(
                f"Cannot evenly distribute {self.config.num_attention_heads} heads across {num_devices} devices"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        if self._variant == ModelVariant.OLM_OCR_7B_0225_Preview:
            for layer in model.model.visual.blocks:
                shard_specs[layer.attn.qkv.weight] = ("model", "batch")
                shard_specs[layer.attn.qkv.bias] = ("model",)
                shard_specs[layer.attn.proj.weight] = ("model", "batch")
                shard_specs[layer.attn.proj.bias] = ("model",)

                shard_specs[layer.mlp.fc1.weight] = ("model", "batch")
                shard_specs[layer.mlp.fc1.bias] = ("model",)
                shard_specs[layer.mlp.fc2.weight] = ("model", "batch")
                shard_specs[layer.mlp.fc2.bias] = ("model",)
        else:
            for layer in model.model.visual.blocks:
                shard_specs[layer.attn.qkv.weight] = ("model", "batch")
                shard_specs[layer.attn.qkv.bias] = ("model",)
                shard_specs[layer.attn.proj.weight] = ("model", "batch")
                shard_specs[layer.attn.proj.bias] = ("model",)

                shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.gate_proj.bias] = ("model",)
                shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.up_proj.bias] = ("model",)
                shard_specs[layer.mlp.down_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.down_proj.bias] = ("model",)

        for layer in model.model.language_model.layers:
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.q_proj.bias] = ("model",)
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.bias] = ("model",)
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.bias] = ("model",)
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            shard_specs[layer.self_attn.o_proj.bias] = ("model",)

            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the OLM_OCR model variant.

        Returns:
            The configuration object for the OLM_OCR model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.config
