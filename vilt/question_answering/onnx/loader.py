# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ViLT ONNX model loader.
"""

import torch

# Reuse the PyTorch ModelLoader as the base
from ...pytorch.loader import ModelLoader as PyTorchModelLoader
from ....tools.utils import export_torch_model_to_onnx


class ViltEmbeddingWrapper(torch.nn.Module):
    """Build text-vision embeddings consumed by the ViLT encoder."""

    def __init__(self, model):
        super().__init__()
        self.vilt_model = model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        pixel_mask=None,
        inputs_embeds=None,
        image_embeds=None,
        image_token_type_idx=None,
    ):
        embeddings, masks = self.vilt_model.vilt.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            image_token_type_idx=image_token_type_idx,
        )
        return embeddings, masks


class ViltQaModelWrapper(torch.nn.Module):
    """Expose ViLT question-answering logits from embedding inputs."""

    def __init__(self, model):
        super().__init__()
        self.vilt_model = model

    def forward(self, embedding_output, attention_mask, head_mask=None):
        head_mask = self.vilt_model.vilt.get_head_mask(
            head_mask, self.vilt_model.vilt.config.num_hidden_layers
        )

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
            torch.float32
        ).min

        encoder_outputs = self.vilt_model.vilt.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            return_dict=False,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.vilt_model.vilt.layernorm(sequence_output)
        pooled_output = (
            self.vilt_model.vilt.pooler(sequence_output)
            if self.vilt_model.vilt.pooler is not None
            else None
        )

        return self.vilt_model.classifier(pooled_output)


class ModelLoader(PyTorchModelLoader):
    """ViLT ONNX loader that inherits from the PyTorch loader."""

    def _prepare_embedding_inputs(self, **kwargs):
        if not hasattr(self, "torch_loader"):
            self.torch_loader = PyTorchModelLoader(variant=self._variant)

        if getattr(self.torch_loader, "model", None) is None:
            torch_model = self.torch_loader.load_model(**kwargs)
        else:
            torch_model = self.torch_loader.model

        raw_inputs = self.torch_loader.load_inputs(**kwargs)
        embedding_model = ViltEmbeddingWrapper(torch_model)
        embedding_model.eval()

        with torch.no_grad():
            embedding_output, attention_mask = embedding_model(**raw_inputs)

        return [
            embedding_output.detach().cpu(),
            attention_mask.detach().cpu().to(torch.float32),
        ]

    def load_model(self, onnx_tmp_path, **kwargs):
        """Load ViLT as a torch model, export to ONNX, then load and return the ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        # default variant ModelVariant.VQA is used if no variant is provided
        self.torch_loader = PyTorchModelLoader(variant=self._variant)
        torch_model = self.torch_loader.load_model(**kwargs)
        self.model = getattr(self.torch_loader, "model", torch_model)

        wrapped_model = ViltQaModelWrapper(torch_model)
        wrapped_model.eval()
        inputs = self._prepare_embedding_inputs(**kwargs)
        model_name = self.torch_loader._variant_config.pretrained_model_name

        return export_torch_model_to_onnx(
            wrapped_model,
            onnx_tmp_path,
            tuple(inputs),
            model_name,
        )

    def load_inputs(self, **kwargs):
        """Load and return ViLT QA embedding inputs for ONNX execution."""
        return self._prepare_embedding_inputs(**kwargs)
