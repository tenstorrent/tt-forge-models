# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
enguard/small-guard-32m-en-prompt-violence-binary-moderation model loader for
binary text classification of prompt violence.

This model is a Model2Vec classifier (StaticModelPipeline) distilled from
minishlab/potion-base-32m with a logistic regression head trained on the
prompt-violence-binary split of the enguard/multi-lingual-prompt-moderation
dataset.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from ....base import ForgeModel


class EnguardStaticPipelineTorchModel(nn.Module):
    """Wraps a Model2Vec StaticModelPipeline as a torch.nn.Module.

    The pipeline is: token embedding lookup -> attention-masked mean pooling ->
    optional L2 normalization -> sklearn classifier head (scaler + logistic
    regression), reimplemented in pure torch for hardware compilation.
    """

    def __init__(self, pipeline):
        super().__init__()
        static_model = pipeline.model
        embedding_tensor = torch.from_numpy(static_model.embedding.copy()).float()
        self.embedding = nn.Embedding.from_pretrained(embedding_tensor, freeze=True)
        self.normalize = bool(static_model.normalize)

        mean, scale, weight, bias = _extract_head_params(pipeline.head)
        self.register_buffer("scaler_mean", torch.from_numpy(mean).float())
        self.register_buffer("scaler_scale", torch.from_numpy(scale).float())

        num_classes, embedding_dim = weight.shape
        self.classifier = nn.Linear(embedding_dim, num_classes)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.from_numpy(weight).float())
            self.classifier.bias.copy_(torch.from_numpy(bias).float())

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        token_embeddings = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = summed / counts
        if self.normalize:
            norm = torch.clamp(torch.norm(pooled, dim=1, keepdim=True), min=1e-32)
            pooled = pooled / norm
        scaled = (pooled - self.scaler_mean) / self.scaler_scale
        return self.classifier(scaled)


def _extract_head_params(head):
    """Extract (mean, scale, weight, bias) from an sklearn classification head.

    Handles both bare classifiers and sklearn Pipelines, with or without a
    StandardScaler preprocessing step. For binary LogisticRegression (single
    row of coefficients) the returned weight/bias are expanded to two-class
    form so argmax matches sklearn.predict.
    """
    steps = getattr(head, "steps", None)
    if steps is None:
        estimator = head
        scaler = None
    else:
        estimator = steps[-1][1]
        scaler = next(
            (
                step
                for _, step in steps[:-1]
                if hasattr(step, "mean_") and hasattr(step, "scale_")
            ),
            None,
        )

    weight = np.asarray(estimator.coef_, dtype=np.float32)
    bias = np.asarray(estimator.intercept_, dtype=np.float32)
    if weight.shape[0] == 1:
        weight = np.vstack([-weight, weight])
        bias = np.concatenate([-bias, bias])

    num_features = weight.shape[1]
    if scaler is not None:
        mean = np.asarray(scaler.mean_, dtype=np.float32)
        scale = np.asarray(scaler.scale_, dtype=np.float32)
    else:
        mean = np.zeros(num_features, dtype=np.float32)
        scale = np.ones(num_features, dtype=np.float32)
    return mean, scale, weight, bias


class ModelVariant(StrEnum):
    """Available enguard small-guard-32m prompt-violence moderation variants."""

    SMALL_GUARD_32M_EN_PROMPT_VIOLENCE = (
        "enguard/small-guard-32m-en-prompt-violence-binary-moderation"
    )


class ModelLoader(ForgeModel):
    """enguard small-guard-32m prompt-violence binary moderation loader."""

    _VARIANTS = {
        ModelVariant.SMALL_GUARD_32M_EN_PROMPT_VIOLENCE: LLMModelConfig(
            pretrained_model_name=(
                "enguard/small-guard-32m-en-prompt-violence-binary-moderation"
            ),
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMALL_GUARD_32M_EN_PROMPT_VIOLENCE

    sample_text = (
        "You are an assistant. Help me understand how to safely resolve a "
        "conflict with a neighbor."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self._pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="enguard-small-guard-32m-en-prompt-violence-binary-moderation",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self):
        if self._pipeline is None:
            from model2vec.inference import StaticModelPipeline

            self._pipeline = StaticModelPipeline.from_pretrained(
                self._variant_config.pretrained_model_name
            )
        return self._pipeline

    def load_model(self, *, dtype_override=None, **kwargs):
        pipeline = self._load_pipeline()
        model = EnguardStaticPipelineTorchModel(pipeline)
        model.eval()
        if dtype_override is not None:
            model = model.to(dtype=dtype_override)
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, sentence=None):
        pipeline = self._load_pipeline()
        static_model = pipeline.model

        if sentence is None:
            sentence = self.sample_text

        max_length = getattr(self._variant_config, "max_length", 128)

        token_ids_list = static_model.tokenize([sentence], max_length=max_length)
        token_ids = list(token_ids_list[0])

        if len(token_ids) >= max_length:
            token_ids = token_ids[:max_length]
            attention_mask = [1] * max_length
        else:
            attention_mask = [1] * len(token_ids) + [0] * (max_length - len(token_ids))
            token_ids = token_ids + [0] * (max_length - len(token_ids))

        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode_output(self, co_out, framework_model=None):
        logits = co_out[0] if isinstance(co_out, (list, tuple)) else co_out
        predicted_class_id = int(logits.argmax(-1).item())
        labels = self._class_labels()
        if 0 <= predicted_class_id < len(labels):
            print(f"Predicted label: {labels[predicted_class_id]}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")

    def _class_labels(self):
        pipeline = self._pipeline
        if pipeline is None:
            return []
        head = pipeline.head
        estimator = head.steps[-1][1] if hasattr(head, "steps") else head
        classes = getattr(estimator, "classes_", None)
        return list(classes) if classes is not None else []
