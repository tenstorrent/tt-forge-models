# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Phi3 model loader implementation
"""

from transformers import (
    AutoTokenizer,
    Phi3Config,
    Phi3ForCausalLM,
    Phi3ForSequenceClassification,
    Phi3ForTokenClassification,
)
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel


class ModelLoader(ForgeModel):
    @classmethod
    def _get_model_info(cls, variant_name: str = None, task: str = "causal_lm"):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'microsoft/phi-3-mini-4k-instruct'.
            task: Task type for the model (causal_lm, token_classification, sequence_classification)

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "microsoft/phi-3-mini-4k-instruct"

        task_mapping = {
            "causal_lm": ModelTask.NLP_CAUSAL_LM,
            "token_classification": ModelTask.NLP_TOKEN_CLASSIFICATION,
            "sequence_classification": ModelTask.NLP_SEQUENCE_CLASSIFICATION,
        }

        return ModelInfo(
            model="phi3",
            variant=variant_name,
            group=ModelGroup.RED,
            task=task_mapping.get(task, ModelTask.NLP_CAUSAL_LM),
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None, task="causal_lm"):
        """Initialize ModelLoader with specified variant and task.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, uses 'microsoft/phi-3-mini-4k-instruct'.
            task: Task type for the model (causal_lm, token_classification, sequence_classification)
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = variant if variant else "microsoft/phi-3-mini-4k-instruct"
        self.task = task
        self.tokenizer = None

        # Task-specific input prompts based on test patterns
        self.input_prompts = {
            "causal_lm": "Africa is an emerging economy because",
            "token_classification": "HuggingFace is a company based in Paris and New York",
            "sequence_classification": "the movie was great!",
        }

    def load_model(self, dtype_override=None):
        """Load Phi3 model from Hugging Face."""

        # Initialize tokenizer first with default or overridden dtype
        tokenizer_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, **tokenizer_kwargs
        )

        # Add pad token for causal LM task if needed
        if self.task == "causal_lm" and self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Load pre-trained model from HuggingFace based on task
        model_kwargs = {"trust_remote_code": True, "use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self.task == "causal_lm":
            model = Phi3ForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        elif self.task == "token_classification":
            model = Phi3ForTokenClassification.from_pretrained(
                self.model_name, **model_kwargs
            )
        elif self.task == "sequence_classification":
            config = Phi3Config.from_pretrained(self.model_name)
            config_dict = config.to_dict()
            config_dict["use_cache"] = False
            config_dict["pad_token_id"] = None
            config = Phi3Config(**config_dict)
            model_kwargs["config"] = config
            model = Phi3ForSequenceClassification.from_pretrained(
                self.model_name, **model_kwargs
            )
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for Phi3 model"""

        # Get the appropriate input prompt for the task
        input_prompt = self.input_prompts.get(
            self.task, self.input_prompts["token_classification"]
        )

        # Data preprocessing based on task
        if self.task == "causal_lm":
            # For causal LM, use padding and truncation as shown in test
            inputs = self.tokenizer(
                input_prompt,
                return_tensors="pt",
                max_length=256,
                padding="max_length",
                truncation=True,
            )
            # Return input_ids and attention_mask as list
            return [inputs["input_ids"], inputs["attention_mask"]]
        else:
            # For token classification and sequence classification
            inputs = self.tokenizer(input_prompt, return_tensors="pt")
            # Return only input_ids as list
            return [inputs["input_ids"]]
