# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from tabulate import tabulate
from transformers import AutoImageProcessor
from loguru import logger


def run_and_print_results(framework_model, compiled_model, inputs, dtype_override=None):
    """
    Runs inference using both a framework model and a compiled model on a list of input images,
    then prints the results in a formatted table.

    Args:
        framework_model: The original framework-based model.
        compiled_model: The compiled version of the model.
        inputs: A list of images to process and classify.
        dtype_override: Optional torch.dtype to override the input's dtype.
    """
    label_dict = framework_model.config.id2label
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

    # Get device from compiled model
    device = next(compiled_model.parameters()).device
    
    # Only print results for non-CPU execution
    if device.type != "cpu":

        # Ensure framework model is on CPU for reference inference
        framework_model = framework_model.cpu()

        results = []
        for i, image in enumerate(inputs):
            processed_inputs = processor(image, return_tensors="pt")["pixel_values"]

            # Apply dtype override if provided
            if dtype_override is not None:
                processed_inputs = processed_inputs.to(dtype_override)

            # Run framework model on CPU with inputs on CPU
            cpu_inputs = processed_inputs.cpu()
            cpu_logits = framework_model(cpu_inputs)[0]
            cpu_conf, cpu_idx = cpu_logits.softmax(-1).max(-1)
            cpu_pred = label_dict.get(cpu_idx.item(), "Unknown")

            # Run compiled model on XLA device with inputs on XLA device
            xla_inputs = processed_inputs.to(device)
            tt_logits = compiled_model(xla_inputs)[0]
            tt_conf, tt_idx = tt_logits.softmax(-1).max(-1)
            tt_pred = label_dict.get(tt_idx.item(), "Unknown")

            results.append([i + 1, cpu_pred, cpu_conf.item(), tt_pred, tt_conf.item()])

        
        logger.info("=========== cpu vs {} results =================",device)
        logger.info(
        "\n" + tabulate(
            results,
            headers=[
                "Example",
                "CPU Prediction",
                "CPU Confidence",
                "Compiled Prediction",
                "Compiled Confidence",
            ],
            tablefmt="grid",
        )
        )
    
