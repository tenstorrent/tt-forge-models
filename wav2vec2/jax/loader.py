# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
from transformers import AutoProcessor
from datasets import load_dataset

from src.modeling_flax_custom_wav2vec2 import FlaxWav2Vec2ForCTCCustom

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

model = FlaxWav2Vec2ForCTCCustom.from_pretrained("facebook/wav2vec2-base-960h")

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)

sample = ds[0]["audio"]

input_values = processor(
    sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="jax"
).input_values

logits = model(input_values).logits

predicted_ids = jnp.argmax(logits, axis=-1)

transcription = processor.decode(predicted_ids[0])
