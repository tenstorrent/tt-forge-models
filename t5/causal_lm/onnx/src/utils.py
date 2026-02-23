# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch


class T5Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, decoder_input_ids):
        inputs = {"input_ids": input_ids, "decoder_input_ids": decoder_input_ids}
        output = self.model(**inputs)
        return output


def pad_inputs(inputs, max_new_tokens=512):
    batch_size, seq_len = inputs.shape
    max_seq_len = seq_len + max_new_tokens
    padded_inputs = torch.zeros(
        (batch_size, max_seq_len), dtype=inputs.dtype, device=inputs.device
    )
    padded_inputs[:, :seq_len] = inputs
    return padded_inputs, seq_len
