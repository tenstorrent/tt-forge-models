# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import pytest
import torch

from third_party.tt_forge_models.boltz.pytorch.loader import ModelLoader, ModelVariant


@pytest.mark.parametrize("variant", [ModelVariant.BOLTZ2])
def test_boltz_cpu_forward_minimal(variant):

    # Load model on CPU with minimal steps for a quick smoke forward
    loader = ModelLoader(variant)
    model = loader.load_model(
        # dtype_override=torch.float32,
        # use_kernels=False,
        # write_confidence_summary=False,
        # write_full_pae=False,
        # write_full_pde=False,
        # skip_run_structure=True,
        # confidence_prediction=False,
        # cache_dir="/proj_sw/user_dev/mramanathan/bgdlab19_nov13_xla/tt-xla/third_party/tt_forge_models/boltz/boltz_results_fast_protein",
    )
    print("Model loaded", model)
    # Prepare one batch of inputs from processed data
    [feats] = loader.load_inputs(
        # cache_dir="/proj_sw/user_dev/mramanathan/bgdlab19_nov13_xla/tt-xla/third_party/tt_forge_models/boltz_git/new",
        # # cache_dir = "/proj_sw/user_dev/mramanathan/bgdlab19_nov13_xla/tt-xla/third_party/tt_forge_models/boltz_git/boltz_results_fast_protein_custom",
        # data="/proj_sw/user_dev/mramanathan/bgdlab19_nov13_xla/tt-xla/tests/torch/single_chip/models/boltz/fast_protein.yaml",
        # out_dir="/proj_sw/user_dev/mramanathan/bgdlab19_nov13_xla/tt-xla/third_party/tt_forge_models/boltz/boltz_results_fast_protein",
        # num_workers=0,
        # affinity=False,
        # dtype_override=torch.float32,
    )

    # Minimal forward invocation
    with torch.no_grad():
        # print("loader inputs", feats)
        out = model(
            feats,
            # recycling_steps=3,
            # num_sampling_steps=200,
            # diffusion_samples=1,
            # max_parallel_samples=None,
            # run_confidence_sequentially=True,
        )
    print("out inside test_boltz_cpu_forward_minimal", out)
    # Basic sanity checks on outputs
    assert isinstance(out, dict)
    assert "s" in out and "z" in out and "pdistogram" in out
