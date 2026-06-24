# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
No-op stubs for the upstream context-parallel helpers used by the vendored
Pyramid Flow CausalVideoVAE.

Vendored from https://github.com/jy0205/Pyramid-Flow (video_vae/), context-parallel
imports stubbed.

tt-xla runs single-process (single-device) decode, so context parallelism is never
initialized. These stubs report CP-disabled, and the scatter/gather helpers act as
pass-throughs so the non-parallel code path runs unchanged. Genuinely-collective
operations raise RuntimeError because they must never be hit single-process.
"""


def is_context_parallel_initialized() -> bool:
    return False


def get_context_parallel_group():
    return None


def get_context_parallel_world_size() -> int:
    return 1


def get_context_parallel_rank() -> int:
    return 0


def get_context_parallel_group_rank() -> int:
    return 0


def conv_scatter_to_context_parallel_region(input_, dim, kernel_size):
    # Single-device: nothing to scatter, return input unchanged.
    return input_


def conv_gather_from_context_parallel_region(input_, dim, kernel_size):
    # Single-device: nothing to gather, return input unchanged.
    return input_


def cp_pass_from_previous_rank(input_, dim, kernel_size):
    # Single-device: no previous rank to receive from, return input unchanged.
    return input_
