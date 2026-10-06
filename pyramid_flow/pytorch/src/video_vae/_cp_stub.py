# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
No-op stubs for the upstream `utils` context-parallel helpers.

The vendored CausalVideoVAE code references `get_context_parallel_group` and a
few related helpers from the upstream top-level `utils` module. tt-xla runs
single-process inference (no context parallelism), so we replace these with
stubs that always report CP-disabled. Every `context_parallel_ops` helper
short-circuits and returns its input unchanged when the world size is 1, so the
group and rank accessors are never reached.

This mirrors `flux_modules/_sp_stub.py`, which does the same for the DiT's
sequence-parallel imports.
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
