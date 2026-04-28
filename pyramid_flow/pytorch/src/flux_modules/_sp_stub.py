# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
No-op stubs for the upstream `trainer_misc` sequence-parallel helpers.

The vendored Pyramid Flow model code references `is_sequence_parallel_initialized`
and a few related helpers from the upstream `trainer_misc` package. tt-xla runs
single-process inference (no sequence parallelism), so we replace these with
stubs that always report SP-disabled and never get exercised.
"""


def is_sequence_parallel_initialized() -> bool:
    return False


def get_sequence_parallel_group():
    return None


def get_sequence_parallel_world_size() -> int:
    return 1


def get_sequence_parallel_rank() -> int:
    return 0


def all_to_all(*args, **kwargs):
    raise RuntimeError(
        "all_to_all should not be reached when sequence parallelism is disabled"
    )
