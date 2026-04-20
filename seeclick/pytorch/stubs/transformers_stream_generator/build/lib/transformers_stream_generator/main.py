# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
def init_stream_support():
    pass


class NewGenerationMixin:
    @staticmethod
    def generate(*args, **kwargs):
        raise NotImplementedError(
            "transformers_stream_generator is not supported with transformers>=5.0"
        )

    @staticmethod
    def sample_stream(*args, **kwargs):
        raise NotImplementedError(
            "transformers_stream_generator is not supported with transformers>=5.0"
        )


class StreamGenerationConfig:
    def __init__(self, *args, **kwargs):
        pass
