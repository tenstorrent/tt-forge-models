# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Stub classes to satisfy gliner's optional llm2vec import check.
# The local `llm2vec/` directory is a namespace package that gliner detects
# as the llm2vec pip package. These stubs prevent the ImportError while
# remaining unused for DeBERTa-based models (which don't use DECODER_MODEL_MAPPING).


class GemmaBiModel:
    pass


class LlamaBiModel:
    pass


class MistralBiModel:
    pass


class Qwen2BiModel:
    pass
