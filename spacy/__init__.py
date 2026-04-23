# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stub for spacy namespace package to prevent import conflicts.
The real spacy package may not be installed; this stub exposes Language
so that libraries (e.g. datasets) that do `if "spacy" in sys.modules`
followed by `spacy.Language` don't raise AttributeError.
"""


class Language:
    pass
