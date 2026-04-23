# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stub package to prevent the spacy/ model group directory from being treated as a
namespace package that shadows the real spaCy library. Provides a minimal Language
class so that libraries like `datasets` which check for spacy.Language don't crash.
"""


class Language:
    pass
