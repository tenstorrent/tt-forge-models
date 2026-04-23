# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
spaCy model implementations for Tenstorrent projects.

This __init__.py converts the spacy/ directory from a namespace package to a
regular package, preventing datasets._dill from failing on spacy.Language
when the real spaCy library is not installed.
"""
from .es_core_news_md import ModelLoader


class Language:
    """Stub for spacy.Language so datasets._dill type-checks do not raise AttributeError."""
