# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
spaCy models package for Tenstorrent projects.

This __init__.py prevents the spacy/ model directory from being treated as a
bare namespace package that would shadow the real spacy library and break
third-party libraries (e.g. datasets._dill) that check for spacy.Language.
"""


class Language:
    """Stub satisfying datasets._dill's spacy.Language subclass check."""
