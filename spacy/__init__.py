# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# This file prevents Python from treating the spacy/ model directory as a bare
# namespace package that shadows the real spaCy PyPI package.  Libraries such as
# HuggingFace `datasets` guard spacy-specific pickling logic with
#   if "spacy" in sys.modules: … issubclass(obj_type, spacy.Language) …
# which raises AttributeError when `spacy` resolves to a featureless namespace
# package.  The stub Language class below satisfies that isinstance check.


class Language:
    """Minimal stub so that spacy.Language is a valid class for issubclass checks."""
