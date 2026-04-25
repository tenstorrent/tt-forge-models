# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


# Stub to prevent datasets._dill from failing when it checks `spacy.Language`.
# Without this, the local spacy/ namespace package shadows the real spacy
# package and causes AttributeError in datasets fingerprinting.
class Language:
    pass
