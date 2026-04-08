# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import importlib


def patch_transformers_output_capturing() -> None:
    """Patch transformers output capturing for torch.compile token edge cases."""
    output_capturing = importlib.import_module("transformers.utils.output_capturing")
    context_var_cls = output_capturing.CompileableContextVar

    if getattr(context_var_cls, "_tt_xla_none_token_patch", False):
        return

    def patched_get(self):
        # Keep the original behavior, but fix compile-mode flag assignment.
        if self.compiling:
            return self.global_var
        if output_capturing.is_torchdynamo_compiling():
            self.compiling = True
            return self.global_var
        return self.context_var.get()

    def patched_reset(self, token):
        # In compile mode, set() may return None instead of a ContextVar Token.
        if token is None:
            self.global_var = None
            self.compiling = False
            return
        if self.compiling:
            self.global_var = None
            self.compiling = False
        else:
            self.context_var.reset(token)

    context_var_cls.get = patched_get
    context_var_cls.reset = patched_reset
    context_var_cls._tt_xla_none_token_patch = True
