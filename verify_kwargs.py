#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Verification script to check that all load_model functions have **kwargs properly implemented.

Checks:
1. load_model signature has **kwargs
2. load_model uses keyword-only args (*, before first param after self)
3. If model_kwargs/pipe_kwargs dict exists, it has |= kwargs
4. If from_pretrained is called directly, it has **kwargs
"""
import re
import ast
import sys
from pathlib import Path
from typing import Tuple, List, Optional


class LoadModelChecker(ast.NodeVisitor):
    """AST visitor to check load_model function implementations."""

    def __init__(self, filepath: str, source: str):
        self.filepath = filepath
        self.source = source
        self.lines = source.split("\n")
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.load_model_found = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name == "load_model":
            self.load_model_found = True
            self._check_load_model(node)
        self.generic_visit(node)

    def _check_load_model(self, node: ast.FunctionDef):
        """Check a load_model function for proper **kwargs implementation."""
        args = node.args

        # Check 1: Has **kwargs
        if args.kwarg is None or args.kwarg.arg != "kwargs":
            self.errors.append(
                f"Line {node.lineno}: load_model missing **kwargs parameter"
            )
            return

        # Check 2: Has keyword-only args (kw_only_args should be non-empty if there are params after self)
        # This means there should be a * before the first parameter
        has_regular_args = len(args.args) > 1  # More than just 'self'
        has_kwonly_args = len(args.kwonlyargs) > 0

        if has_regular_args and not has_kwonly_args:
            # Has regular args but no kwonly args - missing the *
            self.errors.append(
                f"Line {node.lineno}: load_model has positional args but should use keyword-only (*, param=..., **kwargs)"
            )

        # Check 3: Look for from_pretrained calls and model_kwargs usage
        func_source = self._get_function_source(node)

        # Check for dict patterns like model_kwargs, pipe_kwargs
        dict_patterns = ["model_kwargs", "pipe_kwargs", "vae_kwargs"]
        for dict_name in dict_patterns:
            if f"{dict_name}" in func_source:
                # Check if dict is merged with kwargs
                if (
                    f"{dict_name} |= kwargs" not in func_source
                    and f"{dict_name}.update(kwargs)" not in func_source
                ):
                    # Check if it's used with from_pretrained for MODEL loading (not config)
                    # Look for pattern like: Model.from_pretrained(..., **model_kwargs)
                    # Skip if only used for Config.from_pretrained
                    if f"**{dict_name}" in func_source:
                        # Check if this dict is passed to a Model.from_pretrained, not just Config
                        model_pretrained_with_dict = False
                        for line in func_source.split("\n"):
                            if f"**{dict_name}" in line and "from_pretrained" in line:
                                # Check it's not a Config call
                                if "Config" not in line:
                                    model_pretrained_with_dict = True
                                    break
                        if model_pretrained_with_dict:
                            self.errors.append(
                                f"Line {node.lineno}: load_model has {dict_name} dict but missing '{dict_name} |= kwargs'"
                            )

        # Check 4: from_pretrained calls without **kwargs or **dict
        # Only check model loading calls, skip tokenizer/processor/config calls
        # Use a pattern that finds from_pretrained and looks for ** on the same or next few lines
        lines = func_source.split("\n")
        for i, line in enumerate(lines):
            if ".from_pretrained(" in line:
                # Get context - current line plus next few lines until we find closing paren
                context_lines = [line]
                paren_count = line.count("(") - line.count(")")
                j = i + 1
                while paren_count > 0 and j < len(lines):
                    context_lines.append(lines[j])
                    paren_count += lines[j].count("(") - lines[j].count(")")
                    j += 1

                call_context = "\n".join(context_lines)

                # Extract the class name
                match = re.search(r"(\w+)\.from_pretrained\(", line)
                if not match:
                    continue
                class_name = match.group(1)

                # Skip tokenizer, processor, and config calls - they don't need **kwargs
                skip_patterns = [
                    "Tokenizer",
                    "tokenizer",
                    "Processor",
                    "processor",
                    "Config",
                    "config",
                    "AutoFeatureExtractor",
                    "FeatureExtractor",
                    "ImageProcessor",
                ]
                if any(skip in class_name for skip in skip_patterns):
                    continue

                # Check if there's any ** in the call context
                if "**" not in call_context:
                    call_line = node.lineno + i
                    self.errors.append(
                        f"Line ~{call_line}: from_pretrained call without **kwargs or **dict: {line.strip()[:60]}..."
                    )

    def _get_function_source(self, node: ast.FunctionDef) -> str:
        """Get the source code of a function."""
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, "end_lineno") else start_line + 50
        return "\n".join(self.lines[start_line:end_line])


def check_file(filepath: Path) -> Tuple[List[str], List[str], bool]:
    """Check a single file for load_model compliance.

    Returns:
        Tuple of (errors, warnings, has_load_model)
    """
    try:
        with open(filepath, "r") as f:
            source = f.read()
    except Exception as e:
        return [f"Could not read file: {e}"], [], False

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"Syntax error: {e}"], [], False

    checker = LoadModelChecker(str(filepath), source)
    checker.visit(tree)

    return checker.errors, checker.warnings, checker.load_model_found


def main():
    # Find all loader.py files
    repo_root = Path(__file__).parent
    loader_files = list(repo_root.rglob("**/loader.py"))

    # Exclude this script and any test files
    loader_files = [
        f
        for f in loader_files
        if "verify_kwargs" not in str(f) and "test_" not in str(f)
    ]

    print(f"Checking {len(loader_files)} loader.py files...\n")

    total_errors = 0
    total_warnings = 0
    files_with_errors = []
    files_without_load_model = []

    for filepath in sorted(loader_files):
        rel_path = filepath.relative_to(repo_root)
        errors, warnings, has_load_model = check_file(filepath)

        if not has_load_model:
            files_without_load_model.append(str(rel_path))
            continue

        if errors:
            files_with_errors.append(str(rel_path))
            print(f"❌ {rel_path}")
            for error in errors:
                print(f"   {error}")
                total_errors += 1

        if warnings:
            for warning in warnings:
                print(f"   ⚠️  {warning}")
                total_warnings += 1

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total files checked: {len(loader_files)}")
    print(f"Files with load_model: {len(loader_files) - len(files_without_load_model)}")
    print(f"Files with errors: {len(files_with_errors)}")
    print(f"Total errors: {total_errors}")
    print(f"Total warnings: {total_warnings}")

    if files_with_errors:
        print(f"\n❌ Files needing fixes:")
        for f in files_with_errors:
            print(f"   {f}")
        sys.exit(1)
    else:
        print(f"\n✅ All load_model functions have proper **kwargs implementation!")
        sys.exit(0)


if __name__ == "__main__":
    main()
