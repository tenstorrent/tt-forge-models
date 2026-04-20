# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pathlib
import site
import sys


_SETUP_MAPPING_PATCH = {
    "old": (
        "        self._input_channels_mask: torch.Tensor | None = None\n"
        "        self._labram_ch_indices: torch.Tensor | None = None\n"
        "        try:"
    ),
    "new": (
        "        self._input_channels_mask: torch.Tensor | None = None\n"
        "        self._labram_ch_indices: torch.Tensor | None = None\n"
        '        self.register_buffer("_input_channels_indices", None)\n'
        '        self.register_buffer("_precomputed_input_chans", None)\n'
        "        try:"
    ),
}

_SETUP_MAPPING_BUFFERS_PATCH = {
    "old": (
        "        self._input_channels_mask, self._labram_ch_indices"
        " = self._get_channel_indices(\n"
        "            ch_names\n"
        "        )\n"
    ),
    "new": (
        "        self._input_channels_mask, self._labram_ch_indices"
        " = self._get_channel_indices(\n"
        "            ch_names\n"
        "        )\n"
        "        if self._input_channels_mask is not None:\n"
        "            self._input_channels_indices"
        " = torch.where(self._input_channels_mask)[0]\n"
        "        if self._labram_ch_indices is not None:\n"
        "            cls_idx = torch.tensor("
        "[0], dtype=self._labram_ch_indices.dtype)\n"
        "            self._precomputed_input_chans"
        " = torch.cat([cls_idx, self._labram_ch_indices + 1])\n"
    ),
}

_SELECT_CHANNELS_PATCH = {
    "old": (
        "        if ch_names is not None:\n"
        "            assert len(ch_names) == x.shape[1], (\n"
        '                "Length of ch_names must match '
        'number of channels in input tensor."\n'
        "            )\n"
        "            input_channels_mask, labram_ch_indices"
        " = self._get_channel_indices(ch_names)\n"
        "        else:\n"
        "            input_channels_mask = self._input_channels_mask\n"
        "            labram_ch_indices = self._labram_ch_indices\n"
        "\n"
        "        if input_channels_mask is None"
        " or labram_ch_indices is None:\n"
        "            raise ValueError(\n"
        '                "Channel information is not available.'
        " Please either provide \"\n"
        '                "the `ch_names` argument to the forward method,'
        " or ensure that channel information is provided \"\n"
        '                "during model initialization (via `chs_info`)."\n'
        "            )\n"
        "        if len(input_channels_mask) != x.shape[1]:\n"
        "            raise ValueError(\n"
        '                "Length of input_channels_mask does not match'
        ' number of channels in input tensor. "\n'
        '                "Please provide channel information via'
        " the `ch_names` argument to the forward method,"
        " or ensure that channel information is provided"
        ' during model initialization (via `chs_info`)."\n'
        "            )\n"
        "\n"
        "        # Select the channels that are available"
        " in LABRAM_CHANNEL_ORDER\n"
        "        x_available = x[:, input_channels_mask, :]\n"
        "\n"
        "        cls_index = torch.tensor(\n"
        "            [0],\n"
        "            device=labram_ch_indices.device,\n"
        "            dtype=labram_ch_indices.dtype,\n"
        "        )\n"
        "        input_chans = torch.cat("
        "[cls_index, labram_ch_indices + 1])\n"
        "        return x_available, input_chans"
    ),
    "new": (
        "        if ch_names is not None:\n"
        "            assert len(ch_names) == x.shape[1], (\n"
        '                "Length of ch_names must match '
        'number of channels in input tensor."\n'
        "            )\n"
        "            input_channels_mask, labram_ch_indices"
        " = self._get_channel_indices(ch_names)\n"
        "            x_available = x[:, input_channels_mask, :]\n"
        "            cls_index = torch.tensor(\n"
        "                [0],\n"
        "                device=labram_ch_indices.device,\n"
        "                dtype=labram_ch_indices.dtype,\n"
        "            )\n"
        "            input_chans = torch.cat("
        "[cls_index, labram_ch_indices + 1])\n"
        "            return x_available, input_chans\n"
        "\n"
        "        if self._input_channels_indices is None"
        " or self._precomputed_input_chans is None:\n"
        "            raise ValueError(\n"
        '                "Channel information is not available.'
        " Please either provide \"\n"
        '                "the `ch_names` argument to the forward method,'
        " or ensure that channel information is provided \"\n"
        '                "during model initialization (via `chs_info`)."\n'
        "            )\n"
        "\n"
        "        x_available = torch.index_select("
        "x, 1, self._input_channels_indices)\n"
        "        return x_available, self._precomputed_input_chans"
    ),
}

_PATCH_MARKER = "_input_channels_indices"


def _find_labram_source():
    """Locate braindecode/models/labram.py without importing braindecode."""
    candidates = site.getsitepackages() + [site.getusersitepackages()]
    for p in sys.path:
        if "site-packages" in p:
            candidates.append(p)
    for sp in candidates:
        candidate = pathlib.Path(sp) / "braindecode" / "models" / "labram.py"
        if candidate.exists():
            return candidate
    return None


def patch_braindecode_labram():
    """Patch braindecode's labram.py to use registered buffers and index_select.

    The original code stores channel indices as plain tensor attributes that
    don't move with the model to XLA devices, and uses boolean indexing that
    causes graph breaks in torch.compile/dynamo. This patch registers them
    as buffers and uses torch.index_select instead.
    """
    labram_path = _find_labram_source()
    if labram_path is None:
        return

    source = labram_path.read_text()

    if _PATCH_MARKER in source:
        return

    for patch in [
        _SETUP_MAPPING_PATCH,
        _SETUP_MAPPING_BUFFERS_PATCH,
        _SELECT_CHANNELS_PATCH,
    ]:
        if patch["old"] not in source:
            raise RuntimeError(
                f"Could not find expected code in {labram_path}. "
                "braindecode version may have changed."
            )
        source = source.replace(patch["old"], patch["new"], 1)

    labram_path.write_text(source)

    pyc = labram_path.with_suffix(".pyc")
    if pyc.exists():
        pyc.unlink()
    cache_dir = labram_path.parent / "__pycache__"
    if cache_dir.exists():
        for f in cache_dir.glob("labram*.pyc"):
            f.unlink()
