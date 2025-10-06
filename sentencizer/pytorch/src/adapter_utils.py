# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ast
import contextlib
import copy
import functools
import hashlib
import importlib
import inspect
import io
import itertools
import json
import logging
import math
import os
import re
import shutil
import sys
import tarfile
import tempfile
import threading
import types
import warnings
from abc import ABC, ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, dataclass, field, replace
from enum import Enum
from functools import partial
from hashlib import sha256
from os import mkdir
from os.path import basename, exists, isdir, isfile, join
from pathlib import Path
from typing import (
    Any,
    Callable,
    Collection,
    ContextManager,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)
from zipfile import ZipFile, is_zipfile

import numpy as np
import requests
import torch
import torch.nn.functional as F
from filelock import FileLock
from packaging.version import Version
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.activations import get_activation
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.auto.auto_factory import getattribute_from_module
from transformers.models.auto.configuration_auto import model_type_to_module_name
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaOutput,
    XLMRobertaSdpaSelfAttention,
    XLMRobertaSelfAttention,
    XLMRobertaSelfOutput,
)
from transformers.pytorch_utils import Conv1D
from transformers.utils import (
    cached_file,
    http_user_agent,
    is_accelerate_available,
    is_remote_url,
)
from transformers.utils.generic import working_or_temp_dir
from transformers.utils.import_utils import is_torchvision_available

from huggingface_hub import HfApi, HfFolder, snapshot_download
from huggingface_hub.file_download import http_get
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    hf_raise_for_status,
)
from requests.exceptions import HTTPError

try:
    from bitsandbytes.nn import Int8Params, Linear4bit, Linear8bitLt, Params4bit

    bitsandbytes_available = True
except ImportError:
    bitsandbytes_available = False

try:
    from safetensors.torch import load_file, save_file

    safetensors_available = True
except ImportError:
    safetensors_available = False

try:
    import requests
    from filelock import FileLock
    from huggingface_hub import HfApi, HfFolder, snapshot_download
    from huggingface_hub.file_download import http_get
    from huggingface_hub.utils import (
        EntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
        hf_raise_for_status,
    )
    from requests.exceptions import HTTPError
    from transformers.utils import http_user_agent, is_remote_url

    huggingface_hub_available = True
except ImportError:
    huggingface_hub_available = False
import logging
from typing import Iterable, Tuple

import torch.nn as nn

logger = logging.getLogger(__name__)

# ===== Constants from utils.py =====

CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAME = "pytorch_adapter.bin"
SAFE_WEIGHTS_NAME = "adapter.safetensors"
HEAD_CONFIG_NAME = "head_config.json"
HEAD_WEIGHTS_NAME = "pytorch_model_head.bin"
SAFE_HEAD_WEIGHTS_NAME = "model_head.safetensors"
ADAPTERFUSION_CONFIG_NAME = "adapter_fusion_config.json"
ADAPTERFUSION_WEIGHTS_NAME = "pytorch_model_adapter_fusion.bin"
SAFE_ADAPTERFUSION_WEIGHTS_NAME = "model_adapter_fusion.safetensors"
EMBEDDING_FILE = "embedding.pt"
TOKENIZER_PATH = "tokenizer"
SETUP_CONFIG_NAME = "adapter_setup.json"
INTERFACE_CONFIG_NAME = "adapter_interface.json"

ADAPTER_HUB_URL = "https://raw.githubusercontent.com/Adapter-Hub/Hub/master/dist/v2/"
ADAPTER_HUB_INDEX_FILE = ADAPTER_HUB_URL + "index/{}.json"
ADAPTER_HUB_CONFIG_FILE = ADAPTER_HUB_URL + "architectures.json"
ADAPTER_HUB_ALL_FILE = ADAPTER_HUB_URL + "all.json"
ADAPTER_HUB_ADAPTER_ENTRY_JSON = ADAPTER_HUB_URL + "adapters/{}/{}.json"

# the download cache
torch_cache_home = os.getenv(
    "TORCH_HOME",
    os.path.join(os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "torch"),
)
ADAPTER_CACHE = join(torch_cache_home, "adapters")

# these keys are ignored when calculating the config hash
ADAPTER_CONFIG_HASH_IGNORE = []

# old: new
ACTIVATION_RENAME = {
    "gelu": "gelu_new",
    "gelu_orig": "gelu",
}
ADAPTER_CONFIG_HASH_IGNORE_DEFAULT = {
    "phm_layer": True,
    "phm_dim": 4,
    "factorized_phm_W": True,
    "shared_W_phm": False,
    "shared_phm_rule": True,
    "factorized_phm_rule": False,
    "phm_c_init": "normal",
    "phm_init_range": 0.0001,
    "learn_phm": True,
    "hypercomplex_nonlinearity": "glorot-uniform",
    "phm_rank": 1,
    "phm_bias": True,
    "init_weights": "bert",
    "scaling": 1.0,
}
ADAPTER_CONFIG_STRING_PATTERN = re.compile(
    r"^(?P<name>[^\[\]\|\n]+)(?:\[(?P<kvs>.*)\])?$"
)

# ===== AdapterType from utils.py =====
class AdapterType(str, Enum):
    """Models all currently available model adapter types."""

    text_task = "text_task"
    text_lang = "text_lang"

    @classmethod
    def has(cls, value):
        return value in cls.__members__.values()

    def __repr__(self):
        return self.value


@dataclass
class AdapterInfo:
    """
    Holds information about an adapter publicly available on the Hub. Returned by
    :func:`list_adapters()`.

    Args:
        source (str): The source repository of this adapter. Always 'hf' for adapters available on HF Model Hub.
        adapter_id (str): The unique identifier of this adapter.
        model_name (str, optional): The identifier of the model this adapter was trained for.
        task (str, optional): The task this adapter was trained for.
        subtask (str, optional): The subtask or dataset this adapter was trained on.
        username (str, optional): The username of author(s) of this adapter.
        adapter_config (dict, optional): The configuration dictionary of this adapter.
    """

    source: str
    adapter_id: str
    model_name: Optional[str] = None
    task: Optional[str] = None
    subtask: Optional[str] = None
    username: Optional[str] = None
    adapter_config: Optional[dict] = None
    sha1_checksum: Optional[str] = None


def _minimize_dict(d):
    if isinstance(d, Mapping):
        return {k: _minimize_dict(v) for (k, v) in d.items() if v}
    else:
        return d


def get_adapter_config_hash(config, length=16, ignore_params=[]):
    """
    Calculates the hash of a given adapter configuration which is used to identify this configuration.

    Returns:
        str: The resulting hash of the given config dict.
    """
    minimized_config = _minimize_dict(
        {
            k: v
            for (k, v) in config.items()
            if k not in ADAPTER_CONFIG_HASH_IGNORE + ignore_params
        }
    )
    # ensure hash is kept consistent to previous versions
    for name, default in ADAPTER_CONFIG_HASH_IGNORE_DEFAULT.items():
        if minimized_config.get(name, None) == default:
            del minimized_config[name]
    dict_str = json.dumps(minimized_config, sort_keys=True)
    h = hashlib.sha1()
    h.update(dict_str.encode(encoding="utf-8"))
    return h.hexdigest()[:length]


def url_to_filename(url: str, etag: Optional[str] = None) -> str:
    raise EnvironmentError("Remote utilities disabled in sentencizer build")


def download_cached(*args, **kwargs):
    raise EnvironmentError("Remote utilities disabled in sentencizer build")


def parse_adapter_config_string(config_string: str) -> List[Tuple[str, dict]]:
    """
    Parses an adapter configuration string into a list of tuples. Each tuple constists of an adapter config identifier
    and dictionary.
    """
    # First split by "|" into individual adapter configs
    config_string_chunks = config_string.split("|")
    # Now match each adapter config against the regex
    adapter_configs = []
    for config_string_chunk in config_string_chunks:
        match = re.match(ADAPTER_CONFIG_STRING_PATTERN, config_string_chunk.strip())
        if not match or not match.group("name"):
            raise ValueError(
                f"Invalid adapter config string format: '{config_string_chunk}'."
            )
        name = match.group("name")
        if match.group("kvs"):
            kvs = match.group("kvs")
            kvs = re.sub(r"(\w+)=", r"'\1':", kvs)
        else:
            kvs = ""
        # Now evaluate key-value pairs as Python dict
        try:
            config_kwargs = ast.literal_eval("{" + kvs + "}")
        except Exception:
            raise ValueError(f"Invalid adapter configguration '{kvs}' in '{name}'.")
        adapter_configs.append((name, config_kwargs))

    return adapter_configs


def resolve_adapter_config(config: Union[dict, str], local_map=None, **kwargs) -> dict:
    """
    Resolves a given adapter configuration specifier to a full configuration dictionary.

    Args:
        config (Union[dict, str]): The configuration to resolve. Can be either:

            - a dictionary: returned without further action
            - an identifier string available in local_map
            - the path to a file containing a full adapter configuration

    Returns:
        dict: The resolved adapter configuration dictionary.
    """
    # already a dict, so we don't have to do anything
    if isinstance(config, Mapping):
        return config
    # first, look in local map
    if local_map and config in local_map:
        return local_map[config]
    if isfile(config):
        with open(config, "r") as f:
            loaded_config = json.load(f)
            if "config" in loaded_config:
                return loaded_config["config"]
            else:
                return loaded_config
    # parse the config string
    config_pairs = parse_adapter_config_string(config)
    if len(config_pairs) > 0:
        full_configs = []
        for name, config_kwargs in config_pairs:
            # first, look in local map
            if local_map and name in local_map:
                config_obj = local_map[name]
                full_configs.append(config_obj.replace(**config_kwargs))
            else:
                raise ValueError(
                    "Could not identify '{}' as a valid adapter configuration.".format(
                        name
                    )
                )
        if len(full_configs) == 1:
            return full_configs[0]
        elif len(full_configs) > 1:
            return {"architecture": "union", "configs": full_configs}

    raise ValueError(
        "Could not identify '{}' as a valid adapter configuration.".format(config)
    )


def _split_identifier(identifier):
    return None, None, None


def _dict_extract(*args, **kwargs):
    return []


def find_in_index(*args, **kwargs):
    return None


def _get_matching_version(config_entry, org):
    if org:
        return config_entry["versions"].get(org, None)
    elif len(config_entry["versions"]) == 1:
        return list(config_entry["versions"].values())[0]
    elif "default" in config_entry:
        return config_entry["default"]
    else:
        raise ValueError(
            "Multiple adapters with this name are available for this config."
        )


def resolve_adapter_path(
    adapter_name_or_path,
    model_name: str = None,
    adapter_config: Union[dict, str] = None,
    version: str = None,
    do_exists_check: bool = True,
    **kwargs,
) -> str:
    """
    Resolves the path to a pre-trained adapter module. Note: If attempting to resolve an adapter from the Hub,
    adapter_config and model_name must be present.

    Args:
        adapter_name_or_path (str): Can be either:

            - the path to a folder in the file system containing the adapter configuration and weights
            - an url pointing to a zip folder containing the adapter configuration and weights
            - a specifier matching a pre-trained adapter uploaded to Adapter-Hub
        model_name (str, optional): The identifier of the pre-trained model for which to load an adapter.
        adapter_config (Union[dict, str], optional): The configuration of the adapter to be loaded.
        version (str, optional): The version of the adapter to be loaded. Defaults to None.

    Returns:
        str: The local path from where the adapter module can be loaded.
    """
    if is_remote_url(adapter_name_or_path):
        raise EnvironmentError(
            "Remote adapter URLs are not supported; provide a local directory path."
        )
    # path to a local folder saved using save()
    elif isdir(adapter_name_or_path):
        if (
            not do_exists_check
            or (
                isfile(join(adapter_name_or_path, WEIGHTS_NAME))
                or isfile(join(adapter_name_or_path, SAFE_WEIGHTS_NAME))
            )
            and isfile(join(adapter_name_or_path, CONFIG_NAME))
        ):
            return adapter_name_or_path
        else:
            raise EnvironmentError(
                "No file {} or no file {} found in directory {}".format(
                    WEIGHTS_NAME, CONFIG_NAME, adapter_name_or_path
                )
            )


# ===== AdapterConfig class =====
class AdapterConfig(Mapping):
    """
    Base class for all adaptation methods. This class does not define specific configuration keys, but only provides
    some common helper methods.

    Args:
        architecture (str, optional): The type of adaptation method defined by the configuration.
    """

    architecture: Optional[str] = None

    def __init__(self):
        raise TypeError(
            "AdapterConfig is an abstract class and cannot be instantiated."
        )

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise FrozenInstanceError()
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise FrozenInstanceError()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()

    def to_dict(self):
        """Converts the config class to a Python dict."""
        return asdict(self)

    def replace(self, **changes):
        """Returns a new instance of the config class with the specified changes applied."""
        return replace(self, **changes)

    @classmethod
    def from_dict(cls, config):
        """Creates a config class from a Python dict."""
        if isinstance(config, AdapterConfig):
            return config

        # the constructor does not accept additional kwargs, so add them separately
        defined_kwargs, new_kwargs = {}, {}
        for k, v in config.items():
            if k in cls.__dataclass_fields__.keys():
                defined_kwargs[k] = v
            else:
                new_kwargs[k] = v
        obj = cls(**defined_kwargs)
        for k, v in new_kwargs.items():
            setattr(obj, k, v)
        return obj

    @classmethod
    def load(cls, config: Union[dict, str], download_kwargs=None, **kwargs):
        """
        Loads a given adapter configuration specifier into a full AdapterConfig instance.

        Args:
            config (Union[dict, str]): The configuration to load. Can be either:

                - a dictionary representing the full config
                - an identifier string available in ADAPTER_CONFIG_MAP
                - the path to a file containing a full adapter configuration
                - an identifier string available in Adapter-Hub

        Returns:
            dict: The resolved adapter configuration dictionary.
        """
        if not config:
            return None
        if download_kwargs and download_kwargs.get("force_download", False):
            local_map = None
        else:
            local_map = ADAPTER_CONFIG_MAP
        if download_kwargs:
            config_dict = resolve_adapter_config(
                config, local_map=local_map, **download_kwargs
            )
        else:
            config_dict = resolve_adapter_config(config, local_map=local_map)
        # convert back to dict to allow attr overrides
        if isinstance(config_dict, AdapterConfig):
            config_dict = config_dict.to_dict()
        if kwargs:
            config_dict.update((k, v) for k, v in kwargs.items() if v is not None)
        return config_dict


@dataclass(eq=False)
class BnConfig(AdapterConfig):
    """
    Base class that models the architecture of a bottleneck adapter.

    Args:
        mh_adapter (:obj:`bool`): If True, add adapter modules after the multi-head attention block of each layer.
        output_adapter (:obj:`bool`): If True, add adapter modules after the output FFN of each layer.
        reduction_factor (:obj:`float` or :obj:`Mapping`):
            Either a scalar float (> 0) specifying the reduction factor for all layers or a mapping from layer ID
            (starting at 0) to values specifying the reduction_factor for individual layers. If not all layers are
            represented in the mapping a default value should be given e.g. {'1': 8, '6': 32, 'default': 16}.
            Specifying a reduction factor < 1 will result in an up-projection layer.
        non_linearity (:obj:`str`): The activation function to use in the adapter bottleneck.
        original_ln_before (:obj:`bool`, optional):
            If True, apply layer pre-trained normalization and residual connection before the adapter modules. Defaults
            to False. Only applicable if :obj:`is_parallel` is False.
        original_ln_after (:obj:`bool`, optional):
            If True, apply pre-trained layer normalization and residual connection after the adapter modules. Defaults
            to True.
        ln_before (:obj:`bool`, optional): If True, add a new layer normalization before the adapter bottleneck.
            Defaults to False.
        ln_after (:obj:`bool`, optional): If True, add a new layer normalization after the adapter bottleneck.
            Defaults to False.
        init_weights (:obj:`str`, optional): Initialization method for the weights of the adapter modules.
            Currently, this can be either "bert" (default) or "mam_adapter" or "houlsby".
        init_weights_seed (:obj:`int`, optional): The seed to use for the initialization of the adapter weights per layer.
            Important:  set, the seed will be reset for all adapter modules, meaning that all adapter modules will have the same
            initialization. If not set, the seed will be set once and each adapter module has random weights initialization. Defaults to None.
        is_parallel (:obj:`bool`, optional): If True, apply adapter transformations in parallel.
            By default (False), sequential application is used.
        scaling (:obj:`float` or :obj:`str`, optional):
            Scaling factor to use for scaled addition of adapter outputs as done by He et al. (2021). Can be either a
            constant factor (float), or the string "learned", in which case the scaling factor is learned, or the string
            "channel", in which case we initialize a scaling vector of the channel shape that is then learned.
            Defaults to 1.0.
        use_gating (:obj:`bool`, optional):
            Place a trainable gating module besides the added parameter module to control module activation. This is
            e.g. used for UniPELT. Defaults to False.
        residual_before_ln (:obj:`bool` or :obj:`str`, optional):
            If True, take the residual connection around the adapter bottleneck before the layer normalization. If set
            to "post_add", take the residual connection around the adapter bottleneck after the previous residual
            connection. Only applicable if :obj:`original_ln_before` is True.
        adapter_residual_before_ln (:obj:`bool`, optional):
            If True, apply the residual connection around the adapter modules before the new layer normalization within
            the adapter. Only applicable if :obj:`ln_after` is True and :obj:`is_parallel` is False.
        inv_adapter (:obj:`str`, optional):
            If not None (default), add invertible adapter modules after the model embedding layer. Currently, this can
            be either "nice" or "glow".
        inv_adapter_reduction_factor (:obj:`float`, optional):
            The reduction to use within the invertible adapter modules. Only applicable if :obj:`inv_adapter` is not
            None.
        cross_adapter (:obj:`bool`, optional):
            If True, add adapter modules after the cross attention block of each decoder layer in an encoder-decoder
            model. Defaults to False.
        leave_out (:obj:`List[int]`, optional):
            The IDs of the layers (starting at 0) where NO adapter modules should be added.
        dropout (:obj:`float`, optional): The dropout rate used in the adapter layer. Defaults to 0.0.
        phm_layer (:obj:`bool`, optional): If True the down and up projection layers are a PHMLayer.
            Defaults to False
        phm_dim (:obj:`int`, optional): The dimension of the phm matrix.
            Only applicable if `phm_layer` is set to `True`. Defaults to 4.
        shared_phm_rule (:obj:`bool`, optional): Whether the phm matrix is shared across all layers.
            Defaults to True
        factorized_phm_rule (:obj:`bool`, optional):
            Whether the phm matrix is factorized into a left and right matrix. Defaults to False.
        learn_phm (:obj:`bool`, optional): Whether the phm matrix should be learned during training.
            Defaults to True
        factorized_phm_W (:
            obj:`bool`, optional): Whether the weights matrix is factorized into a left and right matrix. Defaults to
            True
        shared_W_phm (:obj:`bool`, optional): Whether the weights matrix is shared across all layers.
            Defaults to False.
        phm_c_init (:obj:`str`, optional): The initialization function for the weights of the phm matrix.
            The possible values are `["normal", "uniform"]`. Defaults to `normal`.
        phm_init_range (:obj:`float`, optional): std for initializing phm weights if `phm_c_init="normal"`.
            Defaults to 0.0001.
        hypercomplex_nonlinearity (:obj:`str`, optional):
            This specifies the distribution to draw the weights in the phm layer from. Defaults to `glorot-uniform`.
        phm_rank (:obj:`int`, optional):
            If the weight matrix is factorized this specifies the rank of the matrix. E.g. the left matrix of the down
            projection has the shape (phm_dim, _in_feats_per_axis, phm_rank) and the right matrix (phm_dim, phm_rank,
            _out_feats_per_axis). Defaults to 1
        phm_bias (:obj:`bool`, optional):
            If True the down and up projection PHMLayer has a bias term. If `phm_layer` is False this is ignored.
            Defaults to True
        stochastic_depth (:obj:`float`, optional):
            This value specifies the probability of the model dropping entire layers during
            training. This parameter should be only used for vision based tasks involving
            residual networks.
    """

    # Required options
    mh_adapter: bool
    output_adapter: bool

    reduction_factor: Union[float, Mapping]
    non_linearity: str

    original_ln_before: bool = False
    original_ln_after: bool = True
    ln_before: bool = False
    ln_after: bool = False
    init_weights: str = "bert"
    init_weights_seed: Optional[int] = None
    is_parallel: bool = False
    scaling: Union[float, str] = 1.0
    use_gating: bool = False
    residual_before_ln: Union[bool, str] = True
    adapter_residual_before_ln: bool = False
    inv_adapter: Optional[str] = None
    inv_adapter_reduction_factor: Optional[float] = None
    cross_adapter: bool = False
    leave_out: List[int] = field(default_factory=list)
    dropout: float = 0.0
    phm_layer: bool = False
    phm_dim: int = 4
    factorized_phm_W: Optional[bool] = True
    shared_W_phm: Optional[bool] = False
    shared_phm_rule: Optional[bool] = True
    factorized_phm_rule: Optional[bool] = False
    phm_c_init: Optional[str] = "normal"
    phm_init_range: Optional[float] = 0.0001
    learn_phm: Optional[bool] = True
    hypercomplex_nonlinearity: Optional[str] = "glorot-uniform"
    phm_rank: Optional[int] = 1
    phm_bias: Optional[bool] = True
    stochastic_depth: Optional[float] = 0.0

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise FrozenInstanceError()
        elif name == "invertible_adapter":
            # Now, we have two config keys directly in the adapter config.
            if value:
                object.__setattr__(self, "inv_adapter", value["block_type"])
                object.__setattr__(
                    self,
                    "inv_adapter_reduction_factor",
                    value["reduction_factor"],
                )
        else:
            object.__setattr__(self, name, value)


@dataclass(eq=False)
class SeqBnConfig(BnConfig):
    """
    The adapter architecture proposed by Pfeiffer et al. (2020). See https://arxiv.org/pdf/2005.00247.pdf.
    """

    original_ln_before: bool = True
    original_ln_after: bool = True
    residual_before_ln: Union[bool, str] = True
    adapter_residual_before_ln: bool = False
    ln_before: bool = False
    ln_after: bool = False
    mh_adapter: bool = False
    output_adapter: bool = True
    non_linearity: str = "relu"
    reduction_factor: Union[float, Mapping] = 16


@dataclass(eq=False)
class CompacterPlusPlusConfig(SeqBnConfig):
    """
    The Compacter++ architecture proposed by Mahabadi et al. (2021). See https://arxiv.org/pdf/2106.04647.pdf.
    """

    phm_layer: bool = True
    reduction_factor: Union[float, Mapping] = 32
    non_linearity: str = "gelu"


@dataclass(eq=False)
class SeqBnInvConfig(SeqBnConfig):
    """
    The adapter architecture proposed by Pfeiffer et al. (2020). See https://arxiv.org/pdf/2005.00247.pdf.
    """

    inv_adapter: Optional[str] = "nice"
    inv_adapter_reduction_factor: Optional[float] = 2


@dataclass(eq=False)
class DoubleSeqBnConfig(BnConfig):
    """
    The adapter architecture proposed by Houlsby et al. (2019). See https://arxiv.org/pdf/1902.00751.pdf.
    """

    original_ln_before: bool = False
    original_ln_after: bool = True
    residual_before_ln: Union[bool, str] = True
    adapter_residual_before_ln: bool = False
    ln_before: bool = False
    ln_after: bool = False
    mh_adapter: bool = True
    output_adapter: bool = True
    non_linearity: str = "swish"
    reduction_factor: Union[float, Mapping] = 16


@dataclass(eq=False)
class CompacterConfig(DoubleSeqBnConfig):
    """
    The Compacter architecture proposed by Mahabadi et al. (2021). See https://arxiv.org/pdf/2106.04647.pdf.
    """

    phm_layer: bool = True
    reduction_factor: Union[float, Mapping] = 32
    non_linearity: str = "gelu"


@dataclass(eq=False)
class DoubleSeqBnInvConfig(DoubleSeqBnConfig):
    """
    The adapter architecture proposed by Houlsby et. al. (2019). See https://arxiv.org/pdf/1902.00751.pdf.
    """

    inv_adapter: Optional[str] = "nice"
    inv_adapter_reduction_factor: Optional[float] = 2


@dataclass(eq=False)
class ParBnConfig(BnConfig):
    """
    The parallel adapter architecture proposed by He et al. (2021). See https://arxiv.org/pdf/2110.04366.pdf.
    """

    original_ln_before: bool = False
    original_ln_after: bool = True
    ln_before: bool = False
    ln_after: bool = False
    mh_adapter: bool = False
    output_adapter: bool = True
    non_linearity: str = "relu"
    reduction_factor: Union[float, Mapping] = 2

    init_weights: str = "mam_adapter"
    is_parallel: bool = True
    scaling: Union[float, str] = 4.0


@dataclass(eq=False)
class AdapterPlusConfig(BnConfig):
    """
    The AdapterPlus config architecture proposed by Jan-Martin O, Steitz and Stefan Roth. See https://arxiv.org/pdf/2406.06820

    Please note that some configurations of the adapters parameters `original_ln_after`, `original_ln_before`, and
    `residual_before_ln` may result in performance issues when training.

    In the general case:
        1) At least one of `original_ln_before` or `original_ln_after` should be set to True in order to ensure that the original residual
           connection from pre-training is preserved.
        2) If `original_ln_after` is set to `False`, `residual_before_ln` must also be set to `False` to ensure convergence during training.
    """

    original_ln_after: bool = False
    original_ln_before: bool = True
    residual_before_ln: bool = False
    stochastic_depth: float = 0.1
    init_weights: str = "houlsby"
    scaling: Union[float, str] = "channel"

    mh_adapter: bool = False
    output_adapter: bool = True
    reduction_factor: Union[float, Mapping] = 96
    non_linearity: str = "gelu"


# IMPORTANT: When adding a new config here, also add it to docs/overview.md!
ADAPTER_CONFIG_MAP = {
    # DEPRECATED STRINGS
    "pfeiffer": SeqBnConfig(),
    "houlsby": DoubleSeqBnConfig(),
    "parallel": ParBnConfig(),
    "scaled_parallel": ParBnConfig(scaling="learned"),
    "pfeiffer+inv": SeqBnInvConfig(),
    "houlsby+inv": DoubleSeqBnInvConfig(),
    # CURRENT STRINGS
    "seq_bn": SeqBnConfig(),
    "double_seq_bn": DoubleSeqBnConfig(),
    "par_bn": ParBnConfig(),
    "scaled_par_bn": ParBnConfig(scaling="learned"),
    "seq_bn_inv": SeqBnInvConfig(),
    "double_seq_bn_inv": DoubleSeqBnInvConfig(),
    "compacter++": CompacterPlusPlusConfig(),
    "compacter": CompacterConfig(),
}

DEFAULT_ADAPTER_CONFIG = "seq_bn"

# ===== build_full_config function =====
def build_full_config(adapter_config, model_config, save_id2label=False, **kwargs):
    config_dict = {
        "model_type": model_config.model_type,
        # some models such as encoder-decoder don't have a model-wide hidden size
        "hidden_size": getattr(model_config, "hidden_size", None),
    }
    config_dict.update(kwargs)
    if not hasattr(model_config, "prediction_heads") and save_id2label:
        config_dict["label2id"] = model_config.label2id
    if isinstance(adapter_config, AdapterConfig):
        config_dict["config"] = adapter_config.to_dict()
    else:
        config_dict["config"] = adapter_config
    return config_dict


class WeightsLoaderHelper:
    """
    A class providing helper methods for saving and loading module weights.
    """

    def __init__(
        self,
        model,
        weights_name,
        config_name,
        use_safetensors: bool = False,
        safe_weights_name: Optional[str] = None,
    ):
        self.model = model
        self.weights_name = weights_name
        self.config_name = config_name
        self.use_safetensors = use_safetensors
        if use_safetensors and not safetensors_available:
            raise ValueError(
                "Safetensors package not available. Please install via `pip install safetensors`."
            )
        self.safe_weights_name = safe_weights_name or weights_name

    def state_dict(self, filter_func):
        return {k: v for (k, v) in self.model.state_dict().items() if filter_func(k)}

    def rename_state_dict(self, state_dict, *rename_funcs):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            for rename_func in rename_funcs:
                new_k = rename_func(new_k)
            new_state_dict[new_k] = v
        return new_state_dict

    def save_weights_config(self, save_directory, config, meta_dict=None):
        if meta_dict:
            for k, v in meta_dict.items():
                if k not in config:
                    config[k] = v
        # save to file system
        output_config_file = join(save_directory, self.config_name)
        with open(output_config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
        logger.info("Configuration saved in {}".format(output_config_file))

    def save_weights(self, save_directory, filter_func):
        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(
                save_directory
            ), "Saving path should be a directory where the module weights can be saved."

        state_dict = self.state_dict(filter_func)
        # Save the adapter weights
        if self.use_safetensors:
            output_file = join(save_directory, self.safe_weights_name)
            save_file(state_dict, output_file)
        else:
            output_file = join(save_directory, self.weights_name)
            torch.save(state_dict, output_file)
        logger.info("Module weights saved in {}".format(output_file))

    def load_weights_config(self, save_directory):
        config_file = join(save_directory, self.config_name)
        logger.info("Loading module configuration from {}".format(config_file))
        # Load the config
        with open(config_file, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
        # For older versions translate the activation function to the new format
        if "version" not in loaded_config:
            if "config" in loaded_config and loaded_config["config"] is not None:
                if (
                    "non_linearity" in loaded_config["config"]
                    and loaded_config["config"]["non_linearity"] in ACTIVATION_RENAME
                ):
                    loaded_config["config"]["non_linearity"] = ACTIVATION_RENAME[
                        loaded_config["config"]["non_linearity"]
                    ]
        return loaded_config

    @staticmethod
    def _load_module_state_dict(module, state_dict, start_prefix=""):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(module, prefix=start_prefix)

        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    module.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        return missing_keys, unexpected_keys

    def load_weights(
        self,
        save_directory,
        filter_func,
        rename_func=None,
        loading_info=None,
        in_base_model=False,
    ):
        # Load the weights of the adapter
        try:
            if self.use_safetensors:
                weights_file = join(save_directory, self.safe_weights_name)
                if exists(weights_file):
                    state_dict = load_file(weights_file, device="cpu")
                else:
                    logger.info(
                        f"No safetensors file found in {save_directory}. Falling back to torch.load..."
                    )
                    weights_file = join(save_directory, self.weights_name)
                    state_dict = torch.load(
                        weights_file, map_location="cpu", weights_only=True
                    )
            else:
                weights_file = join(save_directory, self.weights_name)
                state_dict = torch.load(
                    weights_file, map_location="cpu", weights_only=True
                )
        except Exception:
            raise OSError("Unable to load weights from pytorch checkpoint file. ")
        logger.info("Loading module weights from {}".format(weights_file))

        return self.load_weights_from_state_dict(
            state_dict,
            filter_func,
            rename_func=rename_func,
            loading_info=loading_info,
            in_base_model=in_base_model,
        )

    def load_weights_from_state_dict(
        self,
        state_dict,
        filter_func,
        rename_func=None,
        loading_info=None,
        in_base_model=False,
        start_prefix="",
    ):
        if rename_func:
            if isinstance(rename_func, Sequence):
                state_dict = self.rename_state_dict(state_dict, *rename_func)
            else:
                state_dict = self.rename_state_dict(state_dict, rename_func)

        # Add the weights to the model
        model_to_load = self.model
        has_prefix_module = any(
            s.startswith(self.model.base_model_prefix) for s in state_dict.keys()
        )
        if (
            not start_prefix
            and not hasattr(self.model, self.model.base_model_prefix)
            and has_prefix_module
        ):
            start_prefix = self.model.base_model_prefix + "."
        if (
            in_base_model
            and hasattr(self.model, self.model.base_model_prefix)
            and not has_prefix_module
        ):
            model_to_load = self.model.base_model

        missing_keys, unexpected_keys = self._load_module_state_dict(
            model_to_load, state_dict, start_prefix=start_prefix
        )

        missing_keys = [k for k in missing_keys if filter_func(k)]

        if len(missing_keys) > 0:
            logger.info(
                "Some module weights could not be found in loaded weights file: {}".format(
                    ", ".join(missing_keys)
                )
            )
        if self.model._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [
                k
                for k in unexpected_keys
                if k not in self.model._keys_to_ignore_on_load_unexpected
            ]
        if len(unexpected_keys) > 0:
            logger.info(
                "Some weights of the state_dict could not be loaded into model: {}".format(
                    ", ".join(unexpected_keys)
                )
            )

        if isinstance(loading_info, dict):
            if "missing_keys" not in loading_info:
                loading_info["missing_keys"] = []
            if "unexpected_keys" not in loading_info:
                loading_info["unexpected_keys"] = []
            loading_info["missing_keys"].extend(missing_keys)
            loading_info["unexpected_keys"].extend(unexpected_keys)

        return missing_keys, unexpected_keys


class WeightsLoader(ABC):
    """
    An abstract class providing basic methods for saving and loading weights of a model. Extend this class to build
    custom module weight loaders.
    """

    def __init__(
        self,
        model,
        weights_name,
        config_name,
        use_safetensors: bool = False,
        safe_weights_name: Optional[str] = None,
    ):
        self.model = model
        self.weights_helper = WeightsLoaderHelper(
            model,
            weights_name,
            config_name,
            use_safetensors=use_safetensors,
            safe_weights_name=safe_weights_name,
        )

    @abstractmethod
    def filter_func(self, name: str) -> Callable[[str], bool]:
        """
        The callable returned by this method is used to extract the module weights to be saved or loaded based on their
        names.

        Args:
            name (str): An identifier of the weights to be saved.

        Returns:
            Callable[str, bool]: A function that takes the fully qualified name of a module parameter and returns a
            boolean value that specifies whether this parameter should be extracted.
        """
        pass

    @abstractmethod
    def rename_func(self, old_name: str, new_name: str) -> Callable[[str], str]:
        """
        The callable returned by this method is used to optionally rename the module weights after loading.

        Args:
            old_name (str): The string identifier of the weights as loaded from file.
            new_name (str): The new string identifier to which the weights should be renamed.

        Returns:
            Callable[str, str]: A function that takes the fully qualified name of a module parameter and returns a new
            fully qualified name.
        """
        pass

    def save(self, save_directory, name, **kwargs):
        """
        Saves the module config and weights into the given directory. Override this method for additional saving
        actions.

        Args:
            save_directory (str): The directory to save the weights in.
            name (str): An identifier of the weights to be saved. The details are specified by the implementor.
        """
        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(
                save_directory
            ), "Saving path should be a directory where weights and configuration can be saved."

        config_dict = build_full_config(
            None,
            self.model.config,
            model_name=self.model.model_name,
            name=name,
            model_class=self.model.__class__.__name__,
        )
        meta_dict = kwargs.pop("meta_dict", None)

        # Save the adapter configuration
        self.weights_helper.save_weights_config(
            save_directory, config_dict, meta_dict=meta_dict
        )

        # Save adapter weights
        filter_func = self.filter_func(name)
        self.weights_helper.save_weights(save_directory, filter_func)

    def load(
        self, save_directory, load_as=None, loading_info=None, **kwargs
    ) -> Tuple[str, str]:
        """
        Loads the module weights from the given directory. Override this method for additional loading actions. If
        adding the loaded weights to the model passed to the loader class requires adding additional modules, this
        method should also perform the architectural changes to the model.

        Args:
            save_directory (str): The directory from where to load the weights.
            load_as (str, optional): Load the weights with this name. Defaults to None.

        Returns:
            Tuple[str, str]: A tuple consisting of the local file system directory from which the weights where loaded
            and the name of the loaded weights.
        """
        if not exists(join(save_directory, self.weights_helper.weights_name)):
            raise ValueError(
                "Loading path should be a directory where the weights are saved."
            )

        # Load config
        config = self.weights_helper.load_weights_config(save_directory)

        # Load head weights
        filter_func = self.filter_func(config["name"])
        if load_as:
            rename_func = self.rename_func(config["name"], load_as)
        else:
            rename_func = None
        self.weights_helper.load_weights(
            save_directory,
            filter_func,
            rename_func=rename_func,
            loading_info=loading_info,
        )

        return save_directory, load_as or config["name"]


class AdapterLoader(WeightsLoader):
    """
    A class providing methods for saving and loading adapter modules from the Hub, the filesystem or a remote url.

    Model classes passed to this loader must implement the `ModelAdaptersMixin` class.
    """

    def __init__(self, model, adapter_type=None, use_safetensors: bool = False):
        super().__init__(
            model,
            WEIGHTS_NAME,
            CONFIG_NAME,
            use_safetensors=use_safetensors,
            safe_weights_name=SAFE_WEIGHTS_NAME,
        )
        self.adapter_type = adapter_type
        if adapter_type and not AdapterType.has(self.adapter_type):
            raise ValueError("Invalid adapter type {}".format(self.adapter_type))

    def filter_func(self, adapter_name):
        return (
            lambda x: "_adapters.{}.".format(adapter_name) in x
            or ".adapters.{}.".format(adapter_name) in x
            or ".prefix_tunings.{}.".format(adapter_name) in x
            or ".prefix_gates.{}.".format(adapter_name) in x
            or ".loras.{}.".format(adapter_name) in x
            or ".refts.{}.".format(adapter_name) in x
            or ".prompt_tunings.{}.".format(adapter_name) in x
            or ".shared_parameters.{}.".format(adapter_name) in x
        )

    # This dict maps the original weight names to the currently used equivalents.
    # Old adapters will be loaded and converted to the new format automatically.
    legacy_weights_mapping = {
        "attention_text_task_adapters": "adapters",
        "attention_text_lang_adapters": "adapters",
        "layer_text_task_adapters": "adapters",
        "layer_text_lang_adapters": "adapters",
        "invertible_lang_adapters": "invertible_adapters",
    }

    def _rename_legacy_weights(self, k):
        for old, new in self.legacy_weights_mapping.items():
            k = k.replace(old, new)
        return k

    def _fix_backward_compat(self, config):
        ADAPTER_PREFIX = "adapters."
        MIN_VERSION = Version("1.1.0")

        version = config.get("version", "")
        if (
            version.startswith(ADAPTER_PREFIX)
            and Version(version[len(ADAPTER_PREFIX) :]) < MIN_VERSION
        ):
            if (
                config["config"].get("architecture", None) == "lora"
                and config["config"]["r"] != config["config"]["alpha"]
            ):
                logger.warning(
                    "Loading a LoRA trained using a faulty scaling implementation of a previous library version. Editing the configuration to make sure the adapter works as trained."
                    "See https://github.com/adapter-hub/adapters/pull/770 for more."
                )
                config["config"]["alpha"] = config["config"]["r"]

    # As inv. adapters would be incorrectly used in the new implementation,
    # catch this case here when loading pretrained adapters.
    def _fix_legacy_config(self, adapter_name, missing_keys):
        if self.adapter_type == AdapterType.text_task:
            inv_adapter_keys = [
                x for x in missing_keys if f"invertible_adapters.{adapter_name}." in x
            ]
            if len(inv_adapter_keys) > 0:
                del self.model.base_model.invertible_adapters[adapter_name]
                missing_keys = [k for k in missing_keys if k not in inv_adapter_keys]
                adapter_config_name = self.model.adapters_config.adapters[adapter_name]
                if adapter_config_name in self.model.adapters_config.config_map:
                    adapter_config = self.model.adapters_config.config_map[
                        adapter_config_name
                    ]
                    self.model.adapters_config.config_map[
                        adapter_config_name
                    ] = adapter_config.replace(
                        inv_adapter=None, inv_adapter_reduction_factor=None
                    )
        return missing_keys

    def rename_func(self, old_name, new_name):
        return (
            lambda k: self._rename_legacy_weights(k)
            .replace("adapters.{}.".format(old_name), "adapters.{}.".format(new_name))
            .replace(
                ".prefix_tunings.{}.".format(old_name),
                ".prefix_tunings.{}.".format(new_name),
            )
            .replace(
                ".prefix_gates.{}.".format(old_name),
                ".prefix_gates.{}.".format(new_name),
            )
            .replace(".loras.{}.".format(old_name), ".loras.{}.".format(new_name))
            .replace(
                ".shared_parameters.{}.".format(old_name),
                ".shared_parameters.{}.".format(new_name),
            )
            .replace(".refts.{}.".format(old_name), ".refts.{}.".format(new_name))
        )

    def save_to_state_dict(self, name: str):
        """
        Extracts the weights of a given adapter from the model and returns them as a state dict.

        Args:
            name (str): The name of the adapter to be saved.

        Returns:
            Tuple[dict, dict]: A tuple consisting of the state dict containing the adapter weights and the adapter
            configuration.
        """
        if name not in self.model.adapters_config.adapters:
            raise ValueError(
                "No adapter of this type with the given name is part of this model."
            )

        adapter_config = self.model.adapters_config.get(name)

        config_dict = build_full_config(
            adapter_config,
            self.model.config,
            model_name=self.model.model_name,
            name=name,
            model_class=self.model.__class__.__name__,
        )

        # Save adapter weights
        filter_func = self.filter_func(config_dict["name"])
        state_dict = self.weights_helper.state_dict(filter_func)

        return state_dict, config_dict

    def save(self, save_directory, name, meta_dict=None):
        """
        Saves an adapter and its configuration file to a directory, so that it can be reloaded using the `load()`
        method.

        Args:
            save_directory (str): a path to a directory where the adapter will be saved
            task_name (str): the name of the adapter to be saved
        """
        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(
                save_directory
            ), "Saving path should be a directory where adapter and configuration can be saved."
        assert (
            name in self.model.adapters_config.adapters
        ), "No adapter of this type with the given name is part of this model."

        adapter_config = self.model.adapters_config.get(name)

        self.model.apply_to_adapter_layers(lambda _, layer: layer.pre_save_adapters())

        config_dict = build_full_config(
            adapter_config,
            self.model.config,
            model_name=self.model.model_name,
            name=name,
            model_class=self.model.__class__.__name__,
        )

        # Save the adapter configuration
        self.weights_helper.save_weights_config(
            save_directory, config_dict, meta_dict=meta_dict
        )

        # Save adapter weights
        filter_func = self.filter_func(config_dict["name"])
        self.weights_helper.save_weights(save_directory, filter_func)

    def load_from_state_dict(
        self, state_dict, name, load_as=None, loading_info=None, start_prefix=""
    ):
        """
        Loads the weights of a given adapter from a state dict into the model.

        Args:
            state_dict (dict): The state dict from which to load the adapter weights.
            name (str): The name of the adapter to be loaded.
            load_as (str, optional): Load the adapter using this name. By default, the name with which the adapter was
                saved will be used.
            loading_info (dict, optional):
                A dictionary to which loading information (missing and unexpected keys) will be added.
            start_prefix (str, optional): A custom prefix to be ignored in the given state dict.
        """
        new_adapter_name = load_as or name
        if new_adapter_name not in self.model.adapters_config.adapters:
            raise ValueError(
                "No adapter of this type with the given name is part of this model."
            )

        # Load adapter weights
        filter_func = self.filter_func(name)
        rename_func = self.rename_func(name, new_adapter_name)
        missing_keys, _ = self.weights_helper.load_weights_from_state_dict(
            state_dict,
            filter_func,
            rename_func=rename_func,
            loading_info=loading_info,
            in_base_model=True,
            start_prefix=start_prefix,
        )
        missing_keys = self._fix_legacy_config(new_adapter_name, missing_keys)
        if isinstance(loading_info, Mapping):
            loading_info["missing_keys"] = missing_keys

    def load(
        self,
        adapter_name_or_path,
        config=None,
        version=None,
        model_name=None,
        load_as=None,
        loading_info=None,
        leave_out=None,
        set_active=False,
        **kwargs,
    ):
        """
        Loads a pre-trained pytorch adapter module from the local file system or a remote location.

        Args:
            adapter_name_or_path (str): can be either:

                - the identifier of a pre-trained task adapter to be loaded from Adapter Hub
                - a path to a directory containing adapter weights saved using `model.saved_adapter()`
                - a URL pointing to a zip folder containing a saved adapter module
            config (str, optional): Deprecated.
            version (str, optional): The version of the adapter to be loaded.
            model_name (str, optional): Deprecated.
            load_as (str, optional): Load the adapter using this name. By default, the name with which the adapter was
             saved will be used.

        Returns:
            Tuple[str, str]: A tuple consisting of the local file system directory from which the weights where loaded
            and the name of the loaded weights.
        """
        # Warn about deprecated arguments
        if config is not None or model_name is not None:
            logger.warning(
                "The 'config' and 'model_name' arguments are specific to the now unsupported legacy Hub repo and will"
                " be removed."
                "Please switch to only providing the HF Model Hub identifier.",
            )
        requested_config = AdapterConfig.load(config) if config else None
        model_name = self.model.model_name or model_name
        resolved_folder = resolve_adapter_path(
            adapter_name_or_path,
            model_name,
            adapter_config=requested_config,
            version=version,
            **kwargs,
        )

        # Load config of adapter
        config = self.weights_helper.load_weights_config(resolved_folder)
        if self.adapter_type and "type" in config:
            assert (
                config["type"] == self.adapter_type
            ), "Loaded adapter has to be a {} adapter.".format(self.adapter_type)
        elif "type" in config:
            self.adapter_type = config["type"]
        # post-loading drop of layers
        if leave_out is not None:
            if (
                "leave_out" in config["config"]
                and config["config"]["leave_out"] is not None
            ):
                # The conversion to a set and then back to a list removes all duplicates
                leave_out = list(set(leave_out + config["config"]["leave_out"]))
            config["config"]["leave_out"] = leave_out
        # Fix issues
        self._fix_backward_compat(config)

        adapter_name = load_as or config["name"]
        # If the adapter is not part of the model, add it
        if adapter_name not in self.model.adapters_config.adapters:
            self.model.add_adapter(
                adapter_name, config=config["config"], set_active=set_active
            )
        else:
            logger.warning("Overwriting existing adapter '{}'.".format(adapter_name))

        # Load adapter weights
        filter_func = self.filter_func(adapter_name)
        rename_func = self.rename_func(config["name"], adapter_name)
        missing_keys, _ = self.weights_helper.load_weights(
            resolved_folder,
            filter_func,
            rename_func=rename_func,
            loading_info=loading_info,
            in_base_model=True,
        )
        missing_keys = self._fix_legacy_config(adapter_name, missing_keys)
        if isinstance(loading_info, Mapping):
            loading_info["missing_keys"] = missing_keys

        return resolved_folder, adapter_name


class AdapterCompositionBlock(Sequence):
    def __init__(self, *children):
        self.children = [parse_composition(b, None) for b in children]

    def __getitem__(self, key):
        return self.children[key]

    def __len__(self):
        return len(self.children)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, type(self)):
            return all([c1 == c2 for c1, c2 in zip(self.children, o.children)])
        else:
            return False

    def __repr__(self):
        child_repr = ", ".join(map(str, self.children))
        return f"{self.__class__.__name__}[{child_repr}]"

    def first(self):
        if not isinstance(self.children[0], AdapterCompositionBlock):
            return self.children[0]
        else:
            return self.children[0].first()

    def last(self):
        if not isinstance(self.children[-1], AdapterCompositionBlock):
            return self.children[-1]
        else:
            return self.children[-1].last()

    @property
    def parallel_channels(self):
        return max(
            [
                (b.parallel_channels if isinstance(b, AdapterCompositionBlock) else 1)
                for b in self.children
            ]
        )

    def flatten(self) -> Set[str]:
        return set(
            itertools.chain(
                *[[b] if isinstance(b, str) else b.flatten() for b in self.children]
            )
        )

    def _get_save_kwargs(self):
        return None

    def to_dict(self):
        save_dict = {
            "type": self.__class__.__name__,
            "children": [
                (
                    c.to_dict()
                    if isinstance(c, AdapterCompositionBlock)
                    else {"type": "single", "children": [c]}
                )
                for c in self.children
            ],
        }
        if kwargs := self._get_save_kwargs():
            save_dict["kwargs"] = kwargs
        return save_dict

    @classmethod
    def from_dict(cls, data):
        children = []
        for child in data["children"]:
            if child["type"] == "single":
                children.append(child["children"][0])
            else:
                children.append(cls.from_dict(child))
        return getattr(sys.modules[__name__], data["type"])(
            *children, **data.get("kwargs", {})
        )


class Parallel(AdapterCompositionBlock):
    def __init__(self, *parallel_adapters: List[str]):
        """
        Can be used to perform inference for multiple tasks (i.e., adapters) in parallel (for the same input).

        See AdapterDrop https://arxiv.org/abs/2010.11918
        """
        super().__init__(*parallel_adapters)

    @property
    def parallel_channels(self):
        return len(self.children)


class Stack(AdapterCompositionBlock):
    def __init__(self, *stack_layers: List[Union[AdapterCompositionBlock, str]]):
        super().__init__(*stack_layers)


class Fuse(AdapterCompositionBlock):
    def __init__(
        self,
        *fuse_stacks: List[Union[AdapterCompositionBlock, str]],
        name: Optional[str] = None,
    ):
        super().__init__(*fuse_stacks)
        self._name = name

    @property
    def name(self):
        if self._name:
            return self._name
        else:
            return ",".join(
                [c if isinstance(c, str) else c.last() for c in self.children]
            )


class Split(AdapterCompositionBlock):
    def __init__(
        self,
        *split_adapters: List[Union[AdapterCompositionBlock, str]],
        splits: Union[List[int], int],
    ):
        super().__init__(*split_adapters)
        self.splits = (
            splits if isinstance(splits, list) else [splits] * len(split_adapters)
        )

    def _get_save_kwargs(self):
        return {"splits": self.splits}


class BatchSplit(AdapterCompositionBlock):
    def __init__(
        self,
        *split_adapters: List[Union[AdapterCompositionBlock, str]],
        batch_sizes: Union[List[int], int],
    ):
        super().__init__(*split_adapters)
        self.batch_sizes = (
            batch_sizes
            if isinstance(batch_sizes, list)
            else [batch_sizes] * len(split_adapters)
        )

    def _get_save_kwargs(self):
        return {"batch_sizes": self.batch_sizes}


class MultiTask(AdapterCompositionBlock):
    def __init__(self, *children):
        super().__init__(*children)


class Average(AdapterCompositionBlock):
    def __init__(
        self,
        *average_adapters: List[Union[AdapterCompositionBlock, str]],
        weights: Optional[List[float]] = None,
        normalize_weights: bool = True,
    ):
        super().__init__(*average_adapters)
        if weights is not None:
            # normalize weights
            if normalize_weights:
                sum_weights = sum(weights) if weights else 1
                self.weights = [w / sum_weights for w in weights]
            else:
                self.weights = weights
        else:
            self.weights = [1 / len(average_adapters)] * len(average_adapters)

    def _get_save_kwargs(self):
        return {"weights": self.weights}


# Mapping each composition block type to the allowed nested types
ALLOWED_NESTINGS = {
    Stack: [str, Fuse, Split, Parallel, BatchSplit, Average, MultiTask],
    Fuse: [str, Stack],
    Split: [str, Split, Stack, BatchSplit, Average],
    Parallel: [str, Stack, BatchSplit, Average],
    MultiTask: [str, Stack, Average, Fuse],
    BatchSplit: [str, Stack, Split, BatchSplit, Average],
    Average: [str, Stack, Split, BatchSplit],
}

# Some composition blocks might not be supported by all models.
SUPPORTED_MODELS = {
    Parallel: [
        "bert",
        "xlm-roberta",
        "bert-generation",
    ],
}


def validate_composition(
    adapter_composition: AdapterCompositionBlock, level=0, model_type=None
):
    if level > 1 and not (
        isinstance(adapter_composition, Stack) or isinstance(adapter_composition, str)
    ):
        raise ValueError(
            f"Adapter setup is too deep. Cannot have {adapter_composition} at level {level}."
        )
    if isinstance(adapter_composition, AdapterCompositionBlock):
        block_type = type(adapter_composition)
        if model_type and block_type in SUPPORTED_MODELS:
            if model_type not in SUPPORTED_MODELS[block_type]:
                raise ValueError(
                    f"Models of type {model_type} don't support adapter composition using {block_type.__name__}."
                )
        for child in adapter_composition:
            if not type(child) in ALLOWED_NESTINGS[type(adapter_composition)]:
                raise ValueError(
                    f"Adapter setup is invalid. Cannot nest {child} in {adapter_composition}"
                )
            # recursively validate children
            validate_composition(child, level=level + 1)


def parse_composition(
    adapter_composition, level=0, model_type=None
) -> AdapterCompositionBlock:
    """
    Parses and validates a setup of adapters.

    Args:
        adapter_composition: The adapter setup to be parsed.
        level (int, optional): If set to none, disables validation. Defaults to 0.
    """
    if not adapter_composition:
        return None
    elif isinstance(adapter_composition, AdapterCompositionBlock):
        if level is not None:
            validate_composition(
                adapter_composition, level=level, model_type=model_type
            )
        return adapter_composition
    elif isinstance(adapter_composition, str):
        if level == 0:
            return Stack(adapter_composition)
        else:
            return adapter_composition
    elif isinstance(adapter_composition, Sequence):
        # Functionality of adapter-transformers v1.x
        warnings.warn(
            "Passing list objects for adapter activation is deprecated. Please use Stack or Fuse explicitly.",
            category=FutureWarning,
        )
        if level == 1:
            block_class = Fuse
        else:
            block_class = Stack
        level = level + 1 if level is not None else None
        return block_class(*[parse_composition(b, level) for b in adapter_composition])
    else:
        raise TypeError(adapter_composition)


def adjust_tensors_for_parallel(hidden_states, *tensors):
    """
    Replicates a given list of tensors based on the shape of the reference tensor (first argument).
    """
    outputs = []
    for tensor in tensors:
        if tensor is not None and hidden_states.shape[0] > tensor.shape[0]:
            repeats = [1] * len(tensor.shape)
            repeats[0] = hidden_states.shape[0] // tensor.shape[0]
            new_tensor = tensor.repeat(*repeats)
            outputs.append(new_tensor)
        else:
            outputs.append(tensor)
    return tuple(outputs)


def adjust_tensors_for_parallel_(hidden_states, *tensors):
    """
    In-place version of adjust_tensors_for_parallel().
    """
    for tensor in tensors:
        if tensor is not None and hidden_states.shape[0] > tensor.shape[0]:
            repeats = [1] * len(tensor.shape)
            repeats[0] = hidden_states.shape[0] // tensor.shape[0]
            new_tensor = tensor.repeat(*repeats)
            tensor.set_(new_tensor)


def match_attn_matrices_for_parallel(
    query, key, value
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Matches the shapes of query, key and value matrices for parallel composition.
    """
    max_bsz = max(query.shape[0], key.shape[0], value.shape[0])

    query = query.repeat(max_bsz // query.shape[0], *([1] * len(query.shape[1:])))
    key = key.repeat(max_bsz // key.shape[0], *([1] * len(key.shape[1:])))
    value = value.repeat(max_bsz // value.shape[0], *([1] * len(value.shape[1:])))

    return query, key, value


class AdapterSetup(ContextManager):
    """
    Represents an adapter setup of a model including active adapters and active heads. This class is intended to be
    used as a context manager using the ``with`` statement. The setup defined by the ``AdapterSetup`` context will
    override static adapter setups defined in a model (i.e. setups specified via ``active_adapters``).

    Example::

        with AdapterSetup(Stack("a", "b")):
            # will use the adapter stack "a" and "b" outputs = model(**inputs)

    Note that the context manager is thread-local, i.e. it can be used with different setups in a multi-threaded
    environment.
    """

    # thread-local storage that holds a stack of active contexts
    storage = threading.local()

    def __init__(self, adapter_setup, head_setup=None, ignore_empty: bool = False):
        self.adapter_setup = parse_composition(adapter_setup)
        if head_setup:
            self.head_setup = head_setup
        else:
            self.head_setup = parse_heads_from_composition(self.adapter_setup)
        self._empty = (
            ignore_empty and self.adapter_setup is None and self.head_setup is None
        )

    def __enter__(self):
        if not self._empty:
            AdapterSetup.get_contexts().append(self)
        return self

    def __exit__(self, type, value, traceback):
        if not self._empty:
            AdapterSetup.get_contexts().pop()

    @classmethod
    def get_contexts(cls):
        if not hasattr(cls.storage, "contexts"):
            cls.storage.contexts = []
        return cls.storage.contexts

    @classmethod
    def get_context(cls):
        try:
            return cls.get_contexts()[-1]
        except IndexError:
            return None

    @classmethod
    def get_context(cls):
        contexts = cls.get_contexts()
        if not contexts:
            return None
        return contexts[-1]

    @classmethod
    def get_context_head_setup(cls):
        context = cls.get_context()
        if context:
            return context.head_setup
        return None

    @classmethod
    def get_context_adapter_setup(cls):
        context = cls.get_context()
        if context:
            return context.adapter_setup
        return None


class ForwardContext(ContextManager):
    """
    Holds context information during a forward pass through a model. This class should be used via the
    ``ForwardContext.wrap()`` method.

    Note that the context is thread-local.
    """

    # thread-local storage that holds a stack of active contexts
    storage = threading.local()

    context_args = {
        "output_adapter_gating_scores",
        "output_adapter_fusion_attentions",
        "adapter_input_parallelized",
        "task_ids",
    }
    context_attributes = {
        "adapter_gating_scores",
        "adapter_fusion_attentions",
    }
    # Additional used attributes not exposed to the user
    # - prompt_tokens_length: length of the prompt tokens

    def __init__(self, model, *args, **kwargs):
        # If the model has a method ``forward_context()``, use it to create the context.
        for arg_name in self.context_args:
            setattr(self, arg_name, kwargs.pop(arg_name, None))
        if hasattr(model, "forward_context"):
            model.forward_context(self, *args, **kwargs)

    def __enter__(self):
        ForwardContext.get_contexts().append(self)
        return self

    def __exit__(self, type, value, traceback):
        ForwardContext.get_contexts().pop()

    def _call_forward(self, model, f, *args, **kwargs):
        """
        Calls the forward function of the model with the given arguments and keyword arguments.
        """
        kwargs = {k: v for k, v in kwargs.items() if k not in self.context_args}
        results = f(model, *args, **kwargs)

        # append output attributes
        if isinstance(results, tuple):
            for attr in self.context_attributes:
                if getattr(self, "output_" + attr, False):
                    results = results + (dict(getattr(self, attr)),)
        else:
            for attr in self.context_attributes:
                if getattr(self, "output_" + attr, False):
                    results[attr] = dict(getattr(self, attr))

        return results

    @classmethod
    def add_context_args_in_signature(cls, f):
        old_signature = inspect.signature(f)
        params = list(old_signature.parameters.values())
        param_types = [param.kind for param in params]
        i = min(
            [
                (
                    param_types.index(param_type)
                    if param_type in param_types
                    else float("inf")
                )
                for param_type in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ]
            + [len(params)]
        )
        for name in cls.context_args:
            new_param = inspect.Parameter(
                name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
            )
            if new_param not in params:
                params.insert(i, new_param)
        new_signature = old_signature.replace(parameters=params)
        return new_signature

    @classmethod
    def wrap_base(cls, f):
        """
        Decorator method that wraps a ``forward()`` function of a base model class.
        Unlike ``wrap()``, this method does not create a new context if the is an existing one.
        """

        @functools.wraps(f)
        def wrapper_func(self, *args, **kwargs):
            if (
                self.adapters_config is not None
                and ForwardContext.get_context() is None
            ):
                with cls(self, *args, **kwargs) as ctx:
                    results = ctx._call_forward(self, f, *args, **kwargs)
                return results
            else:
                return f(self, *args, **kwargs)

        return wrapper_func

    @classmethod
    def wrap(cls, f):
        """
        Decorator method that wraps a ``forward()`` function of a model class.
        """

        @functools.wraps(f)
        def wrapper_func(self, *args, **kwargs):
            if self.adapters_config is not None:
                with cls(self, *args, **kwargs) as ctx:
                    results = ctx._call_forward(self, f, *args, **kwargs)
                return results
            else:
                return f(self, *args, **kwargs)

        return wrapper_func

    @classmethod
    def get_contexts(cls):
        if not hasattr(cls.storage, "contexts"):
            cls.storage.contexts = []
        return cls.storage.contexts

    @classmethod
    def get_context(cls):
        contexts = cls.get_contexts()
        if not contexts:
            return None
        return contexts[-1]


class ModelAdaptersConfig(Collection):
    """This class manages the setup and configuration of adapter modules in a pre-trained model."""

    def __init__(self, **kwargs):
        adapters_list = kwargs.pop("adapters", {})
        adapters_list = dict(
            map(
                lambda t: (
                    t[0],
                    t[1][1] or t[1][0] if isinstance(t[1], tuple) else t[1],
                ),
                adapters_list.items(),
            )
        )
        self.adapters: Mapping[str, str] = adapters_list
        self.config_map = kwargs.pop("config_map", {})

        self.fusions: Mapping[str, str] = kwargs.pop("fusions", {})
        self.fusion_config_map = kwargs.pop("fusion_config_map", {})
        self.fusion_name_map = kwargs.pop("fusion_name_map", {})

        self.active_setup: Optional[AdapterCompositionBlock] = None
        self.skip_layers = None

        self._vera_init_shapes = {}

    def __contains__(self, item):
        return item in self.adapters.keys()

    def __iter__(self):
        return iter(self.adapters)

    def __len__(self):
        return len(self.adapters)

    def get(self, adapter_name: str) -> Optional[dict]:
        """
        Gets the config dictionary for a given adapter.

        Args:
            adapter_name (str): The name of the adapter.

        Returns:
            Mapping: The adapter configuration.
        """
        if adapter_name in self.adapters:
            config_name = self.adapters[adapter_name]
            if config_name in self.config_map:
                config = self.config_map.get(config_name, None)
            else:
                config = ADAPTER_CONFIG_MAP.get(config_name, None)
            if isinstance(config, str):
                config = ADAPTER_CONFIG_MAP[config]
        else:
            config = None
        return config

    def match(
        self,
        adapter_name: str,
        config_type: type,
        layer_idx: Optional[int] = None,
        location_key: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Tries to match the given criteria to an existing adapter. Return the adapter config if a match is found,
        otherwise None.
        """
        config = self.get(adapter_name)
        if config is None:
            return None
        elif not isinstance(config, AdapterConfig):
            config = AdapterConfig.load(config)

        if isinstance(config, config_type):
            leave_out = config.get("leave_out", [])
            if layer_idx is None or layer_idx not in leave_out:
                if location_key is None or config.get(location_key, False):
                    return config
        return None

    def add(self, adapter_name: str, config: Optional[Union[str, dict]] = None):
        """
        Adds a new adapter of the name to the model config.

        Args:
            adapter_name (str): The name of the adapter.
            config (Optional[Union[str, dict]], optional): The adapter config. Defaults to None.
        """
        if adapter_name in self.adapters:
            raise ValueError(
                f"An adapter with the name '{adapter_name}' has already been added."
            )
        if config is None:
            config = DEFAULT_ADAPTER_CONFIG
        if isinstance(config, str):
            if config not in ADAPTER_CONFIG_MAP and config not in self.config_map:
                raise ValueError(f"Invalid adapter config identifier '{config}'.")
            config_name = config
        elif isinstance(config, Mapping):
            config_name = get_adapter_config_hash(config)
            self.config_map[config_name] = AdapterConfig.load(config)
        else:
            raise ValueError("Invalid adapter config: {}".format(config))
        self.adapters[adapter_name] = config_name
        logger.info(f"Adding adapter '{adapter_name}'.")

    def get_fusion(
        self, fusion_name: Union[str, List[str]]
    ) -> Tuple[Optional[dict], Optional[list]]:
        """
        Gets the config dictionary for a given AdapterFusion.

        Args:
            fusion_name (Union[str, List[str]]): The name of the AdapterFusion or the adapters to fuse.

        Returns:
            Optional[dict]: The AdapterFusion configuration.
            Optional[list]: The names of the adapters to fuse.
        """
        if isinstance(fusion_name, list):
            fusion_name = ",".join(fusion_name)
        if fusion_name in self.fusions:
            config_name = self.fusions[fusion_name]
            if config_name in self.fusion_config_map:
                config = self.fusion_config_map.get(config_name, None)
            else:
                config = ADAPTERFUSION_CONFIG_MAP.get(config_name, None)

            if fusion_name in self.fusion_name_map:
                adapter_names = self.fusion_name_map[fusion_name]
            else:
                adapter_names = fusion_name.split(",")

            return config, adapter_names
        else:
            return None, None

    def add_fusion(
        self,
        adapter_names: List[str],
        config: Optional[Union[str, dict]] = None,
        fusion_name: Optional[str] = None,
    ):
        """
        Adds a new AdapterFusion.

        Args:
            adapter_names (List[str]): The names of the adapters to fuse.
            config (Optional[Union[str, dict]], optional): AdapterFusion config. Defaults to None.
            fusion_name (Optional[str], optional): The name of the AdapterFusion. If not specified, will default to comma-separated adapter names.
        """
        if fusion_name is None:
            fusion_name = ",".join(adapter_names)
        else:
            self.fusion_name_map[fusion_name] = adapter_names
        if fusion_name in self.fusions:
            raise ValueError(
                f"An AdapterFusion with the name '{fusion_name}' has already been added."
            )
        if config is None:
            config = DEFAULT_ADAPTERFUSION_CONFIG
        if isinstance(config, str):
            if (
                config not in ADAPTERFUSION_CONFIG_MAP
                and config not in self.fusion_config_map
            ):
                raise ValueError(f"Invalid AdapterFusion config identifier '{config}'.")
            config_name = config
        elif isinstance(config, Mapping):
            config_name = get_adapter_config_hash(config)
            self.fusion_config_map[config_name] = config
        else:
            raise ValueError("Invalid AdapterFusion config: {}".format(config))
        self.fusions[fusion_name] = config_name
        logger.info(f"Adding AdapterFusion '{fusion_name}'.")

    def common_config_value(self, adapter_names: list, attribute: str):
        """
        Checks whether all adapters in a list share the same config setting for a given attribute and returns the
        shared value.

        Args:
            adapter_names (list): The adapters to check.
            attribute (str): The config attribute to check.
        """
        common_value = None
        for i, name in enumerate(adapter_names):
            config = self.get(name)
            if not config:
                raise ValueError(
                    f"No adapter with name '{name}' found. Make sure that an adapter with this name is loaded."
                )
            config_value = config.get(attribute, None)
            if i > 0 and config_value != common_value:
                raise ValueError(
                    f"All given adapters must define the same value for config attribute {attribute}."
                )
            common_value = config_value
        return common_value

    def to_dict(self):
        output_dict = {}
        output_dict["adapters"] = copy.deepcopy(self.adapters)
        output_dict["config_map"] = {}
        for k, v in self.config_map.items():
            if isinstance(v, AdapterConfig):
                output_dict["config_map"][k] = v.to_dict()
            else:
                output_dict["config_map"][k] = copy.deepcopy(v)
        output_dict["fusions"] = copy.deepcopy(self.fusions)
        output_dict["fusion_config_map"] = {}
        for k, v in self.fusion_config_map.items():
            if isinstance(v, AdapterConfig):
                output_dict["fusion_config_map"][k] = v.to_dict()
            else:
                output_dict["fusion_config_map"][k] = copy.deepcopy(v)
        output_dict["fusion_name_map"] = copy.deepcopy(self.fusion_name_map)
        return output_dict

    def __eq__(self, other):
        return isinstance(other, ModelAdaptersConfig) and (
            self.__dict__ == other.__dict__
        )


class AdapterMethod:
    """
    Enum of all supported adapter method types.

    Attributes:
        bottleneck: Adapter methods using bottleneck layers.
        prefix_tuning: Adapters methods based on Prefix Tuning. Note that this is currently unsupported via AdapterModelInterface.
        lora: Adapter methods based on low-rank adaptation.
        prompt_tuning: Adapter methods based on Prompt Tuning.
        reft: Adapters methods based on Representation Fine-Tuning.
        invertible: Adapter methods using invertible modules.
    """

    bottleneck = "bottleneck"
    prefix_tuning = "prefix_tuning"
    lora = "lora"
    prompt_tuning = "prompt_tuning"
    reft = "reft"
    invertible = "invertible"

    @staticmethod
    def get_from_config(config) -> List[str]:
        """
        Get the adapter type from a given adapter config.

        Args:
            config: The adapter config.

        Returns:
            List[str]: The adapter type.
        """
        methods = []
        if getattr(config, "inv_adapter", False):
            methods.append(AdapterMethod.invertible)
        if config.architecture is None:
            methods.append(AdapterMethod.bottleneck)
        elif config.architecture == "union":
            for sub_config in config.configs:
                methods.extend(AdapterMethod.get_from_config(sub_config))
        else:
            methods.append(config.architecture)
        return methods


@dataclass
class AdapterModelInterface:
    """
    Defines the main interface for integrating adapter methods into a model class.
    This interface translates generic accessor names to model-specific attribute names.

    Args:
        adapter_methods (List[str]): List of adapter types that are supported by the model. Subset of this list: ["bottleneck", "lora", "reft", "prompt_tuning", "invertible"]
        model_embeddings (str): Name of the model's embedding layer.
        model_layers (str): Name of the model's layer list.
        layer_self_attn (str): Name of the self-attention layer in a transformer layer.
        layer_cross_attn (str): Name of the cross-attention layer in a transformer layer.
        attn_o_proj (str): Name of the output projection layer in an attention layer.
        layer_intermediate_proj (str): Name of the intermediate projection layer in a transformer layer.
        layer_output_proj (str): Name of the output projection layer in a transformer layer.

        # Either the following three attributes must be specified:
        attn_k_proj (Optional[str]): Name of the key projection layer in an attention layer.
        attn_q_proj (Optional[str]): Name of the query projection layer in an attention layer.
        attn_v_proj (Optional[str]): Name of the value projection layer in an attention layer.

        # Or this single attribute must be specified (but not both sets):
        attn_qkv_proj (Optional[str]): Name of the combined query-key-value projection layer (for models like GPT-2 or ModernBERT where QKV are in one tensor).

        layer_pre_self_attn (Optional[str]): Hook point directly before the self attention layer. Used for extended bottleneck adapter support.
        layer_pre_cross_attn (Optional[str]): Hook point directly before the cross attention layer. Used for extended bottleneck adapter support.
        layer_pre_ffn (Optional[str]): Hook point directly before the feed forward layer. Used for extended bottleneck adapter support.
        layer_ln_1 (Optional[str]): Layer norm *after* the self-attention layer. Used for extended bottleneck adapter support.
        layer_ln_2 (Optional[str]): Layer norm *after* the feed forward layer. Used for extended bottleneck adapter support.

        base_model (Optional[str]): Name of the base transformers model holding the layer modules. By default, this uses the model class' base_model_prefix attribute.

    Note:
        You must specify either all three of the individual projection layers (attn_k_proj, attn_q_proj, attn_v_proj) OR the combined projection layer (attn_qkv_proj).
    """

    adapter_methods: List[str]

    model_embeddings: str
    model_layers: str

    layer_self_attn: str
    layer_cross_attn: str

    attn_o_proj: Optional[str]

    layer_intermediate_proj: str
    layer_output_proj: str

    ###
    # Either all of these (this is the default and best working implementation):
    attn_k_proj: Optional[str] = None
    attn_q_proj: Optional[str] = None
    attn_v_proj: Optional[str] = None

    attn_qkv_proj: Optional[str] = None
    ###

    layer_pre_self_attn: Optional[str] = None
    layer_pre_cross_attn: Optional[str] = None
    layer_pre_ffn: Optional[str] = None
    layer_ln_1: Optional[str] = None
    layer_ln_2: Optional[str] = None

    base_model: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    def __post_init__(self):
        """Validate projection attributes after initialization."""

        has_separate_projections = (
            self.attn_k_proj is not None
            and self.attn_q_proj is not None
            and self.attn_v_proj is not None
        )
        has_combined_projection = self.attn_qkv_proj is not None

        if not has_separate_projections and not has_combined_projection:
            raise ValueError(
                "Must specify either individual projections (k,q,v) layers or combined qkv projection layer. You currently are neither specifying attn_qkv_proj nor attn_k_proj, attn_q_proj and attn_v_proj."
            )

        if has_separate_projections and has_combined_projection:
            raise ValueError(
                "Cannot specify both individual projections (k,q,v) and combined qkv projection. You specified attn_qkv_proj as well as attn_k_proj, attn_q_proj and attn_v_proj which makes no sense."
            )

    def _save(self, save_directory, model_config):
        config_dict = {
            "model_type": model_config.model_type,
            "interface": self.to_dict(),
        }
        save_path = os.path.join(save_directory, INTERFACE_CONFIG_NAME)
        with open(save_path, "w") as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)

    @classmethod
    def _load(cls, path_or_repo_id: str, **kwargs):
        resolved_file = cached_file(path_or_repo_id, INTERFACE_CONFIG_NAME, **kwargs)
        with open(resolved_file, "r") as f:
            config_dict = json.load(f)
        return AdapterModelInterface(**config_dict["interface"])


SPECIAL_MODEL_TYPE_TO_MODULE_NAME = {
    "clip_vision_model": "clip",
    "clip_text_model": "clip",
}

# Configuration helpers
CONFIG_CLASS_KEYS_MAPPING = {
    "xlm_roberta": {},
}
SUBMODEL_NAMES = {"encoder-decoder": ["encoder", "decoder"]}


def wrap_config(config: PretrainedConfig):
    """Makes required changes to a model config class to allow usage with adapters."""
    import copy

    type(config).attribute_map = copy.deepcopy(type(config).attribute_map)
    # Ensure missing keys are in class
    if config.model_type in CONFIG_CLASS_KEYS_MAPPING:
        for key, value in CONFIG_CLASS_KEYS_MAPPING[config.model_type].items():
            if key not in config.attribute_map:
                config.attribute_map[key] = value


def init_adapters_config(
    model: PreTrainedModel,
    model_config: PretrainedConfig,
    adapters_config: Optional[ModelAdaptersConfig] = None,
):
    """Initializes the adapters config object of the model to enable adapter support."""
    # Make sure config is wrapped
    model.config = model_config
    wrap_config(model.config)

    # Init ModelAdaptersConfig
    if adapters_config is not None:
        model.adapters_config = adapters_config
    elif not hasattr(model_config, "adapters"):
        model.adapters_config = ModelAdaptersConfig()
    elif model_config.adapters is not None and not isinstance(
        model_config.adapters, ModelAdaptersConfig
    ):
        model.adapters_config = ModelAdaptersConfig(**model_config.adapters)
    if hasattr(model, "base_model") and model.base_model is not model:
        model.base_model.adapters_config = model.adapters_config

    fusion_models = getattr(model_config, "adapter_fusion_models", [])
    fusion_config = getattr(model_config, "adapter_fusion", None)
    for fusion_adapter_names in fusion_models:
        model.adapters_config.add_fusion(fusion_adapter_names, config=fusion_config)


_INTERFACE_ERROR_TEMPLATE = "AdapterInterface: '{layer_name}' is set to '{layer_value}' but this value is not found in the {parent_name}. See https://docs.adapterhub.ml/plugin_interface.html for more information."


def get_module_name(model_type: str) -> str:
    if model_type in SPECIAL_MODEL_TYPE_TO_MODULE_NAME:
        return SPECIAL_MODEL_TYPE_TO_MODULE_NAME[model_type]
    return model_type_to_module_name(model_type)


def replace_with_adapter_class(module: nn.Module, modules_with_adapters) -> None:
    if module.__class__.__name__ in MODEL_MIXIN_MAPPING:
        # Create new wrapper model class
        model_class = type(
            module.__class__.__name__,
            (MODEL_MIXIN_MAPPING[module.__class__.__name__], module.__class__),
            {},
        )
        module.__class__ = model_class
    elif module.__class__.__module__.startswith(
        "transformers.models"
    ) or module.__class__.__module__.startswith("adapters.wrappers.model"):
        try:
            module_class = getattribute_from_module(
                modules_with_adapters,
                module.__class__.__name__ + "WithAdapters",
            )
            module.__class__ = module_class
        except ValueError:
            # Silently fail and keep original module class
            pass


def init(
    model: PreTrainedModel,
    adapters_config: Optional[ModelAdaptersConfig] = None,
    interface: Optional[AdapterModelInterface] = None,
) -> None:
    if isinstance(model, ModelAdaptersMixin):
        return model

    model_name = get_module_name(model.config.model_type)

    # If interface is None, have a look at our pre-supported interfaces
    #     interface = get_adapter_interface(model.config.model_type)

    if interface is not None:
        # Override the default base_model_prefix
        if base_model_prefix := interface.base_model:
            model.base_model_prefix = base_model_prefix
        base_model = model.base_model
        _validate_interface_values(base_model, interface)
        model_class_name = base_model.__class__.__name__
        model_class = type(
            model_class_name,
            (
                EmbeddingAdaptersMixin,
                ModelBaseAdaptersMixin,
                base_model.__class__,
            ),
            {},
        )
        base_model.__class__ = model_class
        base_model.adapter_interface = interface
        base_model.support_prompt_tuning = (
            False  # HACK: will be set to true if init_prompt_tuning() is called
        )
    else:
        if model_name == "xlm_roberta":

            class MockModule:
                XLMRobertaSelfAttentionWithAdapters = (
                    XLMRobertaSelfAttentionWithAdapters
                )
                XLMRobertaSdpaSelfAttentionWithAdapters = (
                    XLMRobertaSdpaSelfAttentionWithAdapters
                )
                XLMRobertaSelfOutputWithAdapters = XLMRobertaSelfOutputWithAdapters
                XLMRobertaOutputWithAdapters = XLMRobertaOutputWithAdapters

            modules_with_adapters = MockModule()
        else:
            raise ValueError(
                f"Model {model_name} not pre-supported by adapters. Please specify and pass `interface` explicitly."
            )
        submodules = list(model.modules())

        # Replace the base model class
        replace_with_adapter_class(submodules.pop(0), modules_with_adapters)

        if False:  # ModelUsingSubmodels not used by sentencizer
            # Otherwise, it would not be shared between the submodels.
            init_adapters_config(model, model.config, adapters_config)
            adapters_config = model.adapters_config
            model.init_submodels()
            submodules = []

        for module in submodules:
            replace_with_adapter_class(module, modules_with_adapters)

    if not isinstance(model, ModelAdaptersMixin):
        if hasattr(model, "base_model_prefix") and hasattr(
            model, model.base_model_prefix
        ):
            base_model = getattr(model, model.base_model_prefix)
            if isinstance(base_model, ModelAdaptersMixin):
                temp_signature = ForwardContext.add_context_args_in_signature(
                    model.forward.__func__
                )
                # Create new wrapper model class
                model_class_name = model.__class__.__name__
                model_class = type(
                    model_class_name,
                    (
                        EmbeddingAdaptersWrapperMixin,
                        ModelWithHeadsAdaptersMixin,
                        model.__class__,
                    ),
                    {},
                )
                model.__class__ = model_class
                model.forward.__func__.__signature__ = temp_signature

    # Finally, initialize adapters
    model.init_adapters(model.config, adapters_config)


def load_model(
    model_name_or_path: Optional[Union[str, os.PathLike]],
    model_class: Type[PreTrainedModel],
    interface: Optional[AdapterModelInterface] = None,
    *model_args: Any,
    **kwargs: Any,
) -> PreTrainedModel:
    """
    Loads a pretrained model with adapters from the given path or url.

    Parameters:
        model_name_or_path (`str` or `os.PathLike`, *optional*):
            Parameter identical to PreTrainedModel.from_pretrained
        model_class (`PreTrainedModel` or `AutoModel`):
            The model class to load (e.g. EncoderDecoderModel and EncoderDecoderAdapterModel both work)
        interface (`AdapterModelInterface`, *optional*):
            The custom adapter interface to use for the model, to be passed to the init() method.
            If not provided, init() will try to use one of the built-in model integrations.
        model_args (sequence of positional arguments, *optional*):
            All remaining positional arguments will be passed to the underlying model's `__init__` method.
        kwargs (remaining dictionary of keyword arguments, *optional*):
            Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
            `output_attentions=True`).
    Returns:
        `PreTrainedModel`: The model with adapters loaded from the given path or url.
    """

    old_init = model_class.__init__

    if interface is None:
        try:
            interface = AdapterModelInterface._load(model_name_or_path, **kwargs)
        except EnvironmentError:
            pass

    def new_init(self, config, *args, **kwargs):
        old_init(self, config, *args, **kwargs)
        init(self, interface=interface)

    # wrap model after it is initialized but before the weights are loaded
    new_model_class = type(model_class.__name__, (model_class,), {})
    new_model_class.__init__ = new_init
    model = new_model_class.from_pretrained(model_name_or_path, *model_args, **kwargs)

    return model


def _validate_interface_values(
    base_model: PreTrainedModel, interface: AdapterModelInterface
) -> None:
    """
    Validates that all values specified in the interface exist in the model.

    Args:
        base_model: The base model to validate against
        interface: The adapter interface to validate

    Raises:
        ValueError: If any specified path is not found in the model
    """

    if not multihasattr(base_model, interface.model_embeddings):
        raise ValueError(
            _INTERFACE_ERROR_TEMPLATE.format(
                layer_name="model_embeddings",
                layer_value=interface.model_embeddings,
                parent_name="base_model",
            )
        )
    layers = multigetattr(base_model, interface.model_layers)
    if not layers:
        raise ValueError(
            _INTERFACE_ERROR_TEMPLATE.format(
                layer_name="model_layers",
                layer_value=interface.model_layers,
                parent_name="base_model",
            )
        )

    if len(layers) == 0:
        raise ValueError(
            f"AdapterInterface: 'model_layers' is set to '{interface.model_layers}'. But accessing this value of the base_model returns an empty list. See https://docs.adapterhub.ml/plugin_interface.html for more information."
        )

    layer = layers[0]

    layer_attributes = [
        "layer_self_attn",
        "layer_cross_attn",
        "layer_intermediate_proj",
        "layer_output_proj",
        "layer_pre_self_attn",
        "layer_pre_cross_attn",
        "layer_pre_ffn",
        "layer_ln_1",
        "layer_ln_2",
    ]
    values_to_check = {
        name: getattr(interface, name)
        for name in layer_attributes
        if getattr(interface, name) is not None
    }

    for layer_name, layer_value in values_to_check.items():
        if not multihasattr(layer, layer_value):
            raise ValueError(
                _INTERFACE_ERROR_TEMPLATE.format(
                    layer_name=layer_name,
                    layer_value=layer_value,
                    parent_name="model layer",
                )
            )

    attention_attributes = ["attn_o_proj"]

    if getattr(interface, "attn_q_proj") is not None:
        attention_attributes += ["attn_q_proj", "attn_k_proj", "attn_v_proj"]
    else:
        attention_attributes += ["attn_qkv_proj"]

    if interface.layer_self_attn is not None:
        self_attn_module = multigetattr(layer, interface.layer_self_attn)
        for attn_name in attention_attributes:
            attn_value = getattr(interface, attn_name)
            if not multihasattr(self_attn_module, attn_value):
                raise ValueError(
                    _INTERFACE_ERROR_TEMPLATE.format(
                        layer_name=attn_name,
                        layer_value=attn_value,
                        parent_name="self-attention layer",
                    )
                )

    if interface.layer_cross_attn is not None:
        cross_attn_module = multigetattr(layer, interface.layer_cross_attn)
        for attn_name in attention_attributes:
            attn_value = getattr(interface, attn_name)
            if not multihasattr(cross_attn_module, attn_value):
                raise ValueError(
                    _INTERFACE_ERROR_TEMPLATE.format(
                        layer_name=attn_name,
                        layer_value=attn_value,
                        parent_name="cross-attention layer",
                    )
                )


class AdapterLayerBase(metaclass=ABCMeta):
    """
    Base class for all adaptation methods that require per-layer modules.

    Make sure the 'adapter_modules_name' attribute is overriden in derived classes.
    """

    adapter_modules_name = ""

    @property
    def adapter_modules(self) -> Collection:
        return getattr(self, self.adapter_modules_name)

    @property
    def layer_idx(self):
        return getattr(self, "_layer_idx", -1)

    @layer_idx.setter
    def layer_idx(self, layer_idx):
        idx = getattr(self, "_layer_idx", layer_idx)
        assert idx == layer_idx
        setattr(self, "_layer_idx", idx)

    def get_active_setup(self):
        if hasattr(self, "adapters_config"):
            # First check current context before falling back to defined setup
            context = AdapterSetup.get_context()
            if context is not None:
                adapter_setup = context.adapter_setup
            else:
                adapter_setup = self.adapters_config.active_setup
        else:
            adapter_setup = None
        skip_adapters = adapter_setup is None or (
            self.adapters_config.skip_layers is not None
            and self.layer_idx in self.adapters_config.skip_layers
        )
        if not skip_adapters and (
            len(set(self.adapter_modules.keys()) & adapter_setup.flatten()) > 0
        ):
            return adapter_setup
        else:
            return None

    def _store_gating_score(self, adapter_name, gating_score):
        context = ForwardContext.get_context()
        if context.output_adapter_gating_scores:
            gating_cache = context.adapter_gating_scores
            if self.layer_idx not in gating_cache[adapter_name]:
                gating_cache[adapter_name][self.layer_idx] = {}
            gating_score = gating_score.detach().squeeze().cpu().numpy()
            if len(gating_score.shape) == 0:
                gating_score = np.expand_dims(gating_score, axis=0)
            cache_score = gating_cache[adapter_name][self.layer_idx].get(
                self.location_key, None
            )
            if cache_score is not None:
                gating_cache[adapter_name][self.layer_idx][
                    self.location_key
                ] = np.column_stack((cache_score, gating_score))
            else:
                gating_cache[adapter_name][self.layer_idx][
                    self.location_key
                ] = gating_score

    def _store_fusion_attentions(self, fusion_name, attentions):
        context = ForwardContext.get_context()
        if context.output_adapter_fusion_attentions:
            attention_cache = context.adapter_fusion_attentions
            if self.layer_idx not in attention_cache[fusion_name]:
                attention_cache[fusion_name][self.layer_idx] = {}
            attention_cache[fusion_name][self.layer_idx][self.location_key] = attentions

    @abstractmethod
    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        """Adds a new adapter module to the layer.

        Args:
            adapter_name (str): The name of the new adapter to add.
            layer_idx (int):
                The index of the adapters layer (this should be set once by the first added adapter and the kept fix).

        Returns:
            bool: True if the adapter was added, False otherwise.
        """
        raise NotImplementedError()

    def average_adapter(
        self,
        adapter_name: str,
        input_adapters: Dict[str, float],
        combine_strategy,
        **kwargs,
    ) -> bool:
        """Averages a set of adapter modules into a new adapter module.

        Args:
            adapter_name (str): The name of the new (averaged) adapter module to add.
            input_adapters (Dict[str, float]): Dictionary of adapter names and their corresponding weights.
            combine_strategy (str): The strategy to combine the adapters. Available strategies depend on the used adapter method, see: https://docs.adapterhub.ml/adapter_composition.html#merging-adapters
            **kwargs: Additional arguments that are specific to the combine_strategy. E.g. svd_rank for LoRA.

        Returns:
            bool: True if the adapter was added, False otherwise.
        """
        # add new adapter
        if self.add_adapter(adapter_name, self.layer_idx):
            if combine_strategy != "linear":
                raise ValueError(
                    f"Combine strategy {combine_strategy} not supported for the chosen adapter methods."
                )

            # average weights linearly
            avg_state_dict = {}
            for name, weight in input_adapters.items():
                if name in self.adapter_modules:
                    module = self.adapter_modules[name]
                    for k, v in module.state_dict().items():
                        if k in avg_state_dict:
                            avg_state_dict[k] += weight * v
                        else:
                            avg_state_dict[k] = weight * v
                else:
                    self.delete_adapter(adapter_name)  # clean up before raising error
                    raise ValueError("Adapter {} not found.".format(name))

            # load averaged weights
            self.adapter_modules[adapter_name].load_state_dict(avg_state_dict)

            return True

        return False

    def delete_adapter(self, adapter_name: str):
        """Deletes an adapter module from the layer.

        Args:
            adapter_name (str): The name of the adapter to delete.
        """
        if adapter_name in self.adapter_modules:
            del self.adapter_modules[adapter_name]

    def share_parameters(
        self,
        name: str,
        adapter_names: List,
        reference_adapter_name: Optional[str],
    ):
        pass  # default implementation does nothing as multi task is not applicable to all methods

    def unshare_parameters(self, name: str):
        pass  # default implementation does nothing as multi task is not applicable to all methods

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # default implementation does nothing as fusion is not applicable to all methods

    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # default implementation does nothing as fusion is not applicable to all methods

    def enable_adapters(
        self,
        adapter_setup: AdapterCompositionBlock,
        unfreeze_adapters: bool,
        unfreeze_fusion: bool,
    ):
        """Enables/ disables a set of adapter modules within the layer.

        Args:
            adapter_setup (AdapterCompositionBlock): The adapter setup to enable/ disable.
            unfreeze_adapters (bool): Whether to unfreeze the adapters.
        """
        if unfreeze_adapters:
            for name in adapter_setup.flatten():
                if name in self.adapter_modules:
                    for param in self.adapter_modules[name].parameters():
                        param.requires_grad = True

    def freeze_adapter(self, adapter_name: str, freeze: bool = True):
        """Freezes/ unfreezes an adapter module.

        Args:
            adapter_name (str): The name of the adapter to freeze/ unfreeze.
            freeze (bool, optional): Whether to freeze the adapter. Defaults to True.
        """
        if adapter_name in self.adapter_modules:
            self.adapter_modules[adapter_name].train(not freeze)
            for param in self.adapter_modules[adapter_name].parameters():
                param.requires_grad = not freeze

    def get_adapter(self, adapter_name: str) -> nn.Module:
        """Returns the adapter module with the given name.

        Args:
            adapter_name (str): The name of the adapter module.
        """
        if adapter_name in self.adapter_modules:
            return self.adapter_modules[adapter_name]
        else:
            return None

    def pre_save_adapters(self):
        """Called before saving the adapters to disk."""
        pass


class ComposableAdapterLayerBase(AdapterLayerBase):
    """
    Base class for all adapter methods that support composition.

    Make sure the 'adapter_modules_name' and 'supported_compositions' attributes as well as all abstract methods are
    overriden in derived classes. 'allow_multi_parallelize' can be set to True to allow inputs to be parallelized
    independently multiple times. This is useful when there are multiple parallel input flows through an adapter layer
    (e.g. in LoRA).
    """

    supported_compositions = []
    allow_multi_parallelize = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_mapping()

    def _init_mapping(self):
        # Mapping between composition block types and names of composition functions
        self.composition_to_func_map = {
            Stack: "compose_stack",
            Fuse: "compose_fuse",
            Split: "compose_split",
            MultiTask: "compose_multi_task",
            BatchSplit: "compose_batch_split",
            Parallel: "compose_parallel",
            Average: "compose_average",
        }

    def _get_compose_func(self, composition_type: type) -> callable:
        """Retrieves the correct composition function based on the mapping in 'composition_to_func_map'."""
        return getattr(self, self.composition_to_func_map[composition_type])

    # START CUSTOMIZABLE METHODS #
    # The following methods should be implemented in derived classes.

    def _bsz(self, state: NamedTuple) -> int:
        """
        Returns the batch size of the given state.
        """
        return state[0].shape[0]

    def pre_block(
        self,
        adapter_setup: Union[AdapterCompositionBlock, str],
        state: NamedTuple,
    ) -> NamedTuple:
        """
        Optional state pre-processing method which is invoked before passing the state to the first child block of a
        composition. By default, this method does not contain any logic. E.g. used for bottleneck adapters to implement
        residuals and LNs.

        Args:
            adapter_setup (Union[AdapterCompositionBlock, str]): The current composition or single adapter.
            state (NamedTuple): The current state.

        Returns:
            NamedTuple: The pre-processed state.
        """
        return state

    @abstractmethod
    def vslice(self, state: NamedTuple, slice_obj: slice) -> NamedTuple:
        """Slices the given state along the batch size (vertical) dimension.
        This is e.g. used by the BatchSplit and Parallel composition blocks. IMPORTANT: Has to be implemented by all
        derived classes.

        Args:
            state (NamedTuple): The state to be sliced.
            slice_obj (slice): The slice object.

        Returns:
            NamedTuple: The sliced state.
        """
        raise NotImplementedError()

    @abstractmethod
    def pad_and_concat(self, states: List[NamedTuple]) -> NamedTuple:
        """Concatenates the given states along the batch size dimension.
        Pads the states before concatenation if necessary. This is e.g. used by the BatchSplit and Parallel composition
        blocks. IMPORTANT: Has to be implemented by all derived classes.

        Args:
            states (List[NamedTuple]): The states to be concatenated.

        Returns:
            NamedTuple: The concatenated state.
        """
        raise NotImplementedError()

    @abstractmethod
    def repeat(self, state: NamedTuple, channels: int) -> NamedTuple:
        """Repeats the given state along the batch size dimension for the given number of times.
        This is e.g. used by the Parallel composition block. IMPORTANT: Has to be implemented by all derived classes.

        Args:
            state (NamedTuple): The state to be repeated.
            channels (int): The number of times the state should be repeated.

        Returns:
            NamedTuple: The repeated state.
        """
        raise NotImplementedError()

    @abstractmethod
    def mean(self, states: List[NamedTuple], weights: torch.Tensor) -> NamedTuple:
        """Averages the given states along the batch size dimension by the given weights.
        This is e.g. used by the Average composition block. IMPORTANT: Has to be implemented by all derived classes.

        Args:
            states (List[NamedTuple]): The states to be averaged.
            weights (torch.Tensor): The averaging weights.

        Returns:
            NamedTuple: The averaged state.
        """
        raise NotImplementedError()

    @abstractmethod
    def compose_single(
        self, adapter_setup: str, state: NamedTuple, lvl: int = 0
    ) -> NamedTuple:
        """Forwards the given state through the given single adapter.

        Args:
            adapter_setup (str): The name of the adapter.
            state (NamedTuple): The state to be forwarded.
            lvl (int, optional): The composition depth. Defaults to 0.

        Returns:
            NamedTuple: The state after forwarding through the adapter.
        """
        raise NotImplementedError()

    # END CUSTOMIZABLE METHODS #

    def check_composition_valid(
        self,
        parent: AdapterCompositionBlock,
        child: AdapterCompositionBlock,
        lvl: int,
    ):
        """Checks whether the given composition is valid.

        Args:
            parent (AdapterCompositionBlock): The parent composition block.
            child (AdapterCompositionBlock): The child composition block.
            lvl (int): The composition depth.

        Raises:
            ValueError: If the composition is invalid.
        """
        if isinstance(parent, Stack) and lvl >= 1:
            raise ValueError(
                "Specified adapter setup is too deep. Cannot have {} at level {}".format(
                    child.__class__.__name__, lvl
                )
            )
        elif type(child) not in ALLOWED_NESTINGS[type(parent)]:
            raise ValueError(
                "Cannot nest {} inside {}. Only the following nestings are allowed: {}".format(
                    child.__class__.__name__,
                    parent.__class__.__name__,
                    ", ".join([t.__name__ for t in ALLOWED_NESTINGS[type(parent)]]),
                )
            )

    def compose_stack(
        self, adapter_setup: Stack, state: NamedTuple, lvl: int = 0
    ) -> NamedTuple:
        """
        For sequentially stacking multiple adapters.
        """
        for i, adapter_stack_layer in enumerate(adapter_setup):
            if isinstance(adapter_stack_layer, AdapterCompositionBlock):
                self.check_composition_valid(adapter_setup, adapter_stack_layer, lvl)
                composition_func = self._get_compose_func(type(adapter_stack_layer))
                state = composition_func(adapter_stack_layer, state, lvl=lvl + 1)
            elif adapter_stack_layer in self.adapter_modules:
                state = self.pre_block(adapter_stack_layer, state)
                state = self.compose_single(adapter_stack_layer, state, lvl=lvl + 1)
            else:
                pass

        return state

    def compose_fuse(self, adapter_setup: Fuse, state: NamedTuple, lvl: int = 0):
        """
        For fusing multiple adapters using adapter fusion. NOTE: This method has no default implementation.
        """
        if set(self.adapter_modules.keys()).isdisjoint(adapter_setup.flatten()):
            return state
        raise NotImplementedError()

    def compose_split(self, adapter_setup: Split, state: NamedTuple, lvl: int = 0):
        """
        For splitting to multiple adapters along the sequence length dimension. NOTE: This method has no default
        implementation.
        """
        if set(self.adapter_modules.keys()).isdisjoint(adapter_setup.flatten()):
            return state
        raise NotImplementedError()

    def compose_batch_split(
        self, adapter_setup: BatchSplit, state: NamedTuple, lvl: int = 0
    ):
        """
        For splitting to multiple adapters along the batch size dimension.
        """
        if sum(adapter_setup.batch_sizes) != self._bsz(state):
            raise IndexError(
                "The given batch has a size of {} which is not equal to the sum of batch_sizes {}".format(
                    self._bsz(state), adapter_setup.batch_sizes
                )
            )

        state = self.pre_block(adapter_setup, state)

        children_states = []
        for i, child in enumerate(adapter_setup):
            # compute ids of sequences that should be passed to the ith adapter
            batch_idx = (
                sum(adapter_setup.batch_sizes[:i]),
                sum(adapter_setup.batch_sizes[: i + 1]),
            )
            if isinstance(child, AdapterCompositionBlock):
                self.check_composition_valid(adapter_setup, child, lvl)
                composition_func = self._get_compose_func(type(child))
                child_state = composition_func(
                    child,
                    self.vslice(state, slice(*batch_idx)),
                    lvl=lvl + 1,
                )
                children_states.append(child_state)
            elif child in self.adapter_modules:
                child_state = self.compose_single(
                    child,
                    self.vslice(state, slice(*batch_idx)),
                    lvl=lvl + 1,
                )
                children_states.append(child_state)
            else:
                children_states.append(self.vslice(state, slice(*batch_idx)))

        # concatenate all outputs and return
        state = self.pad_and_concat(children_states)
        return state

    def compose_multi_task(
        self, adapter_setup: MultiTask, state: NamedTuple, lvl: int = 0
    ):
        """
        For splitting to multiple adapters along the task_ids.
        """
        state = self.pre_block(adapter_setup, state)

        context = ForwardContext.get_context()
        assert hasattr(context, "task_ids")
        task_ids = context.task_ids
        assert task_ids is not None
        if isinstance(task_ids, list) and isinstance(task_ids[0], str):
            children = adapter_setup.children
            task_ids = torch.tensor([children.index(task) for task in task_ids])
        ordering_idx = task_ids.argsort()
        batch_sizes = task_ids.bincount().tolist()
        inter_state = self.compose_batch_split(
            adapter_setup=BatchSplit(*adapter_setup.children, batch_sizes=batch_sizes),
            state=self.vslice(state, ordering_idx),
            lvl=lvl,
        )
        final_state = self.vslice(inter_state, ordering_idx.argsort())
        return final_state

    def compose_parallel(
        self, adapter_setup: Parallel, state: NamedTuple, lvl: int = 0
    ):
        """
        For parallel execution of the adapters on the same input. This means that the input is repeated N times before
        feeding it to the adapters (where N is the number of adapters).
        """

        context = ForwardContext.get_context()
        if not context.adapters_parallelized:
            orig_batch_size = self._bsz(state)
            state = self.repeat(state, adapter_setup.parallel_channels)
            context.adapters_parallelized = True
            context.original_batch_size = orig_batch_size
        else:
            bsz = self._bsz(state)
            # If the input was already parallelized, we can parallelize it again.
            if self.allow_multi_parallelize and bsz == getattr(
                context, "original_batch_size", -1
            ):
                state = self.repeat(state, adapter_setup.parallel_channels)
                orig_batch_size = bsz
            # The base model should handle replication of input.
            elif bsz % adapter_setup.parallel_channels != 0:
                raise ValueError(
                    "The total input batch size in a Parallel adapter block must be divisible by the number of"
                    " parallel channels."
                )
            else:
                orig_batch_size = bsz // adapter_setup.parallel_channels

        state = self.pre_block(adapter_setup, state)

        children_states = []
        for i, child in enumerate(adapter_setup):
            if isinstance(child, AdapterCompositionBlock):
                self.check_composition_valid(adapter_setup, child, lvl)
                composition_func = self._get_compose_func(type(child))
                child_state = composition_func(
                    child,
                    self.vslice(
                        state,
                        slice(i * orig_batch_size, (i + 1) * orig_batch_size),
                    ),
                    lvl=lvl + 1,
                )
                children_states.append(child_state)
            elif child in self.adapter_modules:
                child_state = self.compose_single(
                    child,
                    self.vslice(
                        state,
                        slice(i * orig_batch_size, (i + 1) * orig_batch_size),
                    ),
                    lvl=lvl + 1,
                )
                children_states.append(child_state)
            else:
                children_states.append(
                    self.vslice(
                        state,
                        slice(i * orig_batch_size, (i + 1) * orig_batch_size),
                    )
                )

        # concatenate all outputs and return
        state = self.pad_and_concat(children_states)
        return state

    def compose_average(self, adapter_setup: Average, state: NamedTuple, lvl: int = 0):
        """
        For averaging the output representations of multiple adapters.
        """

        state = self.pre_block(adapter_setup, state)

        children_states = []
        for i, child in enumerate(adapter_setup):
            if isinstance(child, AdapterCompositionBlock):
                self.check_composition_valid(adapter_setup, child, lvl)
                composition_func = self._get_compose_func(type(child))
                child_state = composition_func(child, state, lvl=lvl + 1)
                children_states.append(child_state)
            elif child in self.adapter_modules:
                child_state = self.compose_single(child, state, lvl=lvl + 1)
                children_states.append(child_state)
            else:
                pass

        weights = torch.tensor(adapter_setup.weights)[:, None, None, None].to(
            state[0].device
        )
        state = self.mean(children_states, weights)

        return state

    def compose(
        self,
        adapter_setup: Union[AdapterCompositionBlock, str],
        state: NamedTuple,
    ) -> NamedTuple:
        """The main composition forward method which recursively calls the composition blocks forward methods.
        This method should be called by the forward method of the derived class.

        Args:
            adapter_setup (Union[AdapterCompositionBlock, str]): The adapter setup to be used.
            state (NamedTuple): The current state.

        Returns:
            NamedTuple: The state after forwarding through the adapter setup.
        """
        if isinstance(adapter_setup, AdapterCompositionBlock):
            composition_func = self._get_compose_func(type(adapter_setup))
            state = composition_func(adapter_setup, state, lvl=0)
        elif adapter_setup in self.adapter_modules:
            state = self.compose_single(adapter_setup, state, lvl=0)
        else:
            raise ValueError(
                "Invalid adapter setup: {} is not a valid adapter name or composition block.".format(
                    adapter_setup.__class__.__name__
                )
            )

        return state


def multigetattr(o: object, name: str, default=None) -> Optional[object]:
    if not name:
        return default
    for n in name.split("."):
        if hasattr(o, n):
            o = getattr(o, n)
        else:
            return default
    return o


LAYER_HOOK_UNSUPPORTED = [
    ("original_ln_after", False),
]


class BottleneckState(NamedTuple):
    """
    Models the input and output states of a bottleneck adapter layer.

    Args:
        hidden_states (torch.Tensor): The layer input/ output hidden states.
        input_tensor (torch.Tensor): The Transformer sub-block residual connection inputs.
        adapter_residual (torch.Tensor): The adapter residual connection inputs.
        layer_norm (torch.nn.Module, optional): The Transformer layer norm module.
        bottleneck_up (torch.Tensor, optional):
            The up-projected bottleneck MLP output. This is only for Fuse compositions.
        last (str, optional): Name of the last adapter applied in the composition.
    """

    hidden_states: torch.Tensor
    input_tensor: torch.Tensor
    adapter_residual: torch.Tensor
    layer_norm: Optional[torch.nn.Module]
    bottleneck_up: Optional[torch.Tensor] = None
    last: Optional[str] = None


class BottleneckLayer(ComposableAdapterLayerBase, nn.Module):
    adapter_modules_name = "adapters"
    supported_compositions = [Stack, Fuse, Split, Parallel, BatchSplit, Average]

    def __init__(self, location_key: str, is_layer_hooked: bool = False):
        super().__init__()
        self.location_key = location_key
        self.is_layer_hooked = is_layer_hooked

    def init_adapters(self, model_config, adapters_config):
        self._init_mapping()
        self.model_config = model_config
        self.adapters_config = adapters_config
        self.adapters = nn.ModuleDict(dict())
        self.adapter_fusion_layer = nn.ModuleDict(dict())
        if not hasattr(self, "is_layer_hooked"):
            self.is_layer_hooked = False

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        self.layer_idx = layer_idx
        adapter_config = self.adapters_config.match(
            adapter_name,
            config_type=BnConfig,
            layer_idx=self.layer_idx,
            location_key=self.location_key,
        )
        if adapter_config is not None:
            reduction_factor = adapter_config["reduction_factor"]
            if isinstance(reduction_factor, Mapping):
                if str(self.layer_idx) in reduction_factor:
                    reduction_factor = reduction_factor[str(self.layer_idx)]
                elif "default" in reduction_factor:
                    reduction_factor = reduction_factor["default"]
                else:
                    raise KeyError(
                        "The given reduction factor mapping does not give a default value and does not specify each "
                        "reduction factor individually. You need to provide a default value like this: "
                        '{"1": 16, "default": 16}'
                    )

            if self.is_layer_hooked:
                for key, value in LAYER_HOOK_UNSUPPORTED:
                    if adapter_config.get(key, None) == value:
                        raise ValueError(
                            f"Unsupported configuration for bottleneck layer hooking mode: {key}={value}. "
                            "Please set this configuration to a supported value."
                        )

            if adapter_config.is_parallel:
                adapter_class = ParallelAdapter
            else:
                adapter_class = Adapter
            adapter = adapter_class(
                adapter_name=adapter_name,
                input_size=self.model_config.hidden_size,
                down_sample=int(self.model_config.hidden_size // reduction_factor),
                config=adapter_config,
            )
            # residual & LN are applied by model, so don't apply in adapters
            if self.is_layer_hooked:
                adapter.original_ln_after = False
            adapter.train(self.training)  # make sure training mode is consistent
            self.adapters[adapter_name] = adapter
            return True

        return False

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        """See BertModel.add_fusion_layer"""
        fusion_name = (
            ",".join(adapter_names)
            if isinstance(adapter_names, list)
            else adapter_names
        )
        fusion_config, adapter_names = self.adapters_config.get_fusion(fusion_name)
        if self.adapters_config.common_config_value(adapter_names, self.location_key):
            dropout_prob = fusion_config.dropout_prob or getattr(
                self.model_config, "attention_probs_dropout_prob", 0
            )
            fusion = BertFusion(
                fusion_config,
                self.model_config.hidden_size,
                dropout_prob,
            )
            fusion.train(self.training)  # make sure training mode is consistent
            self.adapter_fusion_layer[fusion_name] = fusion

    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        adapter_names = (
            adapter_names if isinstance(adapter_names, str) else ",".join(adapter_names)
        )
        if adapter_names in self.adapter_fusion_layer:
            del self.adapter_fusion_layer[adapter_names]

    def enable_adapters(
        self,
        adapter_setup: AdapterCompositionBlock,
        unfreeze_adapters: bool,
        unfreeze_fusion: bool,
    ):
        """
        Unfreezes a given list of adapters, the adapter fusion layer, or both

        Args:
            adapter_names: names of adapters to unfreeze (or names of adapters part of the fusion layer to unfreeze)
            unfreeze_adapters: whether the adapter weights should be activated
            unfreeze_fusion: whether the adapter fusion layer for the given adapters should be activated
        """
        if unfreeze_adapters:
            for adapter_name in adapter_setup.flatten():
                if adapter_name in self.adapters:
                    for param in self.adapters[adapter_name].parameters():
                        param.requires_grad = True
        if unfreeze_fusion:
            if isinstance(adapter_setup, Fuse):
                if adapter_setup.name in self.adapter_fusion_layer:
                    for param in self.adapter_fusion_layer[
                        adapter_setup.name
                    ].parameters():
                        param.requires_grad = True
            for sub_setup in adapter_setup:
                if isinstance(sub_setup, Fuse):
                    if sub_setup.name in self.adapter_fusion_layer:
                        for param in self.adapter_fusion_layer[
                            sub_setup.name
                        ].parameters():
                            param.requires_grad = True

    def get_adapter_fusion(self, adapter_names: Union[List, str]):
        adapter_names = (
            adapter_names if isinstance(adapter_names, str) else ",".join(adapter_names)
        )
        if adapter_names in self.adapter_fusion_layer:
            return self.adapter_fusion_layer[adapter_names]
        else:
            return None

    def pre_block(
        self,
        adapter_setup: Union[AdapterCompositionBlock, str],
        state: BottleneckState,
    ) -> BottleneckState:
        if isinstance(adapter_setup, AdapterCompositionBlock):
            adapter_name = adapter_setup.first()
        else:
            adapter_name = adapter_setup
        first_adapter = self.adapters[adapter_name]
        hidden_states, _, residual = first_adapter.pre_forward(
            state.hidden_states, state.input_tensor, state.layer_norm
        )

        return state._replace(hidden_states=hidden_states, adapter_residual=residual)

    def vslice(self, state: BottleneckState, slice_obj: slice) -> BottleneckState:
        return BottleneckState(
            state.hidden_states[slice_obj],
            state.input_tensor[slice_obj],
            state.adapter_residual[slice_obj],
            state.layer_norm,
            (
                state.bottleneck_up[slice_obj]
                if state.bottleneck_up is not None
                else None
            ),
            state.last,
        )

    def pad_and_concat(self, states: List[BottleneckState]) -> BottleneckState:
        return BottleneckState(
            torch.cat([state.hidden_states for state in states], dim=0),
            torch.cat([state.input_tensor for state in states], dim=0),
            torch.cat([state.adapter_residual for state in states], dim=0),
            states[0].layer_norm,
            (
                torch.cat([state.bottleneck_up for state in states], dim=0)
                if states[0].bottleneck_up is not None
                else None
            ),
            states[-1].last,
        )

    def repeat(self, state: BottleneckState, channels: int) -> BottleneckState:
        return BottleneckState(
            state.hidden_states.repeat(channels, 1, 1),
            state.input_tensor.repeat(channels, 1, 1),
            state.adapter_residual.repeat(channels, 1, 1),
            state.layer_norm,
            (
                state.bottleneck_up.repeat(channels, 1, 1)
                if state.bottleneck_up is not None
                else None
            ),
            state.last,
        )

    def mean(
        self, states: List[BottleneckState], weights: torch.Tensor
    ) -> BottleneckState:
        return BottleneckState(
            torch.mean(
                torch.stack([s.hidden_states for s in states], 0) * weights,
                dim=0,
            ),
            states[0].input_tensor,
            states[0].adapter_residual,
            states[0].layer_norm,
            states[0].bottleneck_up,
            states[-1].last,
        )

    def compose_single(
        self, adapter_setup: str, state: BottleneckState, lvl: int = 0
    ) -> BottleneckState:
        adapter_layer = self.adapters[adapter_setup]
        context = ForwardContext.get_context()
        output_gating = (
            context.output_adapter_gating_scores if context is not None else False
        )
        layer_output = adapter_layer(
            state.hidden_states,
            residual_input=state.adapter_residual,
            output_gating=output_gating,
        )
        hidden_states, up = layer_output[0], layer_output[2]
        if output_gating:
            self._store_gating_score(adapter_setup, layer_output[-1])

        return state._replace(
            hidden_states=hidden_states, bottleneck_up=up, last=adapter_setup
        )

    def compose_fuse(self, adapter_setup: Fuse, state: BottleneckState, lvl: int = 0):
        """
        Performs adapter fusion with the given adapters for the given input.
        """
        context = ForwardContext.get_context()

        # config of _last_ fused adapter is significant
        fusion_config, _ = self.adapters_config.get_fusion(adapter_setup.name)
        last = adapter_setup.last()
        last_adapter = self.adapters[last]
        hidden_states, query, residual = last_adapter.pre_forward(
            state.hidden_states,
            state.input_tensor,
            state.layer_norm,
            fusion_config=fusion_config,
        )
        state = state._replace(hidden_states=hidden_states, adapter_residual=residual)

        children_states = []
        for child in adapter_setup:
            if isinstance(child, AdapterCompositionBlock):
                self.check_composition_valid(adapter_setup, child, lvl)
                composition_func = self._get_compose_func(type(child))
                child_state = composition_func(child, state, lvl=lvl + 1)
                children_states.append(child_state)
            elif child in self.adapter_modules:
                child_state = self.compose_single(child, state, lvl=lvl + 1)
                children_states.append(child_state)
            else:
                pass

        if len(children_states) > 0:
            up_list = torch.stack([state.bottleneck_up for state in children_states])
            up_list = up_list.permute(1, 2, 0, 3)

            output_fusion_attns = (
                context.output_adapter_fusion_attentions
                if context is not None
                else False
            )
            fusion_output = self.adapter_fusion_layer[adapter_setup.name](
                query,
                up_list,
                up_list,
                state.adapter_residual,
                output_attentions=output_fusion_attns,
            )
            if output_fusion_attns:
                hidden_states = fusion_output[0]
                self._store_fusion_attentions(adapter_setup.name, fusion_output[-1])
            else:
                hidden_states = fusion_output

        return state._replace(hidden_states=hidden_states, last=last)

    def compose_split(self, adapter_setup: Split, state: BottleneckState, lvl: int = 0):
        """
        Splits the given input between the given adapters.
        """
        if sum(adapter_setup.splits) != state.hidden_states.shape[1]:
            raise IndexError(
                "The given input has sequence length {} which is not equal to the sum of splits {}".format(
                    state.hidden_states.shape[1], adapter_setup.splits
                )
            )

        state = self.pre_block(adapter_setup, state)

        children_states = []
        last = None
        for i, child in enumerate(adapter_setup):
            batch_idx = (
                sum(adapter_setup.splits[:i]),
                sum(adapter_setup.splits[: i + 1]),
            )
            child_state = BottleneckState(
                state.hidden_states[:, batch_idx[0] : batch_idx[1], :],
                state.input_tensor[:, batch_idx[0] : batch_idx[1], :],
                state.adapter_residual[:, batch_idx[0] : batch_idx[1], :],
                state.layer_norm,
                (
                    state.bottleneck_up[:, batch_idx[0] : batch_idx[1], :]
                    if state.bottleneck_up is not None
                    else None
                ),
            )
            if isinstance(child, AdapterCompositionBlock):
                self.check_composition_valid(adapter_setup, child, lvl)
                composition_func = self._get_compose_func(type(child))
                child_state = composition_func(child, child_state, lvl=lvl + 1)
                children_states.append(child_state)
                last = child_state.last or last
            elif child in self.adapter_modules:
                child_state = self.compose_single(child, child_state, lvl=lvl + 1)
                children_states.append(child_state)
                last = child_state.last or last
            else:
                pass

        hidden_states = torch.cat(
            [child.hidden_states for child in children_states], dim=1
        )
        return state._replace(hidden_states=hidden_states, last=last)

    def bottleneck_layer_forward(self, hidden_states, residual_input, layer_norm):
        """Forward pass through the adapter layer.
        NOTE: This method should only be called if the calling module directly inherits from BottleneckLayer.
        Otherwise, call the regular forward() method.

        Args:
            hidden_states (torch.Tensor): Input hidden states to the adapter layer.
            residual_input (torch.Tensor): Residual input to the adapter layer.
            layer_norm (torch.nn.Module): Transformer layer normalization module to be used by the adapter layer.

        Returns:
            torch.Tensor: Output hidden states of the adapter layer.
        """
        # Batch sizes might be different due to prefix tuning w. Parallel block
        if residual_input is not None:
            (residual_input,) = adjust_tensors_for_parallel(
                hidden_states, residual_input
            )
            # Replicate in both directions as residual might be larger (e.g. GPT-J)
            (hidden_states,) = adjust_tensors_for_parallel(
                residual_input, hidden_states
            )
        adapter_setup = self.get_active_setup()
        if adapter_setup is not None:
            input_hidden_states = hidden_states

            state = BottleneckState(
                hidden_states, residual_input, residual_input, layer_norm
            )
            state = self.compose(adapter_setup, state)
            hidden_states, residual_input, _, _, _, last = state

            last_adapter = self.adapters[last]
            hidden_states = last_adapter.post_forward(
                hidden_states, input_hidden_states, residual_input, layer_norm
            )

        elif layer_norm is not None and not self.is_layer_hooked:
            hidden_states = layer_norm(hidden_states + residual_input)
        elif residual_input is not None and not self.is_layer_hooked:
            hidden_states = hidden_states + residual_input

        return hidden_states

    def forward(self, hidden_states, residual_input, layer_norm):
        """Forward pass through the adapter layer.

        Args:
            hidden_states (torch.Tensor): Input hidden states to the adapter layer.
            residual_input (torch.Tensor): Residual input to the adapter layer.
            layer_norm (torch.nn.Module): Transformer layer normalization module to be used by the adapter layer.

        Returns:
            torch.Tensor: Output hidden states of the adapter layer.
        """
        return self.bottleneck_layer_forward(hidden_states, residual_input, layer_norm)


def hook_fn(adapter_layer, ln_get_fn, module, args, output):
    context = ForwardContext.get_context()
    residual_input = getattr(
        context, f"{adapter_layer.location_key}_residual_input", None
    )
    if ln_get_fn is not None:
        layer_norm = ln_get_fn()
    else:
        layer_norm = None
    # Call adapter layer
    if isinstance(output, torch.Tensor):
        return adapter_layer(output, residual_input, layer_norm)
    else:
        return (adapter_layer(output[0], residual_input, layer_norm),) + output[1:]


def _residual_hook_fn(location_key, module, args):
    context = ForwardContext.get_context()
    if context is not None:
        setattr(context, f"{location_key}_residual_input", args[0])


def init_bottleneck(model):
    model = model.base_model
    for _, layer in model.iter_layers():
        if self_attn := multigetattr(
            layer, model.adapter_interface.layer_self_attn, None
        ):
            if o_proj := multigetattr(
                self_attn, model.adapter_interface.attn_o_proj, None
            ):
                if not hasattr(layer, "attention_adapters"):
                    layer.attention_adapters = BottleneckLayer(
                        "mh_adapter", is_layer_hooked=True
                    )
                    ln_1_get_fn = lambda: multigetattr(
                        layer, model.adapter_interface.layer_ln_1, None
                    )
                    o_proj.register_forward_hook(
                        partial(hook_fn, layer.attention_adapters, ln_1_get_fn)
                    )
        if layer_output_proj := multigetattr(
            layer, model.adapter_interface.layer_output_proj, None
        ):
            if not hasattr(layer, "output_adapters"):
                layer.output_adapters = BottleneckLayer(
                    "output_adapter", is_layer_hooked=True
                )
                ln_2_get_fn = lambda: multigetattr(
                    layer, model.adapter_interface.layer_ln_2, None
                )
                layer_output_proj.register_forward_hook(
                    partial(hook_fn, layer.output_adapters, ln_2_get_fn)
                )
        if cross_attn := multigetattr(
            layer, model.adapter_interface.layer_cross_attn, None
        ):
            if not hasattr(cross_attn, "cross_attention_adapters"):
                layer.attention_adapters = BottleneckLayer(
                    "cross_adapter", is_layer_hooked=True
                )
                cross_attn.register_forward_hook(
                    partial(hook_fn, layer.attention_adapters, None)
                )

        if model.adapter_interface.layer_pre_self_attn is not None:
            if pre_self_attn := multigetattr(
                layer, model.adapter_interface.layer_pre_self_attn, None
            ):
                pre_self_attn.register_forward_pre_hook(
                    partial(_residual_hook_fn, "mh_adapter")
                )
        if model.adapter_interface.layer_pre_cross_attn is not None:
            if pre_cross_attn := multigetattr(
                layer, model.adapter_interface.layer_pre_cross_attn, None
            ):
                pre_cross_attn.register_forward_pre_hook(
                    partial(_residual_hook_fn, "cross_adapter")
                )
        if model.adapter_interface.layer_pre_ffn is not None:
            if pre_ffn := multigetattr(
                layer, model.adapter_interface.layer_pre_ffn, None
            ):
                pre_ffn.register_forward_pre_hook(
                    partial(_residual_hook_fn, "output_adapter")
                )


class DummyPrefixTuning(nn.Module):
    """Dummy prefix tuning that just passes through without modification."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        key_states,
        value_states,
        residual_input,
        attention_mask=None,
        invert_mask=True,
    ):
        return key_states, value_states, attention_mask


class BertSelfAttentionAdaptersMixin:
    """Adds adapters to the BertSelfAttention module."""

    def init_adapters(self, model_config, adapters_config):
        # Create dummy prefix_tuning to satisfy forward method expectations
        self.prefix_tuning = DummyPrefixTuning()
        patch_forward(self)


class BertSelfOutputAdaptersMixin(BottleneckLayer):
    """Adds adapters to the BertSelfOutput module."""

    def __init__(self):
        super().__init__("mh_adapter")

    def init_adapters(self, model_config, adapters_config):
        self.location_key = "mh_adapter"
        super().init_adapters(model_config, adapters_config)
        patch_forward(self)


class BertOutputAdaptersMixin(BottleneckLayer):
    """Adds adapters to the BertOutput module."""

    def __init__(self):
        super().__init__("output_adapter")

    def init_adapters(self, model_config, adapters_config):
        self.location_key = "output_adapter"
        super().init_adapters(model_config, adapters_config)
        patch_forward(self)


def inherit_doc(cls):
    for name, func in vars(cls).items():
        if isinstance(func, Callable) and not func.__doc__:
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, "__doc__", None):
                    func.__doc__ = parfunc.__doc__
                    break
    return cls


def patch_forward(module: torch.nn.Module):
    if hasattr(module, "_old_forward"):
        module._old_forward = module.__class__.forward.__get__(module, module.__class__)


if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module


class InvertibleAdaptersMixin:
    """Mixin for Transformer models adding invertible adapters."""

    def init_adapters(self, model_config, adapters_config, **kwargs):
        self.invertible_adapters = nn.ModuleDict(dict())

        init_adapters_config(self, model_config, adapters_config)

        if hasattr(super(), "init_adapters"):
            super().init_adapters(self.config, self.adapters_config, **kwargs)

    def add_invertible_adapter(self, adapter_name: str) -> bool:
        """
        Adds an invertible adapter module for the adapter with the given name. If the given adapter does not specify an
        invertible adapter config, this method does nothing.

        Args:
            adapter_name (str): The name of the adapter for which to add an invertible adapter module.
        """
        if adapter_name in self.invertible_adapters:
            raise ValueError(
                f"Model already contains an adapter module for '{adapter_name}'."
            )
        embedding_size = getattr(self.config, "embedding_size", self.config.hidden_size)
        adapter_config = self.adapters_config.match(
            adapter_name,
            config_type=BnConfig,
            location_key="inv_adapter",
        )
        if adapter_config and adapter_config["inv_adapter"]:
            if adapter_config["inv_adapter"] == "nice":
                inv_adap = NICECouplingBlock(
                    [[embedding_size]],
                    non_linearity=adapter_config["non_linearity"],
                    reduction_factor=adapter_config["inv_adapter_reduction_factor"],
                )
            elif adapter_config["inv_adapter"] == "glow":
                inv_adap = GLOWCouplingBlock(
                    [[embedding_size]],
                    non_linearity=adapter_config["non_linearity"],
                    reduction_factor=adapter_config["inv_adapter_reduction_factor"],
                )
            else:
                raise ValueError(
                    f"Invalid invertible adapter type '{adapter_config['inv_adapter']}'."
                )
            self.invertible_adapters[adapter_name] = inv_adap
            self.invertible_adapters[adapter_name].apply(Adapter.init_bert_weights)
            return True

        return False

    def _average_invertible_adapter(
        self,
        adapter_name: str,
        input_adapters: Dict[str, float],
        combine_strategy: str,
    ) -> bool:
        # add new adapter
        if self.add_invertible_adapter(adapter_name):
            if combine_strategy != "linear":
                raise ValueError(
                    f"Combine strategy {combine_strategy} not supported for invertible adapters. Only 'linear' is"
                    " supported."
                )

            # average weights
            avg_state_dict = {}
            for name, weight in input_adapters.items():
                module = self.invertible_adapters[name]
                if module is not None:
                    for k, v in module.state_dict().items():
                        if k in avg_state_dict:
                            avg_state_dict[k] += weight * v
                        else:
                            avg_state_dict[k] = weight * v
            # load averaged weights
            self.invertible_adapters[adapter_name].load_state_dict(avg_state_dict)
            return True

        return False

    def delete_invertible_adapter(self, adapter_name: str):
        if adapter_name in self.invertible_adapters:
            del self.invertible_adapters[adapter_name]

    def get_invertible_adapter(self):
        if (
            self.adapters_config.active_setup is not None
            and len(self.adapters_config.active_setup) > 0
        ):
            first_adapter = self.adapters_config.active_setup.first()
            if first_adapter in self.invertible_adapters:
                return self.invertible_adapters[first_adapter]
        return None

    def enable_invertible_adapters(self, adapter_names):
        for adapter_name in adapter_names:
            if adapter_name in self.invertible_adapters:
                for param in self.invertible_adapters[adapter_name].parameters():
                    param.requires_grad = True

    def invertible_adapters_forward(self, hidden_states, rev=False):
        adapter_setup = self._get_active_setup()
        if adapter_setup is not None and len(adapter_setup) > 0:
            first_adapter = adapter_setup.first()
            if first_adapter in self.invertible_adapters:
                hidden_states = self.invertible_adapters[first_adapter](
                    hidden_states, rev=rev
                )
        return hidden_states

    def _get_active_setup(self):
        if hasattr(self, "adapters_config"):
            # First check current context before falling back to defined setup
            context = AdapterSetup.get_context()
            if context is not None:
                adapter_setup = context.adapter_setup
            else:
                adapter_setup = self.adapters_config.active_setup
        else:
            adapter_setup = None
        if adapter_setup is not None and (len(adapter_setup.flatten()) > 0):
            return adapter_setup
        else:
            return None


class InvertibleAdaptersWrapperMixin:
    """
    Mixin for Transformer models supporting invertible adapters in a child module. When applying this mixin, set
    `invertible_adapters_base_name` to the name of the child module that includes `InvertibleAdaptersMixin`.
    """

    invertible_adapters_base_name = ""

    @property
    def invertible_adapters_base(self):
        return getattr(self, self.invertible_adapters_base_name, None)

    @property
    def invertible_adapters(self):
        if self.invertible_adapters_base is not None:
            return self.invertible_adapters_base.invertible_adapters
        return None

    def add_invertible_adapter(self, adapter_name: str) -> bool:
        """
        Adds an invertible adapter module for the adapter with the given name. If the given adapter does not specify an
        invertible adapter config, this method does nothing.

        Args:
            adapter_name (str): The name of the adapter for which to add an invertible adapter module.
        """
        if self.invertible_adapters_base is not None:
            return self.invertible_adapters_base.add_invertible_adapter(adapter_name)
        return False

    def _average_invertible_adapter(
        self,
        adapter_name: str,
        input_adapters: Dict[str, float],
        combine_strategy: str,
    ) -> bool:
        if self.invertible_adapters_base is not None:
            return self.invertible_adapters_base._average_invertible_adapter(
                adapter_name, input_adapters, combine_strategy
            )
        return False

    def delete_invertible_adapter(self, adapter_name: str):
        if self.invertible_adapters_base is not None:
            self.invertible_adapters_base.delete_invertible_adapter(adapter_name)

    def get_invertible_adapter(self):
        if self.invertible_adapters_base is not None:
            return self.invertible_adapters_base.get_invertible_adapter()
        return None

    def enable_invertible_adapters(self, adapter_names):
        if self.invertible_adapters_base is not None:
            self.invertible_adapters_base.enable_invertible_adapters(adapter_names)

    def invertible_adapters_forward(self, hidden_states, rev=False):
        if self.invertible_adapters_base is not None:
            return self.invertible_adapters_base.invertible_adapters_forward(
                hidden_states, rev=rev
            )
        return hidden_states


class EmbeddingAdaptersMixin:
    """Mixin for Transformer models adding support for dynamically switching embeddings."""

    def init_adapters(self, model_config, adapters_config, **kwargs):
        self.loaded_embeddings = {}
        self._active_embedding = "default"

        init_adapters_config(self, model_config, adapters_config)

        super().init_adapters(self.config, self.adapters_config, **kwargs)

    def load_embeddings(self, path: str, name: str):
        """
        Load a saved embedding from the given path. If the embedding was saved with a tokenizer it is returned

        Args:
            path: the path to the saved embedding
            name: the name the embedding should be loaded as

        Returns: a tokenizer if it ws saved with the embedding otherwise None

        """
        from transformers.models.auto.tokenization_auto import AutoTokenizer

        if name in self.loaded_embeddings:
            raise ValueError(
                "An embedding with the name {} already exists".format(name)
            )
        tokenizer = None
        tokenizer_path = os.path.join(path, TOKENIZER_PATH)
        if os.path.isdir(tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        embedding_path = os.path.join(path, EMBEDDING_FILE)
        if not os.path.isfile(embedding_path):
            raise FileNotFoundError("No embeddings found at {}".format(embedding_path))
        weights = torch.load(embedding_path, weights_only=True)

        self.loaded_embeddings[name] = nn.Embedding.from_pretrained(weights)
        self.set_active_embeddings(name)
        return tokenizer

    def add_embeddings(
        self,
        name,
        tokenizer,
        reference_embedding=None,
        reference_tokenizer=None,
        embedding_dim=None,
    ):
        """
        Add a new embedding to the model. If a reference embedding and reference tokenizer are provided tokens in the
        present in both tokenizers are initialized to the embedding in the reference_embedding.

        Args:
            name: the name of the embedding
            tokenizer: the tokenizer determining the vocab of the embedding
            reference_embedding:
                the reference embedding to use for initializing the embeddings of tokens present in the newly created
                embedding
            reference_tokenizer: the tokenizer providing the vocab for the reference embedding
            embedding_dim:
                the dimension of the embeddings (if None the embedding_size, or if this doesn't exist the hidden_size,
                from the config is used)
        """
        if name in self.loaded_embeddings:
            raise ValueError(
                "An embedding with the name {} already exists".format(name)
            )
        if embedding_dim is not None:
            embedding_size = embedding_dim
        else:
            embedding_size = getattr(
                self.config, "embedding_size", self.config.hidden_size
            )
        embedding = nn.Embedding(len(tokenizer), embedding_size)
        # Use same initialization as base Transformer model
        embedding.weight.data.normal_(mean=0.0, std=0.02)
        if embedding.padding_idx is not None:
            embedding.weight.data[embedding.padding_idx].zero_()
        embedding.requires_grad_(False)
        if (reference_embedding is not None and reference_tokenizer is None) or (
            reference_tokenizer is not None and reference_embedding is None
        ):
            raise KeyError(
                "Reference embedding and reference tokenizer are required to use initialize embeddings from reference"
                " embedding"
            )
        if reference_embedding is not None and reference_tokenizer is not None:
            tokens = set(tokenizer.get_vocab().keys()) & set(
                reference_tokenizer.get_vocab().keys()
            )
            reference_vocab = reference_tokenizer.get_vocab()
            vocab = tokenizer.get_vocab()
            for t in tokens:
                idx_reference = reference_vocab[t]
                idx = vocab[t]
                embedding.weight[idx] = (
                    self.loaded_embeddings[reference_embedding]
                    .weight[idx_reference]
                    .detach()
                    .clone()
                )
        embedding.train(False)
        self.loaded_embeddings[name] = embedding
        self.set_active_embeddings(name)

    def delete_embeddings(self, name):
        """
        Deletes the embedding with the given name

        Args:
            name: The name of the embedding that should be deleted

        """
        if name not in self.loaded_embeddings:
            raise ValueError("No embedding with name {}".format(name))
        if self.active_embeddings == name:
            logger.warning(
                "The active embedding is deleted. Setting the default embedding as active."
            )
            self.set_active_embeddings("default")
        del self.loaded_embeddings[name]

    def save_embeddings(self, path, name, tokenizer=None):
        """
        Saves the embedding with the given name. If a tokenizer is passed as well the tokenizer is saved together with
        the embedding.

        Args:
            path: The path where the embedding should be saved
            name: The name of the embedding that should be saved
            tokenizer: optionally a tokenizer to save with the embedding (default is None)

        """
        if self.active_embeddings == name:
            self.loaded_embeddings[name] = self.get_input_embeddings()
        os.makedirs(path, exist_ok=True)
        embedding_path = os.path.join(path, EMBEDDING_FILE)
        torch.save(self.loaded_embeddings[name].weight, embedding_path)
        if tokenizer:
            tokenizer_path = os.path.join(path, TOKENIZER_PATH)
            tokenizer.save_pretrained(tokenizer_path)

    def set_active_embeddings(self, name):
        """
        Sets the active embedding for the forward pass of the model

        Args:
            name: The name of the embedding that should be used

        """
        self.loaded_embeddings[self.active_embeddings] = self.get_input_embeddings()
        self.set_input_embeddings(self.loaded_embeddings[name])
        self.config.vocab_size = self.loaded_embeddings[name].num_embeddings
        self._active_embedding = name

    @property
    def active_embeddings(self):
        return self._active_embedding


class EmbeddingAdaptersWrapperMixin:
    def load_embeddings(self, path: str, name: str):
        return self.base_model.load_embeddings(path, name)

    def add_embeddings(
        self,
        name,
        tokenizer,
        reference_embedding=None,
        reference_tokenizer=None,
    ):
        return self.base_model.add_embeddings(
            name, tokenizer, reference_embedding, reference_tokenizer
        )

    def delete_embeddings(self, name):
        return self.base_model.delete_embeddings(name)

    def save_embeddings(self, path, name, tokenizer=None):
        return self.base_model.save_embeddings(path, name, tokenizer)

    def set_active_embeddings(self, name):
        return self.base_model.set_active_embeddings(name)

    @property
    def active_embeddings(self):
        return self.base_model.active_embeddings

    @property
    def loaded_embeddings(self):
        return self.base_model.loaded_embeddings


class ModelAdaptersMixin(ABC):
    """Mixin for transformer models adding support for loading/ saving adapters."""

    add_base_adapters = False
    support_lora_delta_w_svd = True  # If True, the model supports the "lora_delta_w_svd" combine_strategy to merge adapter weights.
    support_prompt_tuning = True  # If False, the prompt tuning layer is not added to the model. If True, the prompt tuning layer is added if add_base_adapters is True.

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    def _link_prefix_to_pool(self, layer):
        if isinstance(layer, PrefixTuningLayer):
            layer.set_pool(self.base_model.prefix_tuning)

    def _add_tied_weights_keys(self):
        """Internal method to add adapter-specific keys to the list of tied weights keys."""
        if self.base_model.support_prompt_tuning:
            prompt_tied_weights_keys = ["prompt_tuning.base_model_embeddings.*"]
            if self._tied_weights_keys is not None:
                self._tied_weights_keys += prompt_tied_weights_keys
            else:
                self._tied_weights_keys = prompt_tied_weights_keys

    @property
    def model_name(self):
        return self.config.name_or_path

    def _init_adapters_submodules(self, model_config, adapters_config):
        # Initialize adapters in all submodules
        for module in self.modules():
            # skip calling module
            if module == self:
                continue
            if hasattr(module, "init_adapters"):
                module.init_adapters(model_config, adapters_config)

    def _default_init_adapter_methods(self, model_config, adapters_config):
        pass

    def init_adapters(self, model_config, adapters_config):
        """
        This method initializes adapter modules and fusion modules from the model config.
        """
        self.base_model.shared_parameters = nn.ModuleDict()

        # Initialize adapters config
        init_adapters_config(self, model_config, adapters_config)

        # Initialize adapter types defined in interface
        if getattr(self.base_model, "adapter_interface", None) is not None:
            for adapter_type in self.base_model.adapter_interface.adapter_methods:
                init_func = METHOD_INIT_MAPPING[adapter_type]
                init_func(self)
        else:
            self._default_init_adapter_methods(self.config, self.adapters_config)

        # Initialize adapters in all submodules
        self._init_adapters_submodules(self.config, self.adapters_config)

        # Link all prefix tunings
        if hasattr(self.base_model, "prefix_tuning"):
            self.apply_to_adapter_layers(
                lambda i, layer: self._link_prefix_to_pool(layer)
            )

        for adapter_name in self.adapters_config:
            self._add_adapter_weights(adapter_name)
        for fusion_name in self.adapters_config.fusions:
            self.apply_to_adapter_layers(
                lambda i, layer: layer.add_fusion_layer(fusion_name)
            )

        if isinstance(self, EmbeddingAdaptersMixin):
            self.loaded_embeddings["default"] = self.get_input_embeddings()

        self._add_tied_weights_keys()

    def supports_adapter(self, type_or_config: Union[str, AdapterConfig]) -> bool:
        """
        Checks if the model supports a given adapter type.

        Args:
            adapter_type (str): The adapter type to check.

        Returns:
            bool: True if the adapter type is supported, False otherwise.
        """
        if isinstance(type_or_config, AdapterConfig):
            types = AdapterMethod.get_from_config(type_or_config)
        else:
            types = [type_or_config]

        supported = []
        for _type in types:
            if getattr(self.base_model, "adapter_interface", None) is not None:
                supported.append(
                    _type in self.base_model.adapter_interface.adapter_methods
                )
            elif _type == AdapterMethod.prompt_tuning:
                supported.append(self.base_model.support_prompt_tuning)
            elif _type == AdapterMethod.invertible:
                supported.append(
                    isinstance(self, InvertibleAdaptersMixin)
                    or isinstance(self, InvertibleAdaptersWrapperMixin)
                )
            else:
                supported.append(True)
        return all(supported)

    # These methods have to be implemented by every deriving class:

    @abstractmethod
    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        """
        Iterates over all layers of the model.

        This abstract method has to ne implemented by every implementing model.
        """
        pass

    def apply_to_adapter_layers(self, fn):
        """
        Applies a function to all adapter layers of the model.
        """
        for i, layer in self.iter_layers():
            for module in layer.modules():
                if isinstance(module, AdapterLayerBase):
                    fn(i, module)

    def apply_to_basemodel_childs(self, fn):
        """
        Applies a function to all direct childs of the model if they are a instance of AdapterLayerBase.
        """
        if self.base_model.add_base_adapters:
            for module in self.base_model.children():
                if isinstance(module, AdapterLayerBase):
                    # These childs don't have a layer index so we pass -1
                    fn(-1, module)

    def train_adapter(
        self,
        adapter_setup: Union[list, AdapterCompositionBlock],
        train_embeddings=False,
    ):
        """Sets the model into mode for training the given adapters."""
        self.train()
        self.freeze_model(True)
        adapter_setup = parse_composition(adapter_setup)
        self.apply_to_adapter_layers(
            lambda i, layer: layer.enable_adapters(adapter_setup, True, False)
        )
        self.apply_to_basemodel_childs(
            lambda i, child: child.enable_adapters(adapter_setup, True, False)
        )
        for adapter_name in adapter_setup:
            if adapter_name in self.base_model.shared_parameters:
                if False:  # LoRAConfig not supported
                    adapter_config = False  # LoRAConfig not supported
                    if isinstance(adapter_config.vera_d, float) or isinstance(
                        adapter_config.vera_b, float
                    ):
                        for param in self.base_model.shared_parameters[
                            adapter_name
                        ].values():
                            param.requires_grad = False
                    else:
                        for param in self.base_model.shared_parameters[
                            adapter_name
                        ].values():
                            param.requires_grad = True
                else:
                    for param in self.base_model.shared_parameters[
                        adapter_name
                    ].values():
                        param.requires_grad = True

        if isinstance(self, InvertibleAdaptersMixin) or isinstance(
            self, InvertibleAdaptersWrapperMixin
        ):
            self.enable_invertible_adapters(adapter_setup.flatten())
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_setup)
        if train_embeddings:
            self.get_input_embeddings().train()
            self.get_input_embeddings().weight.requires_grad = True

    def train_adapter_fusion(
        self,
        adapter_setup: Union[list, AdapterCompositionBlock],
        unfreeze_adapters=False,
    ):
        """Sets the model into mode for training of adapter fusion determined by a list of adapter names."""
        self.train()
        self.freeze_model(True)
        adapter_setup = parse_composition(adapter_setup)
        self.apply_to_adapter_layers(
            lambda i, layer: layer.enable_adapters(
                adapter_setup, unfreeze_adapters, True
            )
        )
        self.apply_to_basemodel_childs(
            lambda i, child: child.enable_adapters(
                adapter_setup, unfreeze_adapters, True
            )
        )
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_setup)

    def has_adapters(self):
        return len(self.adapters_config.adapters) > 0

    @property
    def has_parallel_adapters(self) -> bool:
        if self.adapters_config.active_setup:
            return self.adapters_config.active_setup.parallel_channels > 1
        else:
            return False

    @property
    def active_adapters(self) -> AdapterCompositionBlock:
        return self.adapters_config.active_setup

    @active_adapters.setter
    def active_adapters(self, adapter_setup: Union[list, AdapterCompositionBlock]):
        self.set_active_adapters(adapter_setup)

    def set_shared_parameters(self, param):
        self.base_model.shared_parameters = param

    def set_active_adapters(
        self,
        adapter_setup: Union[list, AdapterCompositionBlock],
        skip_layers: Optional[List[int]] = None,
    ):
        """
        Sets the adapter modules to be used by default in every forward pass. If no adapter with the given name is
        found, no module of the respective type will be activated.

        Args:
            adapter_setup (list):
                The list of adapters to be activated by default. Can be a fusion or stacking configuration.
        """
        adapter_setup = parse_composition(
            adapter_setup, model_type=self.config.model_type
        )
        if adapter_setup:
            for adapter_name in adapter_setup.flatten():
                if adapter_name not in self.adapters_config.adapters:
                    raise ValueError(
                        f"No adapter with name '{adapter_name}' found. Please make sure that all specified adapters"
                        " are correctly loaded."
                    )

        # Make sure LoRA is reset
        self.reset_adapter()
        self.adapters_config.active_setup = adapter_setup
        self.adapters_config.skip_layers = skip_layers

    def add_adapter(
        self,
        adapter_name: str,
        config=None,
        overwrite_ok: bool = False,
        set_active: bool = False,
    ):
        """
        Adds a new adapter module of the specified type to the model.

        Args:
            adapter_name (str): The name of the adapter module to be added.
            config (str or dict or AdapterConfig, optional): The adapter configuration, can be either:

                - the string identifier of a pre-defined configuration dictionary
                - a configuration dictionary specifying the full config
                - if not given, the default configuration for this adapter type will be used
            overwrite_ok (bool, optional):
                Overwrite an adapter with the same name if it exists. By default (False), an
            exception is thrown. set_active (bool, optional):
                Set the adapter to be the active one. By default (False),
            the adapter is added but not activated.
        """
        config = AdapterConfig.load(config)  # ensure config is ok and up-to-date
        config_or_type = config or AdapterMethod.bottleneck
        if not self.supports_adapter(config_or_type):
            raise ValueError(
                f"Adapter config or type '{config_or_type}' is not supported by this model."
            )
        if overwrite_ok and adapter_name in self.adapters_config:
            self.delete_adapter(adapter_name)
        self.adapters_config.add(adapter_name, config=config)
        try:
            self._add_adapter_weights(adapter_name)
        except ValueError as ex:
            self.delete_adapter(adapter_name)
            raise ex
        if set_active:
            self.set_active_adapters(adapter_name)

        # For VeRA adapters, register tied weights patterns
        if False:  # LoRAConfig not supported
            adapter_config = False  # LoRAConfig not supported
            if isinstance(adapter_config.vera_d, float) or isinstance(
                adapter_config.vera_b, float
            ):
                vera_tied_weights_keys = [
                    f"shared_parameters\\.{adapter_name}\\.lora_A",
                    f"shared_parameters\\.{adapter_name}\\.lora_B",
                ]

                if self._tied_weights_keys is not None:
                    self._tied_weights_keys += vera_tied_weights_keys
                else:
                    self._tied_weights_keys = vera_tied_weights_keys

    def _add_adapter_weights(self, adapter_name: str):
        """Helper method that performs the actual parameter additions when adding a new adapter."""
        self.apply_to_adapter_layers(
            lambda i, layer: layer.add_adapter(adapter_name, i)
        )
        self.apply_to_basemodel_childs(
            lambda i, child: child.add_adapter(adapter_name, i)
        )

        # PHM Layer
        if self.adapters_config.match(adapter_name, BnConfig, location_key="phm_layer"):
            adapter_config = self.adapters_config.match(
                adapter_name, BnConfig, location_key="phm_layer"
            )
            if adapter_config["shared_phm_rule"] or adapter_config["shared_W_phm"]:
                if self.config.model_type in SUBMODEL_NAMES:
                    hidden_sizes = [
                        getattr(self.config, key).hidden_size
                        for key in SUBMODEL_NAMES[self.config.model_type]
                    ]
                    if all(hidden_sizes[0] == h for h in hidden_sizes):
                        self.base_model.shared_parameters[
                            adapter_name
                        ] = init_shared_parameters(
                            adapter_config, hidden_sizes[0], self.device
                        )
                    else:
                        raise ValueError(
                            "The model has different hidden sizes {}. Sharing compacter weights is only possible if"
                            " the hidden_sizes match.".format(hidden_sizes)
                        )
                else:
                    self.base_model.shared_parameters[
                        adapter_name
                    ] = init_shared_parameters(
                        adapter_config, self.config.hidden_size, self.device
                    )

        # Vera Initialization
        if False:  # LoRAConfig not supported
            # depends on the architecture field of the adapter config
            adapter_config = False  # LoRAConfig not supported
            if isinstance(adapter_config.vera_d, float) or isinstance(
                adapter_config.vera_b, float
            ):
                if self.config.model_type in SUBMODEL_NAMES:
                    hidden_sizes = [
                        getattr(self.config, key).hidden_size
                        for key in SUBMODEL_NAMES[self.config.model_type]
                    ]
                    if not (all(hidden_sizes[0] == h for h in hidden_sizes)):
                        raise ValueError(
                            "The model has different hidden sizes {}. Vera uses shared LoRA A and B matrices and thus initialization is only possible if the hidden_sizes match.".format(
                                hidden_sizes
                            )
                        )

                # Next, init the shared parameters of Vera
                shapes_info = self.adapters_config._vera_init_shapes[adapter_name]
                lora_A_shape = shapes_info["lora_A_shape"]
                lora_B_shape = shapes_info["lora_B_shape"]
                self.base_model.shared_parameters[
                    adapter_name
                ] = init_shared_vera_parameters(
                    lora_A_shape, lora_B_shape, adapter_config, self.device
                )

        # Prefix Tuning
        for module in self.modules():
            if False:  # PrefixTuningPool not supported
                module.confirm_prefix(adapter_name)
        if isinstance(self, InvertibleAdaptersMixin) or isinstance(
            self, InvertibleAdaptersWrapperMixin
        ):
            self.add_invertible_adapter(adapter_name)

    def share_parameters(
        self,
        adapter_names: Union[MultiTask, list, str],
        name: Optional[str] = None,
        reference_adapter_name: Optional[str] = None,
    ):
        """
        Shares parameters across specified adapter layers and base model children.

        This method enables parameter sharing between multiple adapters by linking
        their parameters to a common reference. It applies the sharing operation to
        both adapter layers and base model child modules.

        Args:
            adapter_names (Union[MultiTask, list, str]): The names of the adapters whose
                parameters should be shared. If a `MultiTask` object is provided, its child
                adapter names will be used.
            name (Optional[str], default=None): A custom name for the shared parameters.
                If not provided, the name is derived by concatenating `adapter_names`.
            reference_adapter_name (Optional[str], default=None): The name of an existing
                adapter to use as a reference for parameter sharing.

        Raises:
            TypeError: If any adapter configuration is not of type `MultiTaskConfig`.
            ValueError: If the reference adapter is not in the provided adapter names.
            AssertionError: If the adapter list is empty.
        """
        if isinstance(adapter_names, MultiTask):
            adapter_names = adapter_names.children
        elif isinstance(adapter_names, str):
            adapter_names = adapter_names.split(",")
        if name is None:
            name = ",".join(adapter_names)

        reference_adapter_name = reference_adapter_name or adapter_names[0]
        assert (
            len(adapter_names) > 0
        ), "Expected at least one adapter name, but got an empty list."

        # Check that all adapter configurations exist and have the same type
        adapter_configs = []
        for adapter_name in adapter_names:
            adapter_config = self.adapters_config.get(adapter_name)
            if adapter_config is None:
                raise ValueError(
                    f"No configuration found for adapter '{adapter_name}'."
                )
            if not isinstance(adapter_config, MultiTaskConfig):
                raise TypeError(
                    f"Expected adapter configuration of type 'MultiTaskConfig' for adapter '{adapter_name}', but got '{type(adapter_config).__name__}' instead."
                )
            adapter_configs.append(adapter_config)

        # Ensure all adapter configurations have the same type
        config_types = {type(config) for config in adapter_configs}
        if len(config_types) > 1:
            raise TypeError(
                f"All adapter configurations must be of the same type, but found multiple types: {config_types}"
            )

        if (
            reference_adapter_name is not None
            and reference_adapter_name not in adapter_names
        ):
            raise ValueError(
                f"Reference adapter '{reference_adapter_name}' not found in the provided adapter names: {adapter_names}."
            )

        self.apply_to_adapter_layers(
            lambda i, layer: layer.share_parameters(
                name=name,
                adapter_names=adapter_names,
                reference_adapter_name=reference_adapter_name,
            )
        )
        self.apply_to_basemodel_childs(
            lambda i, child: child.share_parameters(
                name=name,
                adapter_names=adapter_names,
                reference_adapter_name=reference_adapter_name,
            )
        )

    def unshare_parameters(
        self,
        adapter_names: Union[MultiTask, list, str],
        name: Optional[str] = None,
    ):
        """
        Removes parameter sharing across specified adapter layers and base model children.

        This method detaches shared parameters among the given adapters, restoring them
        to independent parameter sets. The operation is applied to both adapter layers
        and base model child modules.

        Args:
            adapter_names (Union[MultiTask, list, str]): The names of the adapters whose
                shared parameters should be unlinked. If a `MultiTask` object is provided,
                its child adapter names will be used.
            name (Optional[str], default=None): A custom name for the unshared parameters.
                If not provided, the name is derived by concatenating `adapter_names`.

        """
        if isinstance(adapter_names, MultiTask):
            adapter_names = adapter_names.children
        elif isinstance(adapter_names, str):
            adapter_names = adapter_names.split(",")
        if name is None:
            name = ",".join(adapter_names)

        self.apply_to_adapter_layers(
            lambda i, layer: layer.unshare_parameters(
                name=name,
            )
        )
        self.apply_to_basemodel_childs(
            lambda i, child: child.unshare_parameters(
                name=name,
            )
        )

    def add_adapter_fusion(
        self,
        adapter_names: Union[Fuse, list, str],
        config=None,
        name: str = None,
        overwrite_ok: bool = False,
        set_active: bool = False,
    ):
        """
        Adds AdapterFusion to the model with alll the necessary configurations and weight initializations

        Args:
            adapter_names (Fuse or list or str): AdapterFusion layer to add. Can be either:

                - a ``Fuse`` composition block
                - a list of adapter names to fuse
                - a comma-separated string of adapter names to fuse
            config (str or dict): adapter fusion configuration, can be either:

                - a string identifying a pre-defined adapter fusion configuration
                - a dictionary representing the adapter fusion configuration
                - the path to a file containing the adapter fusion configuration
            name (str, optional):
                Name of the AdapterFusion layer. If not specified, the name is generated automatically from the fused adapter names.
            overwrite_ok (bool, optional):
                Overwrite an AdapterFusion layer with the same name if it exists. By default (False), an exception is
                thrown.
            set_active (bool, optional):
                Activate the added AdapterFusion. By default (False), the AdapterFusion is added but not activated.
        """
        if isinstance(adapter_names, Fuse):
            if name is None:
                name = adapter_names.name
            adapter_names = adapter_names.children
        elif isinstance(adapter_names, str):
            adapter_names = adapter_names.split(",")
        if name is None:
            name = ",".join(adapter_names)

        if isinstance(config, dict):
            config = AdapterFusionConfig.from_dict(
                config
            )  # ensure config is ok and up-to-date
        if overwrite_ok and self.adapters_config.get_fusion(name)[0] is not None:
            self.delete_adapter_fusion(name)
        self.adapters_config.add_fusion(adapter_names, config=config, fusion_name=name)
        self.apply_to_adapter_layers(lambda i, layer: layer.add_fusion_layer(name))
        self.apply_to_basemodel_childs(lambda i, child: child.add_fusion_layer(name))
        if set_active:
            self.set_active_adapters(Fuse(*adapter_names, name=name))

    def delete_adapter(self, adapter_name: str):
        """
        Deletes the adapter with the specified name from the model.

        Args:
            adapter_name (str): The name of the adapter.
        """
        if adapter_name not in self.adapters_config:
            logger.info("No adapter '%s' found for deletion. Skipping.", adapter_name)
            return
        self.apply_to_adapter_layers(
            lambda i, layer: layer.delete_adapter(adapter_name)
        )
        self.apply_to_basemodel_childs(
            lambda i, child: child.delete_adapter(adapter_name)
        )
        del self.adapters_config.adapters[adapter_name]
        if adapter_name in self.base_model.shared_parameters:
            del self.base_model.shared_parameters[adapter_name]
        if isinstance(self, InvertibleAdaptersMixin) or isinstance(
            self, InvertibleAdaptersWrapperMixin
        ):
            self.delete_invertible_adapter(adapter_name)

        if self.active_adapters == Stack(adapter_name):
            self.active_adapters = None

    def delete_adapter_fusion(self, adapter_names: Union[Fuse, list, str]):
        """
        Deletes the AdapterFusion layer of the specified adapters.

        Args:
            adapter_names (Union[Fuse, list, str]): AdapterFusion layer to delete.
        """
        if isinstance(adapter_names, Fuse):
            adapter_fusion_name = adapter_names.name
        elif isinstance(adapter_names, list):
            adapter_fusion_name = ",".join(adapter_names)
        elif isinstance(adapter_names, str):
            adapter_fusion_name = adapter_names
        else:
            raise ValueError(
                "Invalid AdapterFusion definition: {}".format(adapter_names)
            )

        if adapter_fusion_name not in self.adapters_config.fusions:
            logger.info(
                "No AdapterFusion '%s' found for deletion. Skipping.",
                adapter_fusion_name,
            )
            return
        del self.adapters_config.fusions[adapter_fusion_name]
        self.apply_to_adapter_layers(
            lambda i, layer: layer.delete_fusion_layer(adapter_fusion_name)
        )
        self.apply_to_basemodel_childs(
            lambda i, child: child.delete_fusion_layer(adapter_fusion_name)
        )
        if self.active_adapters == adapter_names:
            self.active_adapters = None

    def save_adapter(
        self,
        save_directory: str,
        adapter_name: str,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        use_safetensors: bool = False,
        **kwargs,
    ):
        """
        Saves an adapter and its configuration file to a directory so that it can be shared or reloaded using
        `load_adapter()`.

        Args:
            save_directory (str): Path to a directory where the adapter should be saved.
            adapter_name (str): Name of the adapter to be saved.
            use_safetensors (bool, optional): If True, weights are saved via `safetensors`. Otherwise, the regular torch save method is used.

        Raises:
            ValueError: If the given adapter name is invalid.
        """
        loader = AdapterLoader(self, use_safetensors=use_safetensors)
        loader.save(save_directory, adapter_name, meta_dict)
        # save additional custom weights

        # save interface in case it is a custom model
        if interface := getattr(self.base_model, "adapter_interface", None):
            interface._save(save_directory, self.config)

        if custom_weights_loaders:
            for weights_loader in custom_weights_loaders:
                weights_loader.save(save_directory, adapter_name)

    def save_adapter_fusion(
        self,
        save_directory: str,
        adapter_names: Union[Fuse, list, str],
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        use_safetensors: bool = False,
    ):
        """
        Saves an AdapterFusion layer and its configuration file to a directory so that it can be shared or reloaded
        using `load_adapter_fusion()`.

        Args:
            save_directory (str): Path to a directory where the AdapterFusion should be saved.
            adapter_names (Union[Fuse, list, str]): AdapterFusion to be saved.
            use_safetensors (bool, optional): If True, weights are saved via `safetensors`. Otherwise, the regular torch save method is used.

        Raises:
            ValueError: If the given AdapterFusion name is invalid.
        """
        if isinstance(adapter_names, Fuse):
            adapter_fusion_name = adapter_names.name
        elif isinstance(adapter_names, list):
            adapter_fusion_name = ",".join(adapter_names)
        elif isinstance(adapter_names, str):
            adapter_fusion_name = adapter_names
        else:
            raise ValueError(
                "Invalid AdapterFusion definition: {}".format(adapter_names)
            )

        loader = AdapterFusionLoader(self, use_safetensors=use_safetensors)
        loader.save(save_directory, adapter_fusion_name, meta_dict)
        # save additional custom weights
        if custom_weights_loaders:
            for weights_loader in custom_weights_loaders:
                weights_loader.save(save_directory, adapter_fusion_name)

    def load_adapter(
        self,
        adapter_name_or_path: str,
        config: Union[dict, str] = None,
        version: str = None,
        model_name: str = None,
        load_as: str = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        leave_out: Optional[List[int]] = None,
        id2label=None,
        set_active: bool = False,
        use_safetensors: bool = False,
        **kwargs,
    ) -> str:
        """
        Loads a pre-trained pytorch adapter module from the local file system or a remote location.

        Args:
            adapter_name_or_path (str): can be either:

                - the identifier of a pre-trained task adapter to be loaded from Adapter Hub
                - a path to a directory containing adapter weights saved using `model.save_adapter()`
                - a URL pointing to a zip folder containing a saved adapter module
            config (dict or str, optional): Deprecated.
            version (str, optional): The version of the adapter to be loaded.
            model_name (str, optional): Deprecated.
            load_as (str, optional): Load the adapter using this name. By default, the name with which the adapter was
                    saved will be used.
            leave_out: Dynamically drop adapter modules in the specified Transformer layers when loading the adapter.
            set_active (bool, optional):
                Set the loaded adapter to be the active one. By default (False), the adapter is loaded but not
                activated.
            use_safetensors (bool, optional): If True, weights are loaded via `safetensors` if safetensors checkpoint is available. Otherwise, the regular torch save method is used.

        Returns:
            str: The name with which the adapter was added to the model.
        """
        loader = AdapterLoader(self, use_safetensors=use_safetensors)
        load_dir, load_name = loader.load(
            adapter_name_or_path,
            config,
            version,
            model_name,
            load_as,
            leave_out=leave_out,
            set_active=set_active,
            **kwargs,
        )
        # load additional custom weights
        if custom_weights_loaders:
            for weights_loader in custom_weights_loaders:
                weights_loader.load(
                    load_dir,
                    load_as=load_as,
                    loading_info=kwargs.get("loading_info", None),
                    main_load_name=load_name,
                    id2label=id2label,
                    set_active=set_active,
                )
        return load_name

    def load_adapter_fusion(
        self,
        adapter_fusion_name_or_path: str,
        load_as: str = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        set_active: bool = False,
        use_safetensors: bool = False,
        **kwargs,
    ) -> str:
        """
        Loads a pre-trained AdapterFusion layer from the local file system.

        Args:
            adapter_fusion_name_or_path (str):
                a path to a directory containing AdapterFusion weights saved using `model.save_adapter_fusion()`.
            load_as (str, optional): Load the AdapterFusion using this name.
                    By default, the name with which the AdapterFusion layer was saved will be used.
            set_active (bool, optional):
                Activate the loaded AdapterFusion. By default (False), the AdapterFusion is loaded but not activated.
            use_safetensors (bool, optional): If True, weights are loaded via `safetensors` if safetensors checkpoint is available. Otherwise, the regular torch save method is used.

        Returns:
            str: The name with which the AdapterFusion was added to the model.
        """

        loader = AdapterFusionLoader(self, use_safetensors=use_safetensors)
        load_dir, load_name = loader.load(
            adapter_fusion_name_or_path, load_as, set_active=set_active
        )
        # load additional custom weights
        if custom_weights_loaders:
            for weights_loader in custom_weights_loaders:
                weights_loader.load(
                    load_dir,
                    load_as=load_as,
                    loading_info=kwargs.get("loading_info", None),
                    main_load_name=load_name,
                    set_active=set_active,
                )
        return load_name

    def _save_adapter_setup_config(
        self,
        save_directory: str,
        adapter_setup: AdapterCompositionBlock,
        head_setup: Optional[Union[bool, str, list, AdapterCompositionBlock]] = None,
    ):
        setup_config = {
            "adapter_setup": adapter_setup.to_dict(),
            "head_setup": (
                head_setup.to_dict()
                if isinstance(head_setup, AdapterCompositionBlock)
                else head_setup
            ),
        }
        with open(join(save_directory, SETUP_CONFIG_NAME), "w") as f:
            json.dump(setup_config, f, indent=2)

    def _load_adapter_setup_config(
        self, load_directory: str
    ) -> Tuple[AdapterCompositionBlock, Optional[AdapterCompositionBlock]]:
        with open(join(load_directory, SETUP_CONFIG_NAME), "r") as f:
            setup_config = json.load(f)
        adapter_setup = AdapterCompositionBlock.from_dict(setup_config["adapter_setup"])
        head_setup = setup_config["head_setup"]
        if isinstance(head_setup, dict):
            head_setup = AdapterCompositionBlock.from_dict(head_setup)
        return adapter_setup, head_setup

    def _save_adapter_setup_weights(
        self,
        save_directory: str,
        adapter_setup: AdapterCompositionBlock,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        use_safetensors: bool = False,
    ):
        # Save single adapters
        for adapter_name in adapter_setup.flatten():
            save_path = join(save_directory, adapter_name)
            self.save_adapter(
                save_path,
                adapter_name,
                meta_dict=meta_dict,
                use_safetensors=use_safetensors,
            )
        # Save adapter fusions
        fusions = []
        if isinstance(adapter_setup, Fuse):
            fusions.append(adapter_setup)
        for child_setup in adapter_setup.children:
            if isinstance(child_setup, Fuse):
                fusions.append(child_setup)
        for fusion in fusions:
            save_path = join(save_directory, fusion.name)
            self.save_adapter_fusion(
                save_path,
                fusion,
                meta_dict=meta_dict,
                use_safetensors=use_safetensors,
            )
        # Save additional custom weights
        if custom_weights_loaders:
            for weights_loader in custom_weights_loaders:
                weights_loader.save(save_directory, adapter_name)

    def _load_adapter_setup_weights(
        self,
        load_directory: str,
        adapter_setup: AdapterCompositionBlock,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        set_active: bool = False,
        use_safetensors: bool = False,
    ):
        # Load single adapters
        for adapter_name in adapter_setup.flatten():
            save_path = join(load_directory, adapter_name)
            self.load_adapter(save_path, use_safetensors=use_safetensors)
        # Load adapter fusions
        fusions = []
        if isinstance(adapter_setup, Fuse):
            fusions.append(adapter_setup)
        for child_setup in adapter_setup.children:
            if isinstance(child_setup, Fuse):
                fusions.append(child_setup)
        for fusion in fusions:
            save_path = join(load_directory, fusion.name)
            self.load_adapter_fusion(save_path, use_safetensors=use_safetensors)
        # Load additional custom weights
        if custom_weights_loaders:
            for weights_loader in custom_weights_loaders:
                weights_loader.load(load_directory)

        if set_active:
            self.set_active_adapters(adapter_setup)

    def save_adapter_setup(
        self,
        save_directory: str,
        adapter_setup: Union[str, list, AdapterCompositionBlock],
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        use_safetensors: bool = False,
    ):
        """Saves an adapter setup to a directory so that it can be shared or reloaded using `load_adapter_setup()`.

        Args:
            save_directory (str): Path to a directory where the adapter setup should be saved.
            adapter_setup (Union[str, list, AdapterCompositionBlock]): The adapter setup to be saved. Usually an adapter composition block.
            use_safetensors (bool, optional): If True, weights are saved via `safetensors`. Otherwise, the regular torch save method is used.
        """
        os.makedirs(save_directory, exist_ok=True)
        adapter_setup = parse_composition(
            adapter_setup, model_type=self.config.model_type
        )

        self._save_adapter_setup_config(save_directory, adapter_setup)
        self._save_adapter_setup_weights(
            save_directory,
            adapter_setup,
            meta_dict=meta_dict,
            custom_weights_loaders=custom_weights_loaders,
            use_safetensors=use_safetensors,
        )

    def load_adapter_setup(
        self,
        adapter_setup_name_or_path: str,
        version: str = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        set_active: bool = False,
        use_safetensors: bool = False,
        **kwargs,
    ) -> Tuple[AdapterCompositionBlock, Any]:
        """Loads an adapter setup from the local file system or a remote location.

        Args:
            adapter_setup_name_or_path (str): can be either:

                - the identifier of a repository on the HuggingFace Model Hub.
                - a path to a directory containing adapter weights saved using `model.save_adapter_setup()`
                - a URL pointing to a zip folder containing a saved adapter module
            version (str, optional): The version of the adapter to be loaded.
            set_active (bool, optional):
                Set the loaded adapter setup to be the active one. By default (False), the adapter setup is loaded but not
                activated.
            use_safetensors (bool, optional): If True, weights are loaded via `safetensors` if safetensors checkpoint is available. Otherwise, the regular torch save method is used.

        Returns:
            Tuple[AdapterCompositionBlock, Any]: The loaded adapter setup and the head setup if available.
        """
        resolved_folder = resolve_adapter_path(
            adapter_setup_name_or_path,
            version=version,
            do_exists_check=False,
            **kwargs,
        )
        adapter_setup, head_setup = self._load_adapter_setup_config(resolved_folder)
        self._load_adapter_setup_weights(
            resolved_folder,
            adapter_setup,
            custom_weights_loaders=custom_weights_loaders,
            set_active=set_active,
            use_safetensors=use_safetensors,
        )

        if head_setup:
            logger.warning(
                "Loaded adapter setup contains a head setup that is not supported by the current model."
            )

        return adapter_setup, head_setup

    def save_all_adapters(
        self,
        save_directory: str,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        use_safetensors: bool = False,
    ):
        """
        Saves all adapters of this model together with their configuration to subfolders of the given location.

        Args:
            save_directory (str): Path to a directory where the adapters should be saved.
            use_safetensors (bool, optional): If True, weights are saved via `safetensors`. Otherwise, the regular torch save method is used.
        """
        os.makedirs(save_directory, exist_ok=True)
        for name in self.adapters_config:
            adapter_config = self.adapters_config.get(name)
            h = get_adapter_config_hash(adapter_config)
            save_path = join(save_directory, name)
            if meta_dict:
                meta_dict.update({"config_id": h})
            else:
                meta_dict = {"config_id": h}
            self.save_adapter(
                save_path,
                name,
                meta_dict=meta_dict,
                custom_weights_loaders=custom_weights_loaders,
                use_safetensors=use_safetensors,
            )

    def save_all_adapter_fusions(
        self,
        save_directory: str,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        use_safetensors: bool = False,
    ):
        """
        Saves all AdapterFusion layers of this model together with their configuration to subfolders of the given
        location.

        Args:
            save_directory (str): Path to a directory where the AdapterFusion layers should be saved.
            use_safetensors (bool, optional): If True, weights are saved via `safetensors`. Otherwise, the regular torch save method is used.
        """
        os.makedirs(save_directory, exist_ok=True)
        for name in self.adapters_config.fusions:
            adapter_fusion_config, _ = self.adapters_config.get_fusion(name)
            h = get_adapter_config_hash(adapter_fusion_config)
            save_path = join(save_directory, name)
            if meta_dict:
                meta_dict.update({"config_id": h})
            else:
                meta_dict = {"config_id": h}
            self.save_adapter_fusion(
                save_path,
                name,
                meta_dict=meta_dict,
                custom_weights_loaders=custom_weights_loaders,
                use_safetensors=use_safetensors,
            )

    def freeze_model(self, freeze=True):
        """Freezes all weights of the model."""
        # first freeze/ unfreeze all model weights
        for param in self.base_model.parameters():
            param.requires_grad = not freeze
        self.model_frozen = freeze

    def forward_context(self, context: ForwardContext, *args, **kwargs):
        """
        This method is called by the ``ForwardContext`` at the beginning of the forward pass.
        """
        if "task_ids" in kwargs:
            context.task_ids = kwargs.pop("task_ids")

        active_adapters = (
            getattr(self, "active_adapters", None)
            or AdapterSetup.get_context_adapter_setup()
        )
        if not active_adapters:
            if self.has_adapters():
                logger.warning(
                    "There are adapters available but none are activated for the forward pass."
                )
            return

        context.adapters_parallelized = False
        if context.adapter_input_parallelized:
            if active_adapters.parallel_channels > 1:
                context.adapters_parallelized = True
        context.shared_parameters = {
            name: param
            for name, param in self.base_model.shared_parameters.items()
            if name in active_adapters.flatten()
        }

        if hasattr(self.base_model, "prefix_tuning"):
            context.prefix_states = self.base_model.prefix_tuning(*args, **kwargs)

        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
        elif len(args) > 1:
            attention_mask = args[1]
        else:
            attention_mask = None
        if attention_mask is not None:
            context.seqlens = (attention_mask == 1).sum(dim=-1).squeeze()
            context.offsets = attention_mask.argmax(1)

        # Adapter gating and attention outputs
        context.adapter_gating_scores = defaultdict(dict)
        context.adapter_fusion_attentions = defaultdict(dict)

    def get_fusion_regularization_loss(self):
        reg_loss = None

        target = (
            torch.zeros((self.config.hidden_size, self.config.hidden_size))
            .fill_diagonal_(1.0)
            .to(self.device)
        )
        for i, layer in self.iter_layers():
            for module in layer.modules():
                if isinstance(module, BottleneckLayer):
                    for _, layer_fusion in module.adapter_fusion_layer.items():
                        if (
                            hasattr(layer_fusion, "value")
                            and layer_fusion.value.weight.requires_grad
                        ):
                            layer_reg_loss = (
                                0.01 * (target - layer_fusion.value.weight).pow(2).sum()
                            )
                            if reg_loss is None:
                                reg_loss = layer_reg_loss
                            else:
                                reg_loss += layer_reg_loss

        return reg_loss

    def get_adapter(self, name) -> dict:
        """
        Returns a dictionary with all weights of the adapter with the specified name.

        Args:
            name (str): The adapter name.

        Returns:
            dict: A nested dictionary containing the weights of the adapter. The dictionary is structured as follow:
            {<layer id>: {<module location>: <nn.Module>}}. <layer id> = -1 indicates global/ shared weights.
        """
        destination = defaultdict(dict)

        # global weights are saved at index -1
        if name in self.base_model.shared_parameters:
            destination[-1]["shared"] = self.base_model.shared_parameters[name]
        if self.supports_adapter("invertible") and name in self.invertible_adapters:
            destination[-1]["invertible"] = self.invertible_adapters[name]

        pass

        for i, (_, layer) in enumerate(self.iter_layers()):
            for module in layer.modules():
                if isinstance(module, AdapterLayerBase):
                    adapter_module = module.get_adapter(name)
                    if adapter_module is not None:
                        # location_key might already be added before -> concat to ModuleList
                        if module.location_key in destination[i]:
                            old_module = destination[i][module.location_key]
                            if isinstance(old_module, nn.ModuleList):
                                old_module.append(adapter_module)
                            else:
                                destination[i][module.location_key] = nn.ModuleList(
                                    [old_module, adapter_module]
                                )
                        else:
                            destination[i][module.location_key] = adapter_module

        return dict(destination)

    def adapter_to(
        self,
        name: str,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Moves the adapter with the given name to the specified device and data type.

        Args:
            name (str): The name of the adapter to be moved.
            device (torch.device or str, optional): The device on which the adapter should be moved.
            dtype (torch.dtype, optional): The data type to which the adapter should be cast.
        """
        for _, v in self.get_adapter(name).items():
            for _, module in v.items():
                module.to(device=device, dtype=dtype)

    def adapter_fusion_to(
        self,
        adapter_names: Union[Fuse, list, str],
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Moves the adapter fusion layer with the given name to the specified device and data type.

        Args:
            adapter_names (Union[Fuse, list, str]): The name of the adapter fusion layer to be moved.
            device (torch.device or str, optional): The device on which the adapter fusion layer should be moved.
            dtype (torch.dtype, optional): The data type to which the adapter fusion layer should be cast.
        """
        for _, layer in self.iter_layers():
            for module in layer.modules():
                if isinstance(module, BottleneckLayer):
                    fusion = module.get_adapter_fusion(adapter_names)
                    if fusion is not None:
                        fusion.to(device=device, dtype=dtype)

    def adapter_summary(self, as_dict=False) -> Union[str, dict]:
        """
        Returns a string summary of all adapters currently added to the model. Each entry in the summary table has the
        following attributes:

            - name: the name of the adapter
            - architecture: the architectural base of the adapter
            - #param: the number of parameters of the adapter
            - %param: the number of parameters of the adapter relative to the full model
            - active: whether the adapter is active
            - train: whether the adapter weights are enabled for training
        """
        # table header
        header = ["name", "architecture", "#param", "%param", "active", "train"]
        # rows containing adapter info
        rows = []
        for name, config_name in self.adapters_config.adapters.items():
            if config_name in self.adapters_config.config_map:
                config = self.adapters_config.config_map.get(config_name, None)
            else:
                config = ADAPTER_CONFIG_MAP.get(config_name, None)
            if isinstance(config, str):
                config = ADAPTER_CONFIG_MAP[config]
            row = {
                "name": name,
                "architecture": config.get("architecture", None) or "bottleneck",
            }
            weights = self.get_adapter(name)
            row["active"] = (
                self.active_adapters is not None
                and name in self.active_adapters.flatten()
            )
            # count parameters
            no_params = 0
            train = True
            for _, module_dict in weights.items():
                for _, module in module_dict.items():
                    no_params += sum(p.numel() for p in module.parameters())
                    train &= all(p.requires_grad for p in module.parameters())
            row["#param"] = no_params
            row["train"] = train
            rows.append(row)
        # count no. of parameters in base network
        model_no_params = sum(p.numel() for p in self.base_model.parameters())
        model_no_params -= sum([r["#param"] for r in rows])
        # add %param info
        for row in rows:
            row["%param"] = row["#param"] / model_no_params * 100
        # add full model info
        rows.append(
            {
                "name": "Full model",
                "#param": model_no_params,
                "%param": 100.0,
                "train": not getattr(self.base_model, "model_frozen", False),
            }
        )

        if as_dict:
            return rows
        else:
            # print
            total_length = 80
            header_format = "{:<25}{:<15}{:>12}{:>12}{:>8}{:>8}"
            row_format = "{:<25}{:<15}{:>12,}{:>12.3f}{:>8}{:>8}"
            s = ["=" * total_length]
            s.append(header_format.format(*map(lambda x: x.title(), header)))
            s.append("-" * total_length)
            for row in rows:
                s.append(row_format.format(*[row.get(h, "") for h in header]))
            s.insert(len(s) - 1, "-" * total_length)
            s.append("=" * total_length)
            return "\n".join(s)

    def _average_shared_parameters(
        self,
        adapter_name: str,
        input_adapters: Dict[str, float],
        combine_strategy: str,
    ):
        if combine_strategy != "linear":
            raise ValueError(
                f"Combine strategy {combine_strategy} not supported for shared parameters. Only 'linear' is supported."
            )

        avg_state_dict = {}
        for name, weight in input_adapters.items():
            if name in self.base_model.shared_parameters:
                param_dict = self.base_model.shared_parameters[name]
                for key, value in param_dict.items():
                    if key in avg_state_dict:
                        avg_state_dict[key] += weight * value
                    else:
                        avg_state_dict[key] = weight * value
            else:
                raise ValueError(f"Adapter {name} not found in shared parameters.")
        self.base_model.shared_parameters[adapter_name] = nn.ParameterDict(
            avg_state_dict
        )

    def _pre_average_adapter_checks(
        self,
        adapter_name: str,
        adapter_list: List[str],
        combine_strategy: str,
        valid_combination_strategies: List[str],
        is_head=False,
    ):
        if combine_strategy not in valid_combination_strategies:
            raise ValueError(
                f"Invalid combine_strategy '{combine_strategy}'. Must be one of {valid_combination_strategies}"
            )

        # Some strategies are not supported by all models
        if (
            combine_strategy == "lora_delta_w_svd"
            and not self.base_model.support_lora_delta_w_svd
        ):
            raise ValueError(
                "This model specifically does not support 'lora_delta_w_svd' as a merging method. Please use a"
                " different combine_strategy or a different model."
            )

        head_or_adapter = "head" if is_head else "adapter"

        logging.info(
            f"Creating new {head_or_adapter} called {adapter_name} by averaging {adapter_list}."
        )
        if not is_head:
            logging.info(
                "In case you want to create a new head as well please use the `average_head` function."
            )

        if len(adapter_list) == 0:
            raise ValueError(
                "No adapters to average. Please provide at least one adapter to average."
            )
        if len(adapter_list) == 1:
            logging.info(
                "You provided only one adapter to average. If you set `normalize_weights` to true, this will result in"
                " duplicating the adapter. If not this will result in scaling the adapter weights. We will use the"
                " linear combination strategy for this."
            )

        pass

    def average_adapter(
        self,
        adapter_name: str,
        adapter_list: Union[List[str], Dict[str, float]],
        weights: Optional[List[float]] = None,
        combine_strategy: str = "linear",
        normalize_weights: bool = True,
        overwrite_ok: bool = False,
        set_active: bool = False,
        svd_rank: int = None,  # if other combination strategies are implemented that need new parameters, this should be moved to **kwargs
    ):
        """
        Adds a new adapter module as weighted average of a set of existing adapter modules.

        Args:
            adapter_name (str): The name of the adapter module to be added.
            adapter_list (List[str] or Dict[str, float]):
                Specifies the existing adapters whose weights should be averaged. Can either be a list of adapter names
                or a dictionary mapping adapter names to weights.
            weights (Optional[List[float]], optional): The weights corresponding to each adapter module in the list.
                If not provided, equal weights will be assigned to each adapter.
            combine_strategy (str, optional): The strategy to combine the adapter modules.
                Available options are "linear", "lora_linear_only_negate_b", and "lora_delta_w_svd".
                See https://docs.adapterhub.ml/adapter_composition.html#merging-adapters
                Defaults to "linear".
            normalize_weights (bool, optional): Whether to normalize the weights.
                If True, the weights will be normalized to sum up to 1.
                Defaults to True.
            overwrite_ok (bool, optional):
                Overwrite an adapter with the same name if it exists. By default (False), an exception is thrown.
            set_active (bool, optional):
                Set the adapter to be the active one. By default (False), the adapter is added but not activated.
            svd_rank (int, optional): The rank to be used for Singular Value Decomposition (SVD) when averaging LoRA adapters.
                This parameter is only applicable when the combine_strategy is set to "lora_delta_w_svd".
                Defaults to None.
        """

        valid_combination_strategies = [
            "linear",
            "lora_linear_only_negate_b",
            "lora_delta_w_svd",
        ]
        self._pre_average_adapter_checks(
            adapter_name,
            adapter_list,
            combine_strategy,
            valid_combination_strategies,
        )

        config = None
        for name in adapter_list:
            if config is None:
                config = self.adapters_config.get(name)
            elif get_adapter_config_hash(
                config, ignore_params=["dropout", "init_weights"]
            ) != get_adapter_config_hash(
                self.adapters_config.get(name),
                ignore_params=["dropout", "init_weights"],
            ):
                raise ValueError(
                    "Cannot average adapters with different configurations. "
                    "Please make sure all adapters have the same configuration."
                )

        # In case svd_rank is set, change the config to use the new rank
        if svd_rank is not None:
            if False:  # LoRAConfig not supported
                config = config.replace(r=svd_rank)
            else:
                logging.warning(
                    "SVD rank can only be set when averaging LoRA adapters. Ignoring svd_rank."
                )

        if overwrite_ok and adapter_name in self.adapters_config:
            self.delete_adapter(adapter_name)
        self.adapters_config.add(adapter_name, config=config)
        if weights is None:
            eq_weight = 1.0 / len(adapter_list)
            input_adapters = {name: eq_weight for name in adapter_list}
        else:
            # normalize weights
            if normalize_weights:
                sum_weights = sum(weights)
            else:
                sum_weights = 1.0
            input_adapters = {
                name: weight / sum_weights
                for name, weight in zip(adapter_list, weights)
            }
        try:
            self.apply_to_adapter_layers(
                lambda i, layer: layer.average_adapter(
                    adapter_name,
                    input_adapters,
                    combine_strategy,
                    svd_rank=svd_rank,
                )
            )
            self.apply_to_basemodel_childs(
                lambda i, child: child.average_adapter(
                    adapter_name,
                    input_adapters,
                    combine_strategy,
                    svd_rank=svd_rank,
                )
            )
            # PHM Layer
            if self.adapters_config.match(
                adapter_name, BnConfig, location_key="phm_layer"
            ):
                self._average_shared_parameters(
                    adapter_name, input_adapters, combine_strategy
                )

            # Vera Initialization
            if False:  # LoRAConfig not supported
                # depends on the architecture field of the adapter config
                adapter_config = False  # LoRAConfig not supported
                if isinstance(adapter_config.vera_d, float) or isinstance(
                    adapter_config.vera_b, float
                ):
                    self._average_shared_parameters(
                        adapter_name, input_adapters, combine_strategy
                    )

            # Prefix Tuning
            for module in self.modules():
                if False:  # PrefixTuningPool not supported
                    module.average_prefix(
                        adapter_name, input_adapters, combine_strategy
                    )
            if isinstance(self, InvertibleAdaptersMixin) or isinstance(
                self, InvertibleAdaptersWrapperMixin
            ):
                self._average_invertible_adapter(
                    adapter_name, input_adapters, combine_strategy
                )
        except ValueError as ex:
            self.delete_adapter(adapter_name)
            raise ex
        if set_active:
            self.set_active_adapters(adapter_name)

    def eject_prefix_tuning(self, name: str):
        """
        Converts the prefix tuning with the given name from the reparameterized form into the flat form.

        Args:
            name (str): The name of the prefix tuning.
        """
        for module in self.modules():
            if False:  # PrefixTuningPool not supported
                if name in module.prefix_tunings:
                    module.prefix_tunings[name].eject()

    def merge_adapter(self, name: str):
        """
        Merges the weights of the given LoRA module with the Transformer weights as described in the paper.

        Args:
            name (str): LoRA module to merge.
        """
        with ForwardContext(self, torch.empty(0, 1)):
            if self.base_model.shared_parameters:
                ForwardContext.get_context().shared_parameters = (
                    self.base_model.shared_parameters
                )

            for module in self.modules():
                if False:  # LoRALayer not supported:
                    if name in module.loras:
                        module.merge_adapter(name)

    def reset_adapter(self):
        """
        Resets weights of a LoRA module merged using `model.merge_adapter(name)`.
        """
        with ForwardContext(self, torch.empty(0, 1)):
            if self.base_model.shared_parameters:
                ForwardContext.get_context().shared_parameters = (
                    self.base_model.shared_parameters
                )

            for module in self.modules():
                if False:  # LoRALayer not supported:
                    module.reset_adapter()

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name: Optional[str],
        generation_config: GenerationConfig,
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()
        # as the inputs.
        if hasattr(self, "hf_device_map"):
            if hasattr(encoder, "_hf_hook"):
                encoder._hf_hook.io_same_device = True
            else:
                add_hook_to_module(encoder, AlignDevicesHook(io_same_device=True))

        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = (
            "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        )
        forward_context_args = ["adapter_input_parallelized", "task_ids"]
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value
                for argument, value in encoder_kwargs.items()
                if argument in encoder_signature or argument in forward_context_args
            }
        encoder_kwargs["output_attentions"] = generation_config.output_attentions
        encoder_kwargs["output_hidden_states"] = generation_config.output_hidden_states

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = (
            model_input_name if model_input_name is not None else self.main_input_name
        )
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        with ForwardContext(self, **encoder_kwargs):
            for arg_name in forward_context_args:
                encoder_kwargs.pop(
                    arg_name, None
                )  # This should not be passed to actual model

            model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs

    def _prepare_model_inputs(self, *args, **kwargs):
        input_ids, input_name, model_kwargs = super()._prepare_model_inputs(
            *args, **kwargs
        )

        if (
            hasattr(self, "adapters_config")
            and self.adapters_config.active_setup
            and self.adapters_config.active_setup.parallel_channels > 1
        ):
            # Extract original shape
            input_shape = input_ids.shape
            # Replicate input_ids to match the number of parallel channels
            repeat_shape = [
                self.adapters_config.active_setup.parallel_channels
            ] + [  # first dimension is parallel channels
                1
            ] * (
                len(input_shape) - 1
            )  # residual dims should be replicated parallel_channels times
            input_ids = input_ids.repeat(repeat_shape)
            model_kwargs["adapter_input_parallelized"] = True

        return input_ids, input_name, model_kwargs

    # Override to support saving adapters_config
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        **kwargs,
    ):
        self.config.adapters = self.adapters_config.to_dict()

        self.apply_to_adapter_layers(lambda _, layer: layer.pre_save_adapters())
        # Unlink prefix tuning layers to allow safe serialization
        self.apply_to_adapter_layers(
            lambda i, layer: (
                layer.set_pool(None) if isinstance(layer, PrefixTuningLayer) else None
            )
        )

        if interface := getattr(self.base_model, "adapter_interface", None):
            interface._save(save_directory, self.config)
        super().save_pretrained(save_directory, **kwargs)
        # Remove adapters config
        del self.config.adapters

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".

        We pass the `__call__` method of the modules instead of `forward` because `__call__` attaches all the hooks of
        the module. https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2

        Args:
            gradient_checkpointing_kwargs (dict, *optional*):
                Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.
        """
        if not self.supports_gradient_checkpointing:
            raise ValueError(
                f"{self.__class__.__name__} does not support gradient checkpointing."
            )

        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}

        # >>> START AH Changes <<<
        if "use_reentrant" not in gradient_checkpointing_kwargs:
            # use_reentrant must be set.
            gradient_checkpointing_kwargs["use_reentrant"] = False
        else:
            if gradient_checkpointing_kwargs["use_reentrant"]:
                raise ValueError(
                    "Gradient checkpointing with use_reentrant=True is not supported. For gradient checkpointing, we need to set context_fn, which is only supported by PyTorch when use_reentrant is set to False."
                )

        def gradient_checkpointing_function(function, *args, **kwargs):
            context = ForwardContext.get_context()
            context_fn = lambda: (contextlib.nullcontext(), context)
            return checkpoint(function, *args, context_fn=context_fn, **kwargs)

        gradient_checkpointing_func = functools.partial(
            gradient_checkpointing_function, **gradient_checkpointing_kwargs
        )
        # >>> END AH Changes <<<

        # we will fall back to the overwritten `_set_gradient_checkpointing` method
        _is_using_old_format = (
            "value" in inspect.signature(self._set_gradient_checkpointing).parameters
        )

        if not _is_using_old_format:
            self._set_gradient_checkpointing(
                enable=True,
                gradient_checkpointing_func=gradient_checkpointing_func,
            )
        else:
            self.apply(partial(self._set_gradient_checkpointing, value=True))
            logger.warning(
                "You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it)."
                "Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model."
            )

        # >>> START AH Changes <<<
        try:
            self.enable_input_require_grads()
        except NotImplementedError:
            raise NotImplementedError(
                "Model has no enable_input_require_grads method implementation by Hugging Face. Parameter efficient fine-tuning however needs gradients for embeddings. This model therefore doesn't support gradient checkpointing with Adapters nor Hugging Face's PEFT library."
            )
        # >>> END AH Changes <<<


@inherit_doc
class ModelBaseAdaptersMixin(ModelAdaptersMixin):
    adapter_interface: AdapterModelInterface = None
    add_base_adapters = True

    def init_adapters(self, model_config, adapters_config):
        super().init_adapters(model_config, adapters_config)

        patch_forward(self)

    # Adapter Interface Methods

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(
            multigetattr(self, self.adapter_interface.model_layers)
        ):
            yield i, layer

    def get_layer(self, idx: int) -> nn.Module:
        return multigetattr(self, self.adapter_interface.model_layers)[idx]

    def iter_attentions(
        self,
    ) -> Iterable[Tuple[int, Literal["self", "cross"], nn.Module]]:
        for i, layer in self.iter_layers():
            if multihasattr(layer, self.adapter_interface.layer_self_attn or ""):
                yield i, "self", multigetattr(
                    layer, self.adapter_interface.layer_self_attn
                )
            if multihasattr(layer, self.adapter_interface.layer_cross_attn or ""):
                yield i, "cross", multigetattr(
                    layer, self.adapter_interface.layer_cross_attn
                )

    def iter_layer_ffns(
        self,
    ) -> Iterable[Tuple[int, Literal["intermediate", "output"], nn.Module]]:
        for i, layer in self.iter_layers():
            if intermediate_proj := multigetattr(
                layer, self.adapter_interface.layer_intermediate_proj
            ):
                yield i, "intermediate", intermediate_proj
            if output_proj := multigetattr(
                layer, self.adapter_interface.layer_output_proj
            ):
                yield i, "output", output_proj

    def post_embedding_forward(self, module, args, embedding_output):
        if isinstance(self, InvertibleAdaptersMixin) or isinstance(
            self, InvertibleAdaptersWrapperMixin
        ):
            embedding_output = self.invertible_adapters_forward(embedding_output)

        return embedding_output

    @ForwardContext.wrap_base
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class BertLayerAdaptersMixin:
    """Adds adapters to the BertLayer module."""

    def init_adapters(self, model_config, adapters_config):
        pass


class BertModelAdaptersMixin(
    EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin
):
    """Adds adapters to the BertModel module."""

    def init_adapters(self, model_config, adapters_config):
        super().init_adapters(model_config, adapters_config)

        for _, layer in self.iter_layers():
            self._set_layer_hook_for_parallel(layer)

        self.embeddings.register_forward_hook(self.post_embedding_forward)

    def _set_layer_hook_for_parallel(self, layer: nn.Module):
        def hook(module, input):
            adjust_tensors_for_parallel_(input[0], input[1])
            return input

        layer.register_forward_pre_hook(hook)

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.encoder.layer):
            yield i, layer


# XLM-RoBERTa adapter classes
class XLMRobertaSelfAttentionWithAdapters(
    BertSelfAttentionAdaptersMixin, XLMRobertaSelfAttention
):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        attention_mask = prefix_attention_mask(attention_mask)

        mixed_query_layer = self.query(hidden_states)

        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        query_layer, key_layer, value_layer = match_attn_matrices_for_parallel(
            query_layer, key_layer, value_layer
        )
        (attention_mask,) = adjust_tensors_for_parallel(query_layer, attention_mask)

        use_cache = past_key_value is not None
        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        key_layer, value_layer, attention_mask = self.prefix_tuning(
            key_layer, value_layer, hidden_states, attention_mask
        )
        (query_layer,) = adjust_tensors_for_parallel(key_layer, query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(
                    key_length - 1, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            else:
                position_ids_l = torch.arange(
                    query_length, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            position_ids_r = torch.arange(
                key_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class XLMRobertaSdpaSelfAttentionWithAdapters(
    BertSelfAttentionAdaptersMixin, XLMRobertaSdpaSelfAttention
):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        attention_mask = prefix_attention_mask(attention_mask, [2, 3])

        if (
            self.position_embedding_type != "absolute"
            or output_attentions
            or head_mask is not None
        ):
            return super().forward(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

        bsz, tgt_len, _ = hidden_states.size()

        query_layer = self.transpose_for_scores(self.query(hidden_states))

        is_cross_attention = encoder_hidden_states is not None

        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        attention_mask = (
            encoder_attention_mask if is_cross_attention else attention_mask
        )

        if (
            is_cross_attention
            and past_key_value
            and past_key_value[0].shape[2] == current_states.shape[1]
        ):
            key_layer, value_layer = past_key_value
        else:
            key_layer = self.transpose_for_scores(self.key(current_states))
            value_layer = self.transpose_for_scores(self.value(current_states))
            if past_key_value is not None and not is_cross_attention:
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)

        query_layer, key_layer, value_layer = match_attn_matrices_for_parallel(
            query_layer, key_layer, value_layer
        )
        (attention_mask,) = adjust_tensors_for_parallel(query_layer, attention_mask)

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        key_layer, value_layer, attention_mask = self.prefix_tuning(
            key_layer, value_layer, hidden_states, attention_mask
        )
        (query_layer,) = adjust_tensors_for_parallel(key_layer, query_layer)
        bsz = query_layer.size(0)

        if (
            self.require_contiguous_qkv
            and query_layer.device.type == "cuda"
            and attention_mask is not None
        ):
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()

        is_causal = (
            True
            if self.is_decoder
            and not is_cross_attention
            and attention_mask is None
            and tgt_len > 1
            else False
        )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)

        outputs = (attn_output,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class XLMRobertaSelfOutputWithAdapters(
    BertSelfOutputAdaptersMixin, XLMRobertaSelfOutput
):
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.bottleneck_layer_forward(
            hidden_states, input_tensor, self.LayerNorm
        )
        return hidden_states


class XLMRobertaOutputWithAdapters(BertOutputAdaptersMixin, XLMRobertaOutput):
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.bottleneck_layer_forward(
            hidden_states, input_tensor, self.LayerNorm
        )
        return hidden_states


# MODEL_MIXIN_MAPPING must be defined after the classes are defined
MODEL_MIXIN_MAPPING = {
    "BertLayer": BertLayerAdaptersMixin,
    "BertModel": BertModelAdaptersMixin,
    "ElectraLayer": BertLayerAdaptersMixin,
    "ElectraModel": BertModelAdaptersMixin,
    "RobertaLayer": BertLayerAdaptersMixin,
    "RobertaModel": BertModelAdaptersMixin,
    "XLMRobertaLayer": BertLayerAdaptersMixin,
    "XLMRobertaModel": BertModelAdaptersMixin,
    "BertGenerationEncoder": BertModelAdaptersMixin,
    "BertGenerationLayer": BertLayerAdaptersMixin,
}


@inherit_doc
class Activation_Function_Class(nn.Module):
    """
    Implementation of various activation function.
    """

    def __init__(self, hidden_act):
        super().__init__()
        if hidden_act is None:
            self.f = nn.Identity()
        elif hidden_act.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu
        else:
            self.f = get_activation(hidden_act.lower())

    def forward(self, x):
        return self.f(x)


# Single Adapter


class Adapter(nn.Module):
    """
    Implementation of a sequential bottleneck adapter block.
    """

    def __init__(
        self,
        adapter_name,
        input_size,
        down_sample,
        config: BnConfig,
    ):
        super().__init__()
        self.name = adapter_name
        self.input_size = input_size
        self.add_layer_norm_before = config["ln_before"]
        self.add_layer_norm_after = config["ln_after"]
        self.adapter_residual_before_ln = config["adapter_residual_before_ln"]
        self.use_gating = config["use_gating"]

        # Params related to input & output of adapter
        self.residual_before_ln = config["residual_before_ln"]
        self.original_ln_before = config["original_ln_before"]
        self.original_ln_after = config["original_ln_after"]

        seq_list = []

        # If we want to have a layer norm on input, we add it to seq_list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)

        self.down_sample = down_sample
        if down_sample is None:
            self.down_sample = self.input_size // 2

        # ensure that the down sample size is at least 1
        if self.down_sample < 1:
            self.down_sample = 1

        if config["phm_layer"]:
            # Linear down projection of the input
            seq_list.append(
                PHMLayer(
                    adapter_name, self.input_size, self.down_sample, "down", config
                )
            )
        else:
            seq_list.append(nn.Linear(self.input_size, self.down_sample))

        # select non-linearity
        self.non_linearity = Activation_Function_Class(config["non_linearity"].lower())

        seq_list.append(self.non_linearity)

        # residual connection
        self.adapter_down = nn.Sequential(*seq_list)

        # Up projection to input size
        if config["phm_layer"]:
            # Linear down projection of the input
            self.adapter_up = PHMLayer(
                adapter_name, self.down_sample, self.input_size, "up", config
            )
        else:
            self.adapter_up = nn.Linear(self.down_sample, self.input_size)

        if isinstance(config["scaling"], float):
            self.scaling = config["scaling"]
        elif config["scaling"] == "learned":
            self.scaling = nn.Parameter(torch.ones(1))
        elif config["scaling"] == "channel":
            self.scaling = nn.Parameter(torch.ones(input_size))
        else:
            raise ValueError("Unknown scaling type: {}".format(config["scaling"]))

        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

        if self.use_gating:
            self.gate = nn.Linear(self.input_size, 1)

        self.dropout = nn.Dropout(p=config["dropout"])

        fix_seed(config.init_weights_seed)

        if config["init_weights"] == "bert":
            self.adapter_down.apply(self.init_bert_weights)
            self.adapter_up.apply(self.init_bert_weights)
            if self.use_gating:
                self.gate.apply(self.init_bert_weights)
        elif config["init_weights"] == "mam_adapter":
            with torch.no_grad():
                for layer in self.adapter_down:
                    if isinstance(layer, nn.Linear) or isinstance(layer, PHMLayer):
                        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                        nn.init.zeros_(layer.bias)
                nn.init.zeros_(self.adapter_up.weight)
                nn.init.zeros_(self.adapter_up.bias)
                if self.use_gating:
                    self.gate.apply(self.init_bert_weights)
        elif config["init_weights"] == "houlsby":
            for layer in self.adapter_down:
                if isinstance(layer, nn.Linear) or isinstance(layer, PHMLayer):
                    nn.init.trunc_normal_(
                        layer.weight, mean=0, std=1e-2, a=-2 * 1e-2, b=2 * 1e-2
                    )
                    nn.init.zeros_(layer.bias)

            nn.init.trunc_normal_(
                self.adapter_up.weight, mean=0, std=1e-2, a=-2 * 1e-2, b=2 * 1e-2
            )
            nn.init.zeros_(self.adapter_up.bias)
        else:
            raise ValueError(
                "Unknown init_weights type: {}".format(config["init_weights"])
            )

        if config["stochastic_depth"] > 0.0:
            if is_torchvision_available():
                from torchvision.ops.stochastic_depth import StochasticDepth

                self.DropPath = StochasticDepth(
                    p=config["stochastic_depth"], mode="row"
                )
            else:
                raise ImportError(
                    "stochastic_depth requires the package torchvision, but it is not installed"
                )

    def pre_forward(
        self,
        hidden_states,
        input_tensor,
        layer_norm,
        fusion_config=None,
    ):
        """
        Retrieves the hidden_states, query (for Fusion), and residual connection according to the set configuration.

        Args:
            adapter_config: config file according to what the parameters are passed
            hidden_states: output of previous layer
            input_tensor: residual connection before FFN

        Returns: hidden_states, query, residual

        """
        query = None

        if self.residual_before_ln is True:
            residual = hidden_states

        if fusion_config is not None and fusion_config["query_before_ln"]:
            query = hidden_states

        if self.original_ln_before:
            if layer_norm:
                hidden_states = hidden_states + input_tensor
                if self.residual_before_ln == "post_add":
                    residual = hidden_states
                hidden_states = layer_norm(hidden_states)
            else:
                # is applied only once at the end by combining:
                #     output = original_input + attention_output + ffn_output
                #
                # to the FFN. Therefore, this additional check is needed to prevent errors.
                if input_tensor is not None:
                    hidden_states = hidden_states + input_tensor

        if not self.residual_before_ln:
            residual = hidden_states

        if fusion_config is not None and not fusion_config["query_before_ln"]:
            query = hidden_states

        return hidden_states, query, residual

    def forward(self, x, residual_input, output_gating=False):
        down = self.adapter_down(x)

        up = self.adapter_up(down)
        if hasattr(self, "DropPath"):
            up = self.DropPath(up)
        up = up * self.scaling
        output = self.dropout(up)

        if self.use_gating:
            # x.shape = (batch_size, seq_len, hidden_size)
            gate = torch.sigmoid(self.gate(x))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            output = output * gate

        if self.adapter_residual_before_ln:
            output = output + residual_input

        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        if not self.adapter_residual_before_ln:
            output = output + residual_input

        if self.use_gating and output_gating:
            return output, down, up, gate
        return output, down, up

    def post_forward(
        self, hidden_states, input_hidden_states, input_tensor, layer_norm
    ):
        """
        Performs computations after the forward pass of the adapter block(s). This e.g. includes applying the residual
        connection and layer norm if configured in this way.

        Args:
            hidden_states: The hidden states outputted by the adapter block(s).
            input_hidden_states: Residual connection before the adapter block(s).
            input_tensor: Residual connection before the Transformer FFN/ attention layer.
            layer_norm: Transformer LayerNorm.

        Returns:
            The modified hidden states.
        """
        if self.original_ln_after:
            if layer_norm:
                hidden_states = layer_norm(hidden_states + input_tensor)
            else:
                hidden_states = hidden_states + input_tensor

        return hidden_states

    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class ParallelAdapter(Adapter):
    """
    Implementation of a parallel bottleneck adapter block.
    """

    def __init__(self, adapter_name, input_size, down_sample, config: BnConfig):
        super().__init__(adapter_name, input_size, down_sample, config)

    def pre_forward(
        self,
        hidden_states,
        input_tensor,
        layer_norm,
        fusion_config=None,
    ):
        """
        Retrieves the hidden_states, query (for Fusion), and residual connection according to the set configuration.

        Args:
            adapter_config: config file according to what the parameters are passed
            hidden_states: output of previous layer
            input_tensor: residual connection before FFN

        Returns: hidden_states, query, residual

        """
        query = None
        if fusion_config is not None:
            query = input_tensor
        return input_tensor, query, input_tensor

    def forward(self, x, residual_input, output_gating=False):
        down = self.adapter_down(x)

        up = self.adapter_up(down)
        up = up * self.scaling

        output = self.dropout(up)

        if self.use_gating:
            # x.shape = (batch_size, seq_len, hidden_size)
            gate = torch.sigmoid(self.gate(x))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            output = output * gate

        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        if self.use_gating and output_gating:
            return output, down, up, gate
        return output, down, up

    def post_forward(
        self, hidden_states, input_hidden_states, input_tensor, layer_norm
    ):
        """
        Performs computations after the forward pass of the adapter block(s). This e.g. includes applying the residual
        connection and layer norm if configured in this way.

        Args:
            hidden_states: The hidden states outputted by the adapter block(s).
            input_hidden_states: Residual connection before the adapter block(s).
            input_tensor: Residual connection before the Transformer FFN/ attention layer.
            layer_norm: Transformer LayerNorm.

        Returns:
            The modified hidden states.
        """
        hidden_states = hidden_states + input_hidden_states

        if self.original_ln_after:
            if layer_norm:
                hidden_states = layer_norm(hidden_states + input_tensor)
            else:
                hidden_states = hidden_states + input_tensor

        return hidden_states


# Adapter Fusion

# Invertible Adapters


def get_subnet_constructor(non_linearity, reduction_factor):
    def subnet(dims_in, dims_out):
        return nn.Sequential(
            nn.Linear(dims_in, int(dims_in // reduction_factor)),
            Activation_Function_Class(non_linearity),
            nn.Linear(int(dims_in // reduction_factor), dims_out),
        )

    return subnet


class NICECouplingBlock(nn.Module):
    """Coupling Block following the NICE design."""

    def __init__(self, dims_in, dims_c=[], non_linearity="relu", reduction_factor=2):
        super().__init__()

        channels = dims_in[0][0]
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        assert all(
            [dims_c[i][1:] == dims_in[0][1:] for i in range(len(dims_c))]
        ), "Dimensions of input and one or more conditions don't agree."
        self.conditional = len(dims_c) > 0
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        subnet_constructor = get_subnet_constructor(non_linearity, reduction_factor)
        self.F = subnet_constructor(self.split_len2 + condition_length, self.split_len1)
        self.G = subnet_constructor(self.split_len1 + condition_length, self.split_len2)

    def forward(self, x, c=[], rev=False):
        x1, x2 = (x[:, :, : self.split_len1], x[:, :, self.split_len1 :])
        if not rev:
            x2_c = torch.cat([x2, *c], 1) if self.conditional else x2
            y1 = x1 + self.F(x2_c)
            y1_c = torch.cat([y1, *c], 1) if self.conditional else y1
            y2 = x2 + self.G(y1_c)
        else:
            x1_c = torch.cat([x1, *c], 1) if self.conditional else x1
            y2 = x2 - self.G(x1_c)
            y2_c = torch.cat([y2, *c], 1) if self.conditional else y2
            y1 = x1 - self.F(y2_c)

        return torch.cat((y1, y2), -1)

    def jacobian(self, x, rev=False):
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class GLOWCouplingBlock(nn.Module):
    """
    Coupling Block following the GLOW design. The only difference to the RealNVP coupling blocks, is the fact that it
    uses a single subnetwork to jointly predict [s_i, t_i], instead of two separate subnetworks. This reduces
    computational cost and speeds up learning. clamp: Soft clamping for the multiplicative component. The amplification
    or attenuation of each input dimension can be at most ±exp(clamp).
    """

    def __init__(
        self, dims_in, dims_c=[], non_linearity="relu", reduction_factor=2, clamp=5.0
    ):
        super().__init__()

        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = math.exp(clamp)
        self.min_s = math.exp(-clamp)

        assert all(
            [tuple(dims_c[i][1:]) == tuple(dims_in[0][1:]) for i in range(len(dims_c))]
        ), f"Dimensions of input and one or more conditions don't agree: {dims_c} vs {dims_in}."
        self.conditional = len(dims_c) > 0
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        subnet_constructor = get_subnet_constructor(non_linearity, reduction_factor)
        self.s1 = subnet_constructor(
            self.split_len1 + condition_length, self.split_len2 * 2
        )
        self.s2 = subnet_constructor(
            self.split_len2 + condition_length, self.split_len1 * 2
        )

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)

    def forward(self, x, c=[], rev=False):
        x1, x2 = (x[:, :, : self.split_len1], x[:, :, self.split_len1 :])

        if not rev:
            s2, t2 = x1.clone(), x2.clone()
            y1 = self.e(s2) * x1 + t2

            r1 = self.s1(torch.cat([y1, *c], 1) if self.conditional else y1)
            s1, t1 = r1[:, : self.split_len2], r1[:, self.split_len2 :]
            y2 = self.e(s1) * x2 + t1
            self.last_jac = torch.sum(
                self.log_e(s1), dim=tuple(range(1, self.ndims + 1))
            ) + torch.sum(self.log_e(s2), dim=tuple(range(1, self.ndims + 1)))

        else:  # names of x and y are swapped!
            r1 = self.s1(torch.cat([x1, *c], 1) if self.conditional else x1)
            s1, t1 = r1[:, : self.split_len2], r1[:, self.split_len2 :]
            y2 = (x2 - t1) / self.e(s1)

            r2 = self.s2(torch.cat([y2, *c], 1) if self.conditional else y2)
            s2, t2 = r2[:, : self.split_len1], r2[:, self.split_len1 :]
            y1 = (x1 - t2) / self.e(s2)
            self.last_jac = -torch.sum(
                self.log_e(s1), dim=tuple(range(1, self.ndims + 1))
            ) - torch.sum(self.log_e(s2), dim=tuple(range(1, self.ndims + 1)))

        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, c=[], rev=False):
        return self.last_jac

    def output_dims(self, input_dims):
        return input_dims


def init_W(config, W_left=None, W_right=None, W=None):
    """
    Initialize the weights for the compacter module or the shared parameters
    """
    if config["factorized_phm_W"]:
        W_left = W_left
        W_right = W_right
    else:
        W = W
    if config["hypercomplex_nonlinearity"]:
        if config["factorized_phm_W"]:
            for i in range(config["phm_dim"]):
                W_left.data[i] = nn.init.xavier_normal_(W_left.data[i])
                W_right.data[i] = nn.init.xavier_normal_(W_right.data[i])
        else:
            for i in range(config["phm_dim"]):
                W.data[i] = nn.init.xavier_normal_(W.data[i])
    elif config["hypercomplex_nonlinearity"] == "glorot-uniform":
        if config["factorized_phm_W"]:
            for i in range(config["phm_dim"]):
                W_left.data[i] = nn.init.xavier_uniform_(W_left.data[i])
                W_right.data[i] = nn.init.xavier_uniform_(W_right.data[i])
        else:
            for i in range(config["phm_dim"]):
                W.data[i] = nn.init.xavier_uniform_(W.data[i])
    elif config["hypercomplex_nonlinearity"] == "normal":
        if config["factorized_phm_W"]:
            for i in range(config["phm_dim"]):
                W_left.data[i].normal_(mean=0, std=config["phm_init_range"])
                W_right.data[i].normal_(mean=0, std=config["phm_init_range"])
        else:
            for i in range(config["phm_dim"]):
                W.data[i].normal_(mean=0, std=config["phm_init_range"])
    else:
        raise ValueError


class InvertibleAdapterLayer(AdapterLayerBase, nn.ModuleDict):
    adapter_modules_name = "_modules"

    def __init__(self, model_config, adapters_config):
        super().__init__()
        self.location_key = "inv_adapter"
        self.model_config = model_config
        self.adapters_config = adapters_config

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        self.layer_idx = layer_idx
        embedding_size = getattr(
            self.model_config, "embedding_size", self.model_config.hidden_size
        )
        adapter_config = self.adapters_config.match(
            adapter_name,
            config_type=BnConfig,
            location_key="inv_adapter",
        )
        if adapter_config is not None and adapter_config["inv_adapter"]:
            if adapter_config["inv_adapter"] == "nice":
                inv_adap = NICECouplingBlock(
                    [[embedding_size]],
                    non_linearity=adapter_config["non_linearity"],
                    reduction_factor=adapter_config["inv_adapter_reduction_factor"],
                )
            elif adapter_config["inv_adapter"] == "glow":
                inv_adap = GLOWCouplingBlock(
                    [[embedding_size]],
                    non_linearity=adapter_config["non_linearity"],
                    reduction_factor=adapter_config["inv_adapter_reduction_factor"],
                )
            else:
                raise ValueError(
                    f"Invalid invertible adapter type '{adapter_config['inv_adapter']}'."
                )
            self[adapter_name] = inv_adap
            self[adapter_name].apply(Adapter.init_bert_weights)
            return True

        return False

    def get_invertible_adapter(self):
        adapter_setup = self.get_active_setup()
        if adapter_setup is not None and len(adapter_setup) > 0:
            first_adapter = adapter_setup.first()
            if first_adapter in self:
                return self[first_adapter]
        return None

    def forward(self, hidden_states: torch.Tensor, rev=False):
        adapter_setup = self.get_active_setup()
        if adapter_setup is not None and len(adapter_setup) > 0:
            first_adapter = adapter_setup.first()
            if first_adapter in self:
                hidden_states = self[first_adapter](hidden_states, rev=rev)
        return hidden_states


def hook_fn(model, module, args, embedding_output):
    embedding_output = model.invertible_adapters(embedding_output)
    return embedding_output


def inv_hook_fn(model, module, args):
    inv_output = model.invertible_adapters(args[0], rev=True)
    return (inv_output,) + args[1:]


def init_invertible_adapters(model):
    base_model = model.base_model
    if not hasattr(base_model, "invertible_adapters"):
        base_model.invertible_adapters = InvertibleAdapterLayer(
            base_model.config, base_model.adapters_config
        )

        embed_layer = multigetattr(
            base_model, base_model.adapter_interface.model_embeddings
        )
        embed_layer.register_forward_hook(partial(hook_fn, base_model))

        base_model.add_invertible_adapter = types.MethodType(
            lambda self, *args, **kwargs: self.invertible_adapters.add_adapter(
                *args, **kwargs
            ),
            base_model,
        )
        base_model.delete_invertible_adapter = types.MethodType(
            lambda self, *args, **kwargs: self.invertible_adapters.delete_adapter(
                *args, **kwargs
            ),
            base_model,
        )
        base_model.get_invertible_adapter = types.MethodType(
            lambda self: self.invertible_adapters.get_invertible_adapter(), base_model
        )
        base_model.invertible_adapters_forward = types.MethodType(
            lambda self, *args, **kwargs: self.invertible_adapters(*args, **kwargs),
            base_model,
        )

        # Register reverse forward pass
        if output_embedding := model.get_output_embeddings():
            output_embedding.register_forward_pre_hook(partial(inv_hook_fn, base_model))


def dequantize_bnb_weight(weight: torch.nn.Parameter, state=None):
    """
    Helper function to dequantize 4bit or 8bit bnb weights.

    If the weight is not a bnb quantized weight, it will be returned as is.
    """
    if not isinstance(weight, torch.nn.Parameter):
        raise TypeError(
            f"Input weight should be of type nn.Parameter, got {type(weight)} instead"
        )

    cls_name = weight.__class__.__name__
    if cls_name not in ("Params4bit", "Int8Params"):
        return weight

    import bitsandbytes as bnb

    if cls_name == "Params4bit":
        return bnb.functional.dequantize_4bit(weight.data, weight.quant_state)

    if state.SCB is None:
        state.SCB = weight.SCB

    im = torch.eye(weight.data.shape[-1]).contiguous().half().to(weight.device)
    im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
    im, Sim = bnb.functional.transform(im, "col32")
    if state.CxB is None:
        state.CxB, state.SB = bnb.functional.transform(
            weight.data, to_order=state.formatB
        )
    out32, Sout32 = bnb.functional.igemmlt(im, state.CxB, Sim, state.SB)
    return bnb.functional.mm_dequant(out32, Sout32, SCim, state.SCB, bias=None).t()


def multisetattr(o: object, name: str, value: object):
    parts = name.split(".")
    for n in parts[:-1]:
        if hasattr(o, n):
            o = getattr(o, n)
        else:
            return
    setattr(o, parts[-1], value)


def prefix_attention_mask(
    attention_mask, dim: Union[int, List[int]] = 3, prefix_value: int = 0
):
    """
    Adds a prefix to an attention mask. The length of the prefix is determined by the `prefix_attention_mask_length`
    attribute in the ForwardContext.

    Args:
        attention_mask:
            The attention mask to add the prefix to.
        dim (int):
            The dimension along which to concatenate the prefix_attention_mask. Defaults to 3.
        prefix_value (int):
            The value to use for the prefix_attention_mask. Defaults to 0, however some models, e.g. DistilBert, use
            different values. BERT like models invert their extended_attention_mask, hence they use 0 as value for not
            masked tokens. This inversion is usually done in the forward method of the model in 2 different ways:
            1) by calling self.invert_attention_mask, as BERT does 2) by doing the inversion manually, e.g. ALBERT
            does: `extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min`
    """

    forward_context = ForwardContext.get_context()

    if (
        attention_mask is not None
        and forward_context is not None
        and getattr(forward_context, "prompt_tokens_length", None) is not None
    ):
        if isinstance(dim, int):
            dim = [dim]
        for d in dim:
            ones_shape = list(attention_mask.shape)
            ones_shape[d] = forward_context.prompt_tokens_length

            prefix_attention_mask = torch.full(
                ones_shape,
                prefix_value,
                dtype=attention_mask.dtype,
            ).to(attention_mask.device)

            # Concatenate the prefix_attention_mask along the specified dimension
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=d)

    return attention_mask


METHOD_INIT_MAPPING = {
    "bottleneck": init_bottleneck,
    "invertible": init_invertible_adapters,
}


def multihasattr(o: object, name: str) -> bool:
    if not name:
        return False
    parts = name.split(".")
    for n in parts:
        if hasattr(o, n):
            o = getattr(o, n)
        else:
            return False
    return True
