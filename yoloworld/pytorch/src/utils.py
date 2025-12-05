# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import (
    Optional,
    Union,
    Any,
    List,
    Tuple,
    Generator,
    Iterator,
    Sequence,
    Mapping,
)
from time import time as ti
from torch.nn.modules.module import Module
import torch.distributed as dist
from torch.nn.parameter import Parameter
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from logging import FileHandler,Logger, LogRecord
from collections import defaultdict
import functools
from inspect import getfullargspec
from collections import abc
from shutil import get_terminal_size
from packaging.version import parse
import math
import torch.nn.functional as F
from numbers import Number
from torch import Tensor, BoolTensor
from scipy.stats import truncnorm
import shapely.geometry as geometry
import numbers
import cv2
import itertools
from pathlib import Path
import torch.nn as nn
import warnings
import copy
import torch
import datetime
from addict import Dict
import importlib
import pycocotools.mask as maskUtils
from importlib import import_module
import tempfile
import os
import ast
import os.path as osp
import re
from collections import OrderedDict
import inspect
from typing import Type, TypeVar
import threading
import logging
from torch.optim import Optimizer
from torch.distributed import ProcessGroup
from torch import distributed as torch_dist
import sys
from termcolor import colored
from getpass import getuser
from socket import gethostname
import uuid
import platform
import shutil
from abc import ABCMeta, abstractmethod, abstractproperty, abstractstaticmethod
import json
import numpy as np
import yaml
from collections.abc import Callable, Sized
from rich.console import Console
from rich.table import Table
import types
from argparse import Action, ArgumentParser, Namespace

from yaml import CDumper as Dumper  # type: ignore
from yaml import CLoader as Loader  # type: ignore
import pickle
from contextlib import contextmanager
from urllib.request import urlopen
from io import BytesIO, StringIO
from yapf.yapflib.yapf_api import FormatCode
import time
try:
    from PIL import Image
except ImportError:
    Image = None
backends: dict = {}
prefix_to_backends: dict = {}
BASE_KEY = "_base_"
RESERVED_KEYS = ["filename", "text", "pretty_text", "env_variables"]
DEPRECATION_KEY = "_deprecation_"
DELETE_KEY = "_delete_"
_lock = threading.RLock()
T = TypeVar("T")
TORCH_VERSION = torch.__version__
def _accquire_lock() -> None:
    """Acquire the module-level lock for serializing access to shared data.

    This should be released with _release_lock().
    """
    if _lock:
        _lock.acquire()


def _release_lock() -> None:
    """Release the module-level lock acquired by calling _accquire_lock()."""
    if _lock:
        _lock.release()


class ManagerMeta(type):
    """The metaclass for global accessible class.

    The subclasses inheriting from ``ManagerMeta`` will manage their
    own ``_instance_dict`` and root instances. The constructors of subclasses
    must contain the ``name`` argument.

    Examples:
        >>> class SubClass1(metaclass=ManagerMeta):
        >>>     def __init__(self, *args, **kwargs):
        >>>         pass
        AssertionError: <class '__main__.SubClass1'>.__init__ must have the
        name argument.
        >>> class SubClass2(metaclass=ManagerMeta):
        >>>     def __init__(self, name):
        >>>         pass
        >>> # valid format.
    """

    def __init__(cls, *args):
        cls._instance_dict = OrderedDict()
        params = inspect.getfullargspec(cls)
        params_names = params[0] if params[0] else []
        assert "name" in params_names, f"{cls} must have the `name` argument"
        super().__init__(*args)


class ManagerMixin(metaclass=ManagerMeta):
    """``ManagerMixin`` is the base class for classes that have global access
    requirements.

    The subclasses inheriting from ``ManagerMixin`` can get their
    global instances.

    Examples:
        >>> class GlobalAccessible(ManagerMixin):
        >>>     def __init__(self, name=''):
        >>>         super().__init__(name)
        >>>
        >>> GlobalAccessible.get_instance('name')
        >>> instance_1 = GlobalAccessible.get_instance('name')
        >>> instance_2 = GlobalAccessible.get_instance('name')
        >>> assert id(instance_1) == id(instance_2)

    Args:
        name (str): Name of the instance. Defaults to ''.
    """

    def __init__(self, name: str = "", **kwargs):
        assert (
            isinstance(name, str) and name
        ), "name argument must be an non-empty string."
        self._instance_name = name

    @classmethod
    def get_instance(cls: Type[T], name: str, **kwargs) -> T:
        """Get subclass instance by name if the name exists.

        If corresponding name instance has not been created, ``get_instance``
        will create an instance, otherwise ``get_instance`` will return the
        corresponding instance.

        Examples
            >>> instance1 = GlobalAccessible.get_instance('name1')
            >>> # Create name1 instance.
            >>> instance.instance_name
            name1
            >>> instance2 = GlobalAccessible.get_instance('name1')
            >>> # Get name1 instance.
            >>> assert id(instance1) == id(instance2)

        Args:
            name (str): Name of instance. Defaults to ''.

        Returns:
            object: Corresponding name instance, the latest instance, or root
            instance.
        """
        _accquire_lock()
        assert isinstance(name, str), f"type of name should be str, but got {type(cls)}"
        instance_dict = cls._instance_dict  # type: ignore
        # Get the instance by name.
        if name not in instance_dict:
            instance = cls(name=name, **kwargs)  # type: ignore
            instance_dict[name] = instance  # type: ignore
        elif kwargs:
            warnings.warn(
                f"{cls} instance named of {name} has been created, "
                "the method `get_instance` should not accept any other "
                "arguments"
            )
        # Get latest instantiated instance or root instance.
        _release_lock()
        return instance_dict[name]

    @classmethod
    def get_current_instance(cls):
        """Get latest created instance.

        Before calling ``get_current_instance``, The subclass must have called
        ``get_instance(xxx)`` at least once.

        Examples
            >>> instance = GlobalAccessible.get_current_instance()
            AssertionError: At least one of name and current needs to be set
            >>> instance = GlobalAccessible.get_instance('name1')
            >>> instance.instance_name
            name1
            >>> instance = GlobalAccessible.get_current_instance()
            >>> instance.instance_name
            name1

        Returns:
            object: Latest created instance.
        """
        _accquire_lock()
        if not cls._instance_dict:
            raise RuntimeError(
                f"Before calling {cls.__name__}.get_current_instance(), you "
                "should call get_instance(name=xxx) at least once."
            )
        name = next(iter(reversed(cls._instance_dict)))
        _release_lock()
        return cls._instance_dict[name]

    @classmethod
    def check_instance_created(cls, name: str) -> bool:
        """Check whether the name corresponding instance exists.

        Args:
            name (str): Name of instance.

        Returns:
            bool: Whether the name corresponding instance exists.
        """
        return name in cls._instance_dict

    @property
    def instance_name(self) -> str:
        """Get the name of instance.

        Returns:
            str: Name of instance.
        """
        return self._instance_name


def is_distributed() -> bool:
    """Return True if distributed environment has been initialized."""
    return torch_dist.is_available() and torch_dist.is_initialized()


def get_default_group() -> Optional[ProcessGroup]:
    """Return default process group."""

    return torch_dist.distributed_c10d._get_default_group()

def get_rank(group: Optional[ProcessGroup] = None) -> int:
    """Return the rank of the given process group.

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Note:
        Calling ``get_rank`` in non-distributed environment will return 0.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Return the rank of the process group if in distributed
        environment, otherwise 0.
    """

    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_rank(group)
    else:
        return 0


def _get_rank():
    """Support using logging module without torch."""
    return get_rank()


def _get_device_id():
    """Get device id of current machine."""
    try:
        import torch
    except ImportError:
        return 0
    else:
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        # TODO: return device id of npu and mlu.
        if not torch.cuda.is_available():
            return local_rank
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices is None:
            num_device = torch.cuda.device_count()
            cuda_visible_devices = list(range(num_device))
        else:
            cuda_visible_devices = cuda_visible_devices.split(",")
        return int(cuda_visible_devices[local_rank])


class MMFormatter(logging.Formatter):
    """Colorful format for MMLogger. If the log level is error, the logger will
    additionally output the location of the code.

    Args:
        color (bool): Whether to use colorful format. filehandler is not
            allowed to use color format, otherwise it will be garbled.
        blink (bool): Whether to blink the ``INFO`` and ``DEBUG`` logging
            level.
        **kwargs: Keyword arguments passed to
            :meth:`logging.Formatter.__init__`.
    """

    _color_mapping: dict = dict(
        ERROR="red", WARNING="yellow", INFO="white", DEBUG="green"
    )

    def __init__(self, color: bool = True, blink: bool = False, **kwargs):
        super().__init__(**kwargs)
        assert not (
            not color and blink
        ), "blink should only be available when color is True"
        # Get prefix format according to color.
        error_prefix = self._get_prefix("ERROR", color, blink=True)
        warn_prefix = self._get_prefix("WARNING", color, blink=True)
        info_prefix = self._get_prefix("INFO", color, blink)
        debug_prefix = self._get_prefix("DEBUG", color, blink)

        # Config output format.
        self.err_format = (
            f"%(asctime)s - %(name)s - {error_prefix} - "
            "%(pathname)s - %(funcName)s - %(lineno)d - "
            "%(message)s"
        )
        self.warn_format = f"%(asctime)s - %(name)s - {warn_prefix} - %(" "message)s"
        self.info_format = f"%(asctime)s - %(name)s - {info_prefix} - %(" "message)s"
        self.debug_format = f"%(asctime)s - %(name)s - {debug_prefix} - %(" "message)s"

    def _get_prefix(self, level: str, color: bool, blink=False) -> str:
        """Get the prefix of the target log level.

        Args:
            level (str): log level.
            color (bool): Whether to get colorful prefix.
            blink (bool): Whether the prefix will blink.

        Returns:
            str: The plain or colorful prefix.
        """
        if color:
            attrs = ["underline"]
            if blink:
                attrs.append("blink")
            prefix = colored(level, self._color_mapping[level], attrs=attrs)
        else:
            prefix = level
        return prefix

    def format(self, record: LogRecord) -> str:
        """Override the `logging.Formatter.format`` method `. Output the
        message according to the specified log level.

        Args:
            record (LogRecord): A LogRecord instance represents an event being
                logged.

        Returns:
            str: Formatted result.
        """
        if record.levelno == logging.ERROR:
            self._style._fmt = self.err_format
        elif record.levelno == logging.WARNING:
            self._style._fmt = self.warn_format
        elif record.levelno == logging.INFO:
            self._style._fmt = self.info_format
        elif record.levelno == logging.DEBUG:
            self._style._fmt = self.debug_format

        result = logging.Formatter.format(self, record)
        return result


class FilterDuplicateWarning(logging.Filter):
    """Filter the repeated warning message.

    Args:
        name (str): name of the filter.
    """

    def __init__(self, name: str = "mmengine"):
        super().__init__(name)
        self.seen: set = set()

    def filter(self, record: LogRecord) -> bool:
        """Filter the repeated warning message.

        Args:
            record (LogRecord): The log record.

        Returns:
            bool: Whether to output the log record.
        """
        if record.levelno != logging.WARNING:
            return True

        if record.msg not in self.seen:
            self.seen.add(record.msg)
            return True
        return False


def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    """Return the number of the given process group.

    Note:
        Calling ``get_world_size`` in non-distributed environment will return
        1.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Return the number of processes of the given process group if in
        distributed environment, otherwise 1.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_world_size(group)
    else:
        return 1


def _get_world_size():
    """Support using logging module without torch."""
    return get_world_size()


def _get_host_info() -> str:
    """Get hostname and username.

    Return empty string if exception raised, e.g. ``getpass.getuser()`` will
    lead to error in docker container
    """
    host = ""
    try:
        host = f"{getuser()}@{gethostname()}"
    except Exception as e:
        warnings.warn(f"Host or user not found: {str(e)}")
    finally:
        return host


class MMLogger(Logger, ManagerMixin):
    """Formatted logger used to record messages.

    ``MMLogger`` can create formatted logger to log message with different
    log levels and get instance in the same way as ``ManagerMixin``.
    ``MMLogger`` has the following features:

    - Distributed log storage, ``MMLogger`` can choose whether to save log of
      different ranks according to `log_file`.
    - Message with different log levels will have different colors and format
      when displayed on terminal.

    Note:
        - The `name` of logger and the ``instance_name`` of ``MMLogger`` could
          be different. We can only get ``MMLogger`` instance by
          ``MMLogger.get_instance`` but not ``logging.getLogger``. This feature
          ensures ``MMLogger`` will not be incluenced by third-party logging
          config.
        - Different from ``logging.Logger``, ``MMLogger`` will not log warning
          or error message without ``Handler``.

    Examples:
        >>> logger = MMLogger.get_instance(name='MMLogger',
        >>>                                logger_name='Logger')
        >>> # Although logger has name attribute just like `logging.Logger`
        >>> # We cannot get logger instance by `logging.getLogger`.
        >>> assert logger.name == 'Logger'
        >>> assert logger.instance_name = 'MMLogger'
        >>> assert id(logger) != id(logging.getLogger('Logger'))
        >>> # Get logger that do not store logs.
        >>> logger1 = MMLogger.get_instance('logger1')
        >>> # Get logger only save rank0 logs.
        >>> logger2 = MMLogger.get_instance('logger2', log_file='out.log')
        >>> # Get logger only save multiple ranks logs.
        >>> logger3 = MMLogger.get_instance('logger3', log_file='out.log',
        >>>                                 distributed=True)

    Args:
        name (str): Global instance name.
        logger_name (str): ``name`` attribute of ``Logging.Logger`` instance.
            If `logger_name` is not defined, defaults to 'mmengine'.
        log_file (str, optional): The log filename. If specified, a
            ``FileHandler`` will be added to the logger. Defaults to None.
        log_level (str): The log level of the handler. Defaults to
            'INFO'. If log level is 'DEBUG', distributed logs will be saved
            during distributed training.
        file_mode (str): The file mode used to open log file. Defaults to 'w'.
        distributed (bool): Whether to save distributed logs, Defaults to
            false.
    """

    def __init__(
        self,
        name: str,
        logger_name="mmengine",
        log_file: Optional[str] = None,
        log_level: Union[int, str] = "INFO",
        file_mode: str = "w",
        distributed=False,
    ):
        Logger.__init__(self, logger_name)
        ManagerMixin.__init__(self, name)
        # Get rank in DDP mode.
        if isinstance(log_level, str):
            log_level = logging._nameToLevel[log_level]
        global_rank = _get_rank()
        device_id = _get_device_id()

        # Config stream_handler. If `rank != 0`. stream_handler can only
        # export ERROR logs.
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        # `StreamHandler` record month, day, hour, minute, and second
        # timestamp.
        stream_handler.setFormatter(MMFormatter(color=True, datefmt="%m/%d %H:%M:%S"))
        # Only rank0 `StreamHandler` will log messages below error level.
        if global_rank == 0:
            stream_handler.setLevel(log_level)
        else:
            stream_handler.setLevel(logging.ERROR)
        stream_handler.addFilter(FilterDuplicateWarning(logger_name))
        self.handlers.append(stream_handler)

        if log_file is not None:
            world_size = _get_world_size()
            is_distributed = (
                log_level <= logging.DEBUG or distributed
            ) and world_size > 1
            if is_distributed:
                filename, suffix = osp.splitext(osp.basename(log_file))
                hostname = _get_host_info()
                if hostname:
                    filename = (
                        f"{filename}_{hostname}_device{device_id}_"
                        f"rank{global_rank}{suffix}"
                    )
                else:
                    # Omit hostname if it is empty
                    filename = (
                        f"{filename}_device{device_id}_" f"rank{global_rank}{suffix}"
                    )
                log_file = osp.join(osp.dirname(log_file), filename)
            # Save multi-ranks logs if distributed is True. The logs of rank0
            # will always be saved.
            if global_rank == 0 or is_distributed:
                # Here, the default behaviour of the official logger is 'a'.
                # Thus, we provide an interface to change the file mode to
                # the default behaviour. `FileHandler` is not supported to
                # have colors, otherwise it will appear garbled.
                file_handler = logging.FileHandler(log_file, file_mode)
                # `StreamHandler` record year, month, day hour, minute,
                # and second timestamp. file_handler will only record logs
                # without color to avoid garbled code saved in files.
                file_handler.setFormatter(
                    MMFormatter(color=False, datefmt="%Y/%m/%d %H:%M:%S")
                )
                file_handler.setLevel(log_level)
                file_handler.addFilter(FilterDuplicateWarning(logger_name))
                self.handlers.append(file_handler)
        self._log_file = log_file

    @property
    def log_file(self):
        return self._log_file

    @classmethod
    def get_current_instance(cls) -> "MMLogger":
        """Get latest created ``MMLogger`` instance.

        :obj:`MMLogger` can call :meth:`get_current_instance` before any
        instance has been created, and return a logger with the instance name
        "mmengine".

        Returns:
            MMLogger: Configured logger instance.
        """
        if not cls._instance_dict:
            cls.get_instance("mmengine")
        return super().get_current_instance()

    def callHandlers(self, record: LogRecord) -> None:
        """Pass a record to all relevant handlers.

        Override ``callHandlers`` method in ``logging.Logger`` to avoid
        multiple warning messages in DDP mode. Loop through all handlers of
        the logger instance and its parents in the logger hierarchy. If no
        handler was found, the record will not be output.

        Args:
            record (LogRecord): A ``LogRecord`` instance contains logged
                message.
        """
        for handler in self.handlers:
            if record.levelno >= handler.level:
                handler.handle(record)

    def setLevel(self, level):
        """Set the logging level of this logger.

        If ``logging.Logger.selLevel`` is called, all ``logging.Logger``
        instances managed by ``logging.Manager`` will clear the cache. Since
        ``MMLogger`` is not managed by ``logging.Manager`` anymore,
        ``MMLogger`` should override this method to clear caches of all
        ``MMLogger`` instance which is managed by :obj:`ManagerMixin`.

        level must be an int or a str.
        """
        self.level = logging._checkLevel(level)
        _accquire_lock()
        # The same logic as `logging.Manager._clear_cache`.
        for logger in MMLogger._instance_dict.values():
            logger._cache.clear()
        _release_lock()


def print_log(
    msg, logger: Optional[Union[Logger, str]] = None, level=logging.INFO
) -> None:
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (Logger or str, optional): If the type of logger is
        ``logging.Logger``, we directly use logger to log messages.
            Some special loggers are:

            - "silent": No message will be printed.
            - "current": Use latest created logger to log message.
            - other str: Instance name of logger. The corresponding logger
              will log message if it has been created, otherwise ``print_log``
              will raise a `ValueError`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object, "current", or a created logger instance name.
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == "silent":
        pass
    elif logger == "current":
        logger_instance = MMLogger.get_current_instance()
        logger_instance.log(level, msg)
    elif isinstance(logger, str):
        # If the type of `logger` is `str`, but not with value of `current` or
        # `silent`, we assume it indicates the name of the logger. If the
        # corresponding logger has not been created, `print_log` will raise
        # a `ValueError`.
        if MMLogger.check_instance_created(logger):
            logger_instance = MMLogger.get_instance(logger)
            logger_instance.log(level, msg)
        else:
            raise ValueError(f"MMLogger: {logger} has not been created!")
    else:
        raise TypeError(
            "`logger` should be either a logging.Logger object, str, "
            f'"silent", "current" or None, but got {type(logger)}'
        )


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Defaults to False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(f"custom_imports must be a list but got type {type(imports)}")
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported.")
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f"{imp} failed to import and is ignored.", UserWarning)
                imported_tmp = None
            else:
                raise ImportError(f"Failed to import {imp}")
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


class ConfigDict(Dict):
    """A dictionary for config which has the same interface as python's built-
    in dictionary and can be used as a normal dictionary.

    The Config class would transform the nested fields (dictionary-like fields)
    in config file into ``ConfigDict``.
    """

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no " f"attribute '{name}'"
            )
        except Exception as e:
            raise e
        else:
            return value


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


class RemoveAssignFromAST(ast.NodeTransformer):
    """Remove Assign node if the target's name match the key.

    Args:
        key (str): The target name of the Assign node.
    """

    def __init__(self, key):
        self.key = key

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Name) and node.targets[0].id == self.key:
            return None
        else:
            return node


class BaseFileHandler(metaclass=ABCMeta):
    # `str_like` is a flag to indicate whether the type of file object is
    # str-like object or bytes-like object. Pickle only processes bytes-like
    # objects but json only processes str-like object. If it is str-like
    # object, `StringIO` will be used to process the buffer.
    str_like = True

    @abstractmethod
    def load_from_fileobj(self, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_fileobj(self, obj, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_str(self, obj, **kwargs):
        pass

    def load_from_path(self, filepath, mode="r", **kwargs):
        with open(filepath, mode) as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_path(self, obj, filepath, mode="w", **kwargs):
        with open(filepath, mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)


def set_default(obj):
    """Set default json values for non-serializable values.

    It helps convert ``set``, ``range`` and ``np.ndarray`` data types to list.
    It also converts ``np.generic`` (including ``np.int32``, ``np.float32``,
    etc.) into plain numbers of plain python built-in types.
    """
    if isinstance(obj, (set, range)):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"{type(obj)} is unsupported for json dump")


class YamlHandler(BaseFileHandler):

    def load_from_fileobj(self, file, **kwargs):
        kwargs.setdefault("Loader", Loader)
        return yaml.load(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault("Dumper", Dumper)
        yaml.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault("Dumper", Dumper)
        return yaml.dump(obj, **kwargs)


class JsonHandler(BaseFileHandler):

    def load_from_fileobj(self, file):
        return json.load(file)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault("default", set_default)
        json.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault("default", set_default)
        return json.dumps(obj, **kwargs)


class PickleHandler(BaseFileHandler):

    str_like = False

    def load_from_fileobj(self, file, **kwargs):
        return pickle.load(file, **kwargs)

    def load_from_path(self, filepath, **kwargs):
        return super().load_from_path(filepath, mode="rb", **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault("protocol", 2)
        return pickle.dumps(obj, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault("protocol", 2)
        pickle.dump(obj, file, **kwargs)

    def dump_to_path(self, obj, filepath, **kwargs):
        super().dump_to_path(obj, filepath, mode="wb", **kwargs)


file_handlers = {
    'json': JsonHandler(),
    'yaml': YamlHandler(),
    'yml': YamlHandler(),
    'pickle': PickleHandler(),
    'pkl': PickleHandler(),
}


def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends.

    All backends need to implement two apis: :meth:`get()` and
    :meth:`get_text()`.

    - :meth:`get()` reads the file as a byte stream.
    - :meth:`get_text()` reads the file as texts.
    """

    # a flag to indicate whether the backend can create a symlink for a file
    # This attribute will be deprecated in future.
    _allow_symlink = False

    @property
    def allow_symlink(self):
        print_log(
            "allow_symlink will be deprecated in future",
            logger="current",
            level=logging.WARNING,
        )
        return self._allow_symlink

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass


class MemcachedBackend(BaseStorageBackend):
    """Memcached storage backend.

    Attributes:
        server_list_cfg (str): Config file for memcached server list.
        client_cfg (str): Config file for memcached client.
        sys_path (str, optional): Additional path to be appended to `sys.path`.
            Defaults to None.
    """

    def __init__(self, server_list_cfg, client_cfg, sys_path=None):
        if sys_path is not None:
            import sys

            sys.path.append(sys_path)
        try:
            import mc
        except ImportError:
            raise ImportError("Please install memcached to enable MemcachedBackend.")

        self.server_list_cfg = server_list_cfg
        self.client_cfg = client_cfg
        self._client = mc.MemcachedClient.GetInstance(
            self.server_list_cfg, self.client_cfg
        )
        # mc.pyvector servers as a point which points to a memory cache
        self._mc_buffer = mc.pyvector()

    def get(self, filepath: Union[str, Path]):
        """Get values according to the filepath.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Expected bytes object.

        Examples:
            >>> server_list_cfg = '/path/of/server_list.conf'
            >>> client_cfg = '/path/of/mc.conf'
            >>> backend = MemcachedBackend(server_list_cfg, client_cfg)
            >>> backend.get('/path/of/file')
            b'hello world'
        """
        filepath = str(filepath)
        import mc

        self._client.Get(filepath, self._mc_buffer)
        value_buf = mc.ConvertBuffer(self._mc_buffer)
        return value_buf

    def get_text(self, filepath, encoding=None):
        raise NotImplementedError


class LocalBackend(BaseStorageBackend):
    """Raw local storage backend."""

    _allow_symlink = True

    def get(self, filepath: Union[str, Path]) -> bytes:
        """Read bytes from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Expected bytes object.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.get(filepath)
            b'hello world'
        """
        with open(filepath, "rb") as f:
            value = f.read()
        return value

    def get_text(self, filepath: Union[str, Path], encoding: str = "utf-8") -> str:
        """Read text from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Defaults to 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.get_text(filepath)
            'hello world'
        """
        with open(filepath, encoding=encoding) as f:
            text = f.read()
        return text

    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        """Write bytes to a given ``filepath`` with 'wb' mode.

        Note:
            ``put`` will create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (bytes): Data to be written.
            filepath (str or Path): Path to write data.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.put(b'hello world', filepath)
        """
        mmengine.mkdir_or_exist(osp.dirname(filepath))
        with open(filepath, "wb") as f:
            f.write(obj)

    def put_text(
        self, obj: str, filepath: Union[str, Path], encoding: str = "utf-8"
    ) -> None:
        """Write text to a given ``filepath`` with 'w' mode.

        Note:
            ``put_text`` will create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to open the ``filepath``.
                Defaults to 'utf-8'.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.put_text('hello world', filepath)
        """
        mmengine.mkdir_or_exist(osp.dirname(filepath))
        with open(filepath, "w", encoding=encoding) as f:
            f.write(obj)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.

        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.exists(filepath)
            True
        """
        return osp.exists(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
            ``False`` otherwise.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/dir'
            >>> backend.isdir(filepath)
            True
        """
        return osp.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
            otherwise.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.isfile(filepath)
            True
        """
        return osp.isfile(filepath)

    def join_path(
        self, filepath: Union[str, Path], *filepaths: Union[str, Path]
    ) -> str:
        r"""Concatenate all file paths.

        Join one or more filepath components intelligently. The return value
        is the concatenation of filepath and any members of \*filepaths.

        Args:
            filepath (str or Path): Path to be concatenated.

        Returns:
            str: The result of concatenation.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath1 = '/path/of/dir1'
            >>> filepath2 = 'dir2'
            >>> filepath3 = 'path/of/file'
            >>> backend.join_path(filepath1, filepath2, filepath3)
            '/path/of/dir/dir2/path/of/file'
        """
        # TODO, if filepath or filepaths are Path, should return Path
        return osp.join(filepath, *filepaths)

    @contextmanager
    def get_local_path(
        self,
        filepath: Union[str, Path],
    ) -> Generator[Union[str, Path], None, None]:
        """Only for unified API and do nothing.

        Args:
            filepath (str or Path): Path to be read data.
            backend_args (dict, optional): Arguments to instantiate the
                corresponding backend. Defaults to None.

        Examples:
            >>> backend = LocalBackend()
            >>> with backend.get_local_path('s3://bucket/abc.jpg') as path:
            ...     # do something here
        """
        yield filepath

    def copyfile(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Copy a file src to dst and return the destination file.

        src and dst should have the same prefix. If dst specifies a directory,
        the file will be copied into dst using the base filename from src. If
        dst specifies a file that already exists, it will be replaced.

        Args:
            src (str or Path): A file to be copied.
            dst (str or Path): Copy file to dst.

        Returns:
            str: The destination file.

        Raises:
            SameFileError: If src and dst are the same file, a SameFileError
                will be raised.

        Examples:
            >>> backend = LocalBackend()
            >>> # dst is a file
            >>> src = '/path/of/file'
            >>> dst = '/path1/of/file1'
            >>> # src will be copied to '/path1/of/file1'
            >>> backend.copyfile(src, dst)
            '/path1/of/file1'

            >>> # dst is a directory
            >>> dst = '/path1/of/dir'
            >>> # src will be copied to '/path1/of/dir/file'
            >>> backend.copyfile(src, dst)
            '/path1/of/dir/file'
        """
        return shutil.copy(src, dst)

    def copytree(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Recursively copy an entire directory tree rooted at src to a
        directory named dst and return the destination directory.

        src and dst should have the same prefix and dst must not already exist.

        TODO: Whether to support dirs_exist_ok parameter.

        Args:
            src (str or Path): A directory to be copied.
            dst (str or Path): Copy directory to dst.

        Returns:
            str: The destination directory.

        Raises:
            FileExistsError: If dst had already existed, a FileExistsError will
                be raised.

        Examples:
            >>> backend = LocalBackend()
            >>> src = '/path/of/dir1'
            >>> dst = '/path/of/dir2'
            >>> backend.copytree(src, dst)
            '/path/of/dir2'
        """
        return shutil.copytree(src, dst)

    def copyfile_from_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Copy a local file src to dst and return the destination file. Same
        as :meth:`copyfile`.

        Args:
            src (str or Path): A local file to be copied.
            dst (str or Path): Copy file to dst.

        Returns:
            str: If dst specifies a directory, the file will be copied into dst
            using the base filename from src.

        Raises:
            SameFileError: If src and dst are the same file, a SameFileError
                will be raised.

        Examples:
            >>> backend = LocalBackend()
            >>> # dst is a file
            >>> src = '/path/of/file'
            >>> dst = '/path1/of/file1'
            >>> # src will be copied to '/path1/of/file1'
            >>> backend.copyfile_from_local(src, dst)
            '/path1/of/file1'

            >>> # dst is a directory
            >>> dst = '/path1/of/dir'
            >>> # src will be copied to
            >>> backend.copyfile_from_local(src, dst)
            '/path1/of/dir/file'
        """
        return self.copyfile(src, dst)

    def copytree_from_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Recursively copy an entire directory tree rooted at src to a
        directory named dst and return the destination directory. Same as
        :meth:`copytree`.

        Args:
            src (str or Path): A local directory to be copied.
            dst (str or Path): Copy directory to dst.

        Returns:
            str: The destination directory.

        Examples:
            >>> backend = LocalBackend()
            >>> src = '/path/of/dir1'
            >>> dst = '/path/of/dir2'
            >>> backend.copytree_from_local(src, dst)
            '/path/of/dir2'
        """
        return self.copytree(src, dst)

    def copyfile_to_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Copy the file src to local dst and return the destination file. Same
        as :meth:`copyfile`.

        If dst specifies a directory, the file will be copied into dst using
        the base filename from src. If dst specifies a file that already
        exists, it will be replaced.

        Args:
            src (str or Path): A file to be copied.
            dst (str or Path): Copy file to to local dst.

        Returns:
            str: If dst specifies a directory, the file will be copied into dst
            using the base filename from src.

        Examples:
            >>> backend = LocalBackend()
            >>> # dst is a file
            >>> src = '/path/of/file'
            >>> dst = '/path1/of/file1'
            >>> # src will be copied to '/path1/of/file1'
            >>> backend.copyfile_to_local(src, dst)
            '/path1/of/file1'

            >>> # dst is a directory
            >>> dst = '/path1/of/dir'
            >>> # src will be copied to
            >>> backend.copyfile_to_local(src, dst)
            '/path1/of/dir/file'
        """
        return self.copyfile(src, dst)

    def copytree_to_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Recursively copy an entire directory tree rooted at src to a local
        directory named dst and return the destination directory.

        Args:
            src (str or Path): A directory to be copied.
            dst (str or Path): Copy directory to local dst.
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.

        Returns:
            str: The destination directory.

        Examples:
            >>> backend = LocalBackend()
            >>> src = '/path/of/dir1'
            >>> dst = '/path/of/dir2'
            >>> backend.copytree_from_local(src, dst)
            '/path/of/dir2'
        """
        return self.copytree(src, dst)

    def remove(self, filepath: Union[str, Path]) -> None:
        """Remove a file.

        Args:
            filepath (str or Path): Path to be removed.

        Raises:
            IsADirectoryError: If filepath is a directory, an IsADirectoryError
                will be raised.
            FileNotFoundError: If filepath does not exist, an FileNotFoundError
                will be raised.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.remove(filepath)
        """
        if not self.exists(filepath):
            raise FileNotFoundError(f"filepath {filepath} does not exist")

        if self.isdir(filepath):
            raise IsADirectoryError("filepath should be a file")

        os.remove(filepath)

    def rmtree(self, dir_path: Union[str, Path]) -> None:
        """Recursively delete a directory tree.

        Args:
            dir_path (str or Path): A directory to be removed.

        Examples:
            >>> dir_path = '/path/of/dir'
            >>> backend.rmtree(dir_path)
        """
        shutil.rmtree(dir_path)

    def copy_if_symlink_fails(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> bool:
        """Create a symbolic link pointing to src named dst.

        If failed to create a symbolic link pointing to src, directly copy src
        to dst instead.

        Args:
            src (str or Path): Create a symbolic link pointing to src.
            dst (str or Path): Create a symbolic link named dst.

        Returns:
            bool: Return True if successfully create a symbolic link pointing
            to src. Otherwise, return False.

        Examples:
            >>> backend = LocalBackend()
            >>> src = '/path/of/file'
            >>> dst = '/path1/of/file1'
            >>> backend.copy_if_symlink_fails(src, dst)
            True
            >>> src = '/path/of/dir'
            >>> dst = '/path1/of/dir1'
            >>> backend.copy_if_symlink_fails(src, dst)
            True
        """
        try:
            os.symlink(src, dst)
            return True
        except Exception:
            if self.isfile(src):
                self.copyfile(src, dst)
            else:
                self.copytree(src, dst)
            return False

    def list_dir_or_file(
        self,
        dir_path: Union[str, Path],
        list_dir: bool = True,
        list_file: bool = True,
        suffix: Optional[Union[str, Tuple[str]]] = None,
        recursive: bool = False,
    ) -> Iterator[str]:
        """Scan a directory to find the interested directories or files in
        arbitrary order.

        Note:
            :meth:`list_dir_or_file` returns the path relative to ``dir_path``.

        Args:
            dir_path (str or Path): Path of the directory.
            list_dir (bool): List the directories. Defaults to True.
            list_file (bool): List the path of files. Defaults to True.
            suffix (str or tuple[str], optional): File suffix that we are
                interested in. Defaults to None.
            recursive (bool): If set to True, recursively scan the directory.
                Defaults to False.

        Yields:
            Iterable[str]: A relative path to ``dir_path``.

        Examples:
            >>> backend = LocalBackend()
            >>> dir_path = '/path/of/dir'
            >>> # list those files and directories in current directory
            >>> for file_path in backend.list_dir_or_file(dir_path):
            ...     print(file_path)
            >>> # only list files
            >>> for file_path in backend.list_dir_or_file(dir_path, list_dir=False):
            ...     print(file_path)
            >>> # only list directories
            >>> for file_path in backend.list_dir_or_file(dir_path, list_file=False):
            ...     print(file_path)
            >>> # only list files ending with specified suffixes
            >>> for file_path in backend.list_dir_or_file(dir_path, suffix='.txt'):
            ...     print(file_path)
            >>> # list all files and directory recursively
            >>> for file_path in backend.list_dir_or_file(dir_path, recursive=True):
            ...     print(file_path)
        """  # noqa: E501
        if list_dir and suffix is not None:
            raise TypeError("`suffix` should be None when `list_dir` is True")

        if (suffix is not None) and not isinstance(suffix, (str, tuple)):
            raise TypeError("`suffix` must be a string or tuple of strings")

        root = dir_path

        def _list_dir_or_file(dir_path, list_dir, list_file, suffix, recursive):
            for entry in os.scandir(dir_path):
                if not entry.name.startswith(".") and entry.is_file():
                    rel_path = osp.relpath(entry.path, root)
                    if (suffix is None or rel_path.endswith(suffix)) and list_file:
                        yield rel_path
                elif osp.isdir(entry.path):
                    if list_dir:
                        rel_dir = osp.relpath(entry.path, root)
                        yield rel_dir
                    if recursive:
                        yield from _list_dir_or_file(
                            entry.path, list_dir, list_file, suffix, recursive
                        )

        return _list_dir_or_file(dir_path, list_dir, list_file, suffix, recursive)


class HardDiskBackend(LocalBackend):
    """Raw hard disks storage backend."""

    def __init__(self) -> None:
        print_log(
            '"HardDiskBackend" is the alias of "LocalBackend" '
            "and the former will be deprecated in future.",
            logger="current",
            level=logging.WARNING,
        )

    @property
    def name(self):
        return self.__class__.__name__


class LmdbBackend(BaseStorageBackend):
    """Lmdb storage backend.

    Args:
        db_path (str): Lmdb database path.
        readonly (bool): Lmdb environment parameter. If True, disallow any
            write operations. Defaults to True.
        lock (bool): Lmdb environment parameter. If False, when concurrent
            access occurs, do not lock the database. Defaults to False.
        readahead (bool): Lmdb environment parameter. If False, disable the OS
            filesystem readahead mechanism, which may improve random read
            performance when a database is larger than RAM. Defaults to False.
        **kwargs: Keyword arguments passed to `lmdb.open`.

    Attributes:
        db_path (str): Lmdb database path.
    """

    def __init__(self, db_path, readonly=True, lock=False, readahead=False, **kwargs):
        try:
            import lmdb  # noqa: F401
        except ImportError:
            raise ImportError('Please run "pip install lmdb" to enable LmdbBackend.')

        self.db_path = str(db_path)
        self.readonly = readonly
        self.lock = lock
        self.readahead = readahead
        self.kwargs = kwargs
        self._client = None

    def get(self, filepath: Union[str, Path]) -> bytes:
        """Get values according to the filepath.

        Args:
            filepath (str or Path): Here, filepath is the lmdb key.

        Returns:
            bytes: Expected bytes object.

        Examples:
            >>> backend = LmdbBackend('path/to/lmdb')
            >>> backend.get('key')
            b'hello world'
        """
        if self._client is None:
            self._client = self._get_client()

        filepath = str(filepath)
        with self._client.begin(write=False) as txn:
            value_buf = txn.get(filepath.encode("ascii"))
        return value_buf

    def get_text(self, filepath, encoding=None):
        raise NotImplementedError

    def _get_client(self):
        import lmdb

        return lmdb.open(
            self.db_path,
            readonly=self.readonly,
            lock=self.lock,
            readahead=self.readahead,
            **self.kwargs,
        )

    def __del__(self):
        if self._client is not None:
            self._client.close()


def has_method(obj: object, method: str) -> bool:
    """Check whether the object has a method.

    Args:
        method (str): The method name to check.
        obj (object): The object to check.

    Returns:
        bool: True if the object has the method else False.
    """
    return hasattr(obj, method) and callable(getattr(obj, method))


class PetrelBackend(BaseStorageBackend):
    """Petrel storage backend (for internal usage).

    PetrelBackend supports reading and writing data to multiple clusters.
    If the file path contains the cluster name, PetrelBackend will read data
    from specified cluster or write data to it. Otherwise, PetrelBackend will
    access the default cluster.

    Args:
        path_mapping (dict, optional): Path mapping dict from local path to
            Petrel path. When ``path_mapping={'src': 'dst'}``, ``src`` in
            ``filepath`` will be replaced by ``dst``. Defaults to None.
        enable_mc (bool, optional): Whether to enable memcached support.
            Defaults to True.
        conf_path (str, optional): Config path of Petrel client. Default: None.
            `New in version 0.3.3`.

    Examples:
        >>> backend = PetrelBackend()
        >>> filepath1 = 'petrel://path/of/file'
        >>> filepath2 = 'cluster-name:petrel://path/of/file'
        >>> backend.get(filepath1)  # get data from default cluster
        >>> client.get(filepath2)  # get data from 'cluster-name' cluster
    """

    def __init__(
        self,
        path_mapping: Optional[dict] = None,
        enable_mc: bool = True,
        conf_path: Optional[str] = None,
    ):
        try:
            from petrel_client import client
        except ImportError:
            raise ImportError(
                "Please install petrel_client to enable " "PetrelBackend."
            )

        self._client = client.Client(conf_path=conf_path, enable_mc=enable_mc)
        assert isinstance(path_mapping, dict) or path_mapping is None
        self.path_mapping = path_mapping

    def _map_path(self, filepath: Union[str, Path]) -> str:
        """Map ``filepath`` to a string path whose prefix will be replaced by
        :attr:`self.path_mapping`.

        Args:
            filepath (str or Path): Path to be mapped.
        """
        filepath = str(filepath)
        if self.path_mapping is not None:
            for k, v in self.path_mapping.items():
                filepath = filepath.replace(k, v, 1)
        return filepath

    def _format_path(self, filepath: str) -> str:
        """Convert a ``filepath`` to standard format of petrel oss.

        If the ``filepath`` is concatenated by ``os.path.join``, in a Windows
        environment, the ``filepath`` will be the format of
        's3://bucket_name\\image.jpg'. By invoking :meth:`_format_path`, the
        above ``filepath`` will be converted to 's3://bucket_name/image.jpg'.

        Args:
            filepath (str): Path to be formatted.
        """
        return re.sub(r"\\+", "/", filepath)

    def _replace_prefix(self, filepath: Union[str, Path]) -> str:
        filepath = str(filepath)
        return filepath.replace("petrel://", "s3://")

    def get(self, filepath: Union[str, Path]) -> bytes:
        """Read bytes from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Return bytes read from filepath.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.get(filepath)
            b'hello world'
        """
        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        value = self._client.Get(filepath)
        return value

    def get_text(
        self,
        filepath: Union[str, Path],
        encoding: str = "utf-8",
    ) -> str:
        """Read text from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Defaults to 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.get_text(filepath)
            'hello world'
        """
        return str(self.get(filepath), encoding=encoding)

    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        """Write bytes to a given ``filepath``.

        Args:
            obj (bytes): Data to be saved.
            filepath (str or Path): Path to write data.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.put(b'hello world', filepath)
        """
        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        self._client.put(filepath, obj)

    def put_text(
        self,
        obj: str,
        filepath: Union[str, Path],
        encoding: str = "utf-8",
    ) -> None:
        """Write text to a given ``filepath``.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to encode the ``obj``.
                Defaults to 'utf-8'.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.put_text('hello world', filepath)
        """
        self.put(bytes(obj, encoding=encoding), filepath)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.

        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.exists(filepath)
            True
        """
        if not (
            has_method(self._client, "contains") and has_method(self._client, "isdir")
        ):
            raise NotImplementedError(
                "Current version of Petrel Python SDK has not supported "
                "the `contains` and `isdir` methods, please use a higher"
                "version or dev branch instead."
            )

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        return self._client.contains(filepath) or self._client.isdir(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
            ``False`` otherwise.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/dir'
            >>> backend.isdir(filepath)
            True
        """
        if not has_method(self._client, "isdir"):
            raise NotImplementedError(
                "Current version of Petrel Python SDK has not supported "
                "the `isdir` method, please use a higher version or dev"
                " branch instead."
            )

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        return self._client.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
            otherwise.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.isfile(filepath)
            True
        """
        if not has_method(self._client, "contains"):
            raise NotImplementedError(
                "Current version of Petrel Python SDK has not supported "
                "the `contains` method, please use a higher version or "
                "dev branch instead."
            )

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        return self._client.contains(filepath)

    def join_path(
        self,
        filepath: Union[str, Path],
        *filepaths: Union[str, Path],
    ) -> str:
        r"""Concatenate all file paths.

        Join one or more filepath components intelligently. The return value
        is the concatenation of filepath and any members of \*filepaths.

        Args:
            filepath (str or Path): Path to be concatenated.

        Returns:
            str: The result after concatenation.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.join_path(filepath, 'another/path')
            'petrel://path/of/file/another/path'
            >>> backend.join_path(filepath, '/another/path')
            'petrel://path/of/file/another/path'
        """
        filepath = self._format_path(self._map_path(filepath))
        if filepath.endswith("/"):
            filepath = filepath[:-1]
        formatted_paths = [filepath]
        for path in filepaths:
            formatted_path = self._format_path(self._map_path(path))
            formatted_paths.append(formatted_path.lstrip("/"))

        return "/".join(formatted_paths)

    @contextmanager
    def get_local_path(
        self,
        filepath: Union[str, Path],
    ) -> Generator[Union[str, Path], None, None]:
        """Download a file from ``filepath`` to a local temporary directory,
        and return the temporary path.

        ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Args:
            filepath (str or Path): Download a file from ``filepath``.

        Yields:
            Iterable[str]: Only yield one temporary path.

        Examples:
            >>> backend = PetrelBackend()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> filepath = 'petrel://path/of/file'
            >>> with backend.get_local_path(filepath) as path:
            ...     # do something here
        """
        assert self.isfile(filepath)
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.get(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

    def copyfile(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Copy a file src to dst and return the destination file.

        src and dst should have the same prefix. If dst specifies a directory,
        the file will be copied into dst using the base filename from src. If
        dst specifies a file that already exists, it will be replaced.

        Args:
            src (str or Path): A file to be copied.
            dst (str or Path): Copy file to dst.

        Returns:
            str: The destination file.

        Raises:
            SameFileError: If src and dst are the same file, a SameFileError
                will be raised.

        Examples:
            >>> backend = PetrelBackend()
            >>> # dst is a file
            >>> src = 'petrel://path/of/file'
            >>> dst = 'petrel://path/of/file1'
            >>> backend.copyfile(src, dst)
            'petrel://path/of/file1'

            >>> # dst is a directory
            >>> dst = 'petrel://path/of/dir'
            >>> backend.copyfile(src, dst)
            'petrel://path/of/dir/file'
        """
        src = self._format_path(self._map_path(src))
        dst = self._format_path(self._map_path(dst))
        if self.isdir(dst):
            dst = self.join_path(dst, src.split("/")[-1])

        if src == dst:
            raise SameFileError("src and dst should not be same")

        self.put(self.get(src), dst)
        return dst

    def copytree(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Recursively copy an entire directory tree rooted at src to a
        directory named dst and return the destination directory.

        src and dst should have the same prefix.

        Args:
            src (str or Path): A directory to be copied.
            dst (str or Path): Copy directory to dst.
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.

        Returns:
            str: The destination directory.

        Raises:
            FileExistsError: If dst had already existed, a FileExistsError will
                be raised.

        Examples:
            >>> backend = PetrelBackend()
            >>> src = 'petrel://path/of/dir'
            >>> dst = 'petrel://path/of/dir1'
            >>> backend.copytree(src, dst)
            'petrel://path/of/dir1'
        """
        src = self._format_path(self._map_path(src))
        dst = self._format_path(self._map_path(dst))

        if self.exists(dst):
            raise FileExistsError("dst should not exist")

        for path in self.list_dir_or_file(src, list_dir=False, recursive=True):
            src_path = self.join_path(src, path)
            dst_path = self.join_path(dst, path)
            self.put(self.get(src_path), dst_path)

        return dst

    def copyfile_from_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Upload a local file src to dst and return the destination file.

        Args:
            src (str or Path): A local file to be copied.
            dst (str or Path): Copy file to dst.
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.

        Returns:
            str: If dst specifies a directory, the file will be copied into dst
            using the base filename from src.

        Examples:
            >>> backend = PetrelBackend()
            >>> # dst is a file
            >>> src = 'path/of/your/file'
            >>> dst = 'petrel://path/of/file1'
            >>> backend.copyfile_from_local(src, dst)
            'petrel://path/of/file1'

            >>> # dst is a directory
            >>> dst = 'petrel://path/of/dir'
            >>> backend.copyfile_from_local(src, dst)
            'petrel://path/of/dir/file'
        """
        dst = self._format_path(self._map_path(dst))
        if self.isdir(dst):
            dst = self.join_path(dst, osp.basename(src))

        with open(src, "rb") as f:
            self.put(f.read(), dst)

        return dst

    def copytree_from_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Recursively copy an entire directory tree rooted at src to a
        directory named dst and return the destination directory.

        Args:
            src (str or Path): A local directory to be copied.
            dst (str or Path): Copy directory to dst.

        Returns:
            str: The destination directory.

        Raises:
            FileExistsError: If dst had already existed, a FileExistsError will
                be raised.

        Examples:
            >>> backend = PetrelBackend()
            >>> src = 'path/of/your/dir'
            >>> dst = 'petrel://path/of/dir1'
            >>> backend.copytree_from_local(src, dst)
            'petrel://path/of/dir1'
        """
        dst = self._format_path(self._map_path(dst))
        if self.exists(dst):
            raise FileExistsError("dst should not exist")

        src = str(src)

        for cur_dir, _, files in os.walk(src):
            for f in files:
                src_path = osp.join(cur_dir, f)
                dst_path = self.join_path(dst, src_path.replace(src, ""))
                self.copyfile_from_local(src_path, dst_path)

        return dst

    def copyfile_to_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> Union[str, Path]:
        """Copy the file src to local dst and return the destination file.

        If dst specifies a directory, the file will be copied into dst using
        the base filename from src. If dst specifies a file that already
        exists, it will be replaced.

        Args:
            src (str or Path): A file to be copied.
            dst (str or Path): Copy file to to local dst.

        Returns:
            str: If dst specifies a directory, the file will be copied into dst
            using the base filename from src.

        Examples:
            >>> backend = PetrelBackend()
            >>> # dst is a file
            >>> src = 'petrel://path/of/file'
            >>> dst = 'path/of/your/file'
            >>> backend.copyfile_to_local(src, dst)
            'path/of/your/file'

            >>> # dst is a directory
            >>> dst = 'path/of/your/dir'
            >>> backend.copyfile_to_local(src, dst)
            'path/of/your/dir/file'
        """
        if osp.isdir(dst):
            basename = osp.basename(src)
            if isinstance(dst, str):
                dst = osp.join(dst, basename)
            else:
                assert isinstance(dst, Path)
                dst = dst / basename

        with open(dst, "wb") as f:
            f.write(self.get(src))

        return dst

    def copytree_to_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> Union[str, Path]:
        """Recursively copy an entire directory tree rooted at src to a local
        directory named dst and return the destination directory.

        Args:
            src (str or Path): A directory to be copied.
            dst (str or Path): Copy directory to local dst.
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.

        Returns:
            str: The destination directory.

        Examples:
            >>> backend = PetrelBackend()
            >>> src = 'petrel://path/of/dir'
            >>> dst = 'path/of/your/dir'
            >>> backend.copytree_to_local(src, dst)
            'path/of/your/dir'
        """
        for path in self.list_dir_or_file(src, list_dir=False, recursive=True):
            dst_path = osp.join(dst, path)
            mmengine.mkdir_or_exist(osp.dirname(dst_path))
            with open(dst_path, "wb") as f:
                f.write(self.get(self.join_path(src, path)))

        return dst

    def remove(self, filepath: Union[str, Path]) -> None:
        """Remove a file.

        Args:
            filepath (str or Path): Path to be removed.

        Raises:
            FileNotFoundError: If filepath does not exist, an FileNotFoundError
                will be raised.
            IsADirectoryError: If filepath is a directory, an IsADirectoryError
                will be raised.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.remove(filepath)
        """
        if not has_method(self._client, "delete"):
            raise NotImplementedError(
                "Current version of Petrel Python SDK has not supported "
                "the `delete` method, please use a higher version or dev "
                "branch instead."
            )

        if not self.exists(filepath):
            raise FileNotFoundError(f"filepath {filepath} does not exist")

        if self.isdir(filepath):
            raise IsADirectoryError("filepath should be a file")

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        self._client.delete(filepath)

    def rmtree(self, dir_path: Union[str, Path]) -> None:
        """Recursively delete a directory tree.

        Args:
            dir_path (str or Path): A directory to be removed.

        Examples:
            >>> backend = PetrelBackend()
            >>> dir_path = 'petrel://path/of/dir'
            >>> backend.rmtree(dir_path)
        """
        for path in self.list_dir_or_file(dir_path, list_dir=False, recursive=True):
            filepath = self.join_path(dir_path, path)
            self.remove(filepath)

    def copy_if_symlink_fails(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> bool:
        """Create a symbolic link pointing to src named dst.

        Directly copy src to dst because PetrelBacekend does not support create
        a symbolic link.

        Args:
            src (str or Path): A file or directory to be copied.
            dst (str or Path): Copy a file or directory to dst.
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.

        Returns:
            bool: Return False because PetrelBackend does not support create
            a symbolic link.

        Examples:
            >>> backend = PetrelBackend()
            >>> src = 'petrel://path/of/file'
            >>> dst = 'petrel://path/of/your/file'
            >>> backend.copy_if_symlink_fails(src, dst)
            False
            >>> src = 'petrel://path/of/dir'
            >>> dst = 'petrel://path/of/your/dir'
            >>> backend.copy_if_symlink_fails(src, dst)
            False
        """
        if self.isfile(src):
            self.copyfile(src, dst)
        else:
            self.copytree(src, dst)
        return False

    def list_dir_or_file(
        self,
        dir_path: Union[str, Path],
        list_dir: bool = True,
        list_file: bool = True,
        suffix: Optional[Union[str, Tuple[str]]] = None,
        recursive: bool = False,
    ) -> Iterator[str]:
        """Scan a directory to find the interested directories or files in
        arbitrary order.

        Note:
            Petrel has no concept of directories but it simulates the directory
            hierarchy in the filesystem through public prefixes. In addition,
            if the returned path ends with '/', it means the path is a public
            prefix which is a logical directory.

        Note:
            :meth:`list_dir_or_file` returns the path relative to ``dir_path``.
            In addition, the returned path of directory will not contains the
            suffix '/' which is consistent with other backends.

        Args:
            dir_path (str | Path): Path of the directory.
            list_dir (bool): List the directories. Defaults to True.
            list_file (bool): List the path of files. Defaults to True.
            suffix (str or tuple[str], optional):  File suffix
                that we are interested in. Defaults to None.
            recursive (bool): If set to True, recursively scan the
                directory. Defaults to False.

        Yields:
            Iterable[str]: A relative path to ``dir_path``.

        Examples:
            >>> backend = PetrelBackend()
            >>> dir_path = 'petrel://path/of/dir'
            >>> # list those files and directories in current directory
            >>> for file_path in backend.list_dir_or_file(dir_path):
            ...     print(file_path)
            >>> # only list files
            >>> for file_path in backend.list_dir_or_file(dir_path, list_dir=False):
            ...     print(file_path)
            >>> # only list directories
            >>> for file_path in backend.list_dir_or_file(dir_path, list_file=False):
            ...     print(file_path)
            >>> # only list files ending with specified suffixes
            >>> for file_path in backend.list_dir_or_file(dir_path, suffix='.txt'):
            ...     print(file_path)
            >>> # list all files and directory recursively
            >>> for file_path in backend.list_dir_or_file(dir_path, recursive=True):
            ...     print(file_path)
        """  # noqa: E501
        if not has_method(self._client, "list"):
            raise NotImplementedError(
                "Current version of Petrel Python SDK has not supported "
                "the `list` method, please use a higher version or dev"
                " branch instead."
            )

        dir_path = self._map_path(dir_path)
        dir_path = self._format_path(dir_path)
        dir_path = self._replace_prefix(dir_path)
        if list_dir and suffix is not None:
            raise TypeError("`list_dir` should be False when `suffix` is not None")

        if (suffix is not None) and not isinstance(suffix, (str, tuple)):
            raise TypeError("`suffix` must be a string or tuple of strings")

        # Petrel's simulated directory hierarchy assumes that directory paths
        # should end with `/`
        if not dir_path.endswith("/"):
            dir_path += "/"

        root = dir_path

        def _list_dir_or_file(dir_path, list_dir, list_file, suffix, recursive):
            for path in self._client.list(dir_path):
                # the `self.isdir` is not used here to determine whether path
                # is a directory, because `self.isdir` relies on
                # `self._client.list`
                if path.endswith("/"):  # a directory path
                    next_dir_path = self.join_path(dir_path, path)
                    if list_dir:
                        # get the relative path and exclude the last
                        # character '/'
                        rel_dir = next_dir_path[len(root) : -1]
                        yield rel_dir
                    if recursive:
                        yield from _list_dir_or_file(
                            next_dir_path, list_dir, list_file, suffix, recursive
                        )
                else:  # a file path
                    absolute_path = self.join_path(dir_path, path)
                    rel_path = absolute_path[len(root) :]
                    if (suffix is None or rel_path.endswith(suffix)) and list_file:
                        yield rel_path

        return _list_dir_or_file(dir_path, list_dir, list_file, suffix, recursive)

    def generate_presigned_url(
        self, url: str, client_method: str = "get_object", expires_in: int = 3600
    ) -> str:
        """Generate the presigned url of video stream which can be passed to
        mmcv.VideoReader. Now only work on Petrel backend.

        Note:
            Now only work on Petrel backend.

        Args:
            url (str): Url of video stream.
            client_method (str): Method of client, 'get_object' or
                'put_object'. Default: 'get_object'.
            expires_in (int): expires, in seconds. Default: 3600.

        Returns:
            str: Generated presigned url.
        """
        return self._client.generate_presigned_url(url, client_method, expires_in)


class HTTPBackend(BaseStorageBackend):
    """HTTP and HTTPS storage bachend."""

    def get(self, filepath: str) -> bytes:
        """Read bytes from a given ``filepath``.

        Args:
            filepath (str): Path to read data.

        Returns:
            bytes: Expected bytes object.

        Examples:
            >>> backend = HTTPBackend()
            >>> backend.get('http://path/of/file')
            b'hello world'
        """
        return urlopen(filepath).read()

    def get_text(self, filepath, encoding="utf-8") -> str:
        """Read text from a given ``filepath``.

        Args:
            filepath (str): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Defaults to 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.

        Examples:
            >>> backend = HTTPBackend()
            >>> backend.get_text('http://path/of/file')
            'hello world'
        """
        return urlopen(filepath).read().decode(encoding)

    @contextmanager
    def get_local_path(self, filepath: str) -> Generator[Union[str, Path], None, None]:
        """Download a file from ``filepath`` to a local temporary directory,
        and return the temporary path.

        ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Args:
            filepath (str): Download a file from ``filepath``.

        Yields:
            Iterable[str]: Only yield one temporary path.

        Examples:
            >>> backend = HTTPBackend()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> with backend.get_local_path('http://path/of/file') as path:
            ...     # do something here
        """
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.get(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

def is_filepath(x):
    return is_str(x) or isinstance(x, Path)


def _get_file_backend(prefix: str, backend_args: dict):
    """Return a file backend based on the prefix or backend_args.

    Args:
        prefix (str): Prefix of uri.
        backend_args (dict): Arguments to instantiate the corresponding
            backend.
    """
    # backend name has a higher priority
    if "backend" in backend_args:
        # backend_args should not be modified
        backend_args_bak = backend_args.copy()
        backend_name = backend_args_bak.pop("backend")
        backend = backends[backend_name](**backend_args_bak)
    else:
        backend = prefix_to_backends[prefix](**backend_args)
    return backend


class FileClient:
    """A general file client to access files in different backends.

    The client loads a file or text in a specified backend from its path
    and returns it as a binary or text file. There are two ways to choose a
    backend, the name of backend and the prefix of path. Although both of them
    can be used to choose a storage backend, ``backend`` has a higher priority
    that is if they are all set, the storage backend will be chosen by the
    backend argument. If they are all `None`, the disk backend will be chosen.
    Note that It can also register other backend accessor with a given name,
    prefixes, and backend class. In addition, We use the singleton pattern to
    avoid repeated object creation. If the arguments are the same, the same
    object will be returned.

    Warning:
        `FileClient` will be deprecated in future. Please use io functions
        in https://mmengine.readthedocs.io/en/latest/api/fileio.html#file-io

    Args:
        backend (str, optional): The storage backend type. Options are "disk",
            "memcached", "lmdb", "http" and "petrel". Defaults to None.
        prefix (str, optional): The prefix of the registered storage backend.
            Options are "s3", "http", "https". Defaults to None.

    Examples:
        >>> # only set backend
        >>> file_client = FileClient(backend='petrel')
        >>> # only set prefix
        >>> file_client = FileClient(prefix='s3')
        >>> # set both backend and prefix but use backend to choose client
        >>> file_client = FileClient(backend='petrel', prefix='s3')
        >>> # if the arguments are the same, the same object is returned
        >>> file_client1 = FileClient(backend='petrel')
        >>> file_client1 is file_client
        True

    Attributes:
        client (:obj:`BaseStorageBackend`): The backend object.
    """

    _backends = {
        "disk": HardDiskBackend,
        "memcached": MemcachedBackend,
        "lmdb": LmdbBackend,
        "petrel": PetrelBackend,
        "http": HTTPBackend,
    }

    _prefix_to_backends: dict = {
        "s3": PetrelBackend,
        "petrel": PetrelBackend,
        "http": HTTPBackend,
        "https": HTTPBackend,
    }

    _instances: dict = {}

    client: Any

    def __new__(cls, backend=None, prefix=None, **kwargs):
        print_log(
            '"FileClient" will be deprecated in future. Please use io '
            "functions in "
            "https://mmengine.readthedocs.io/en/latest/api/fileio.html#file-io",  # noqa: E501
            logger="current",
            level=logging.WARNING,
        )
        if backend is None and prefix is None:
            backend = "disk"
        if backend is not None and backend not in cls._backends:
            raise ValueError(
                f"Backend {backend} is not supported. Currently supported ones"
                f" are {list(cls._backends.keys())}"
            )
        if prefix is not None and prefix not in cls._prefix_to_backends:
            raise ValueError(
                f"prefix {prefix} is not supported. Currently supported ones "
                f"are {list(cls._prefix_to_backends.keys())}"
            )

        # concatenate the arguments to a unique key for determining whether
        # objects with the same arguments were created
        arg_key = f"{backend}:{prefix}"
        for key, value in kwargs.items():
            arg_key += f":{key}:{value}"

        # if a backend was overridden, it will create a new object
        if arg_key in cls._instances:
            _instance = cls._instances[arg_key]
        else:
            # create a new object and put it to _instance
            _instance = super().__new__(cls)
            if backend is not None:
                _instance.client = cls._backends[backend](**kwargs)
            else:
                _instance.client = cls._prefix_to_backends[prefix](**kwargs)

            cls._instances[arg_key] = _instance

        return _instance

    @property
    def name(self):
        return self.client.name

    @property
    def allow_symlink(self):
        return self.client.allow_symlink

    @staticmethod
    def parse_uri_prefix(uri: Union[str, Path]) -> Optional[str]:
        """Parse the prefix of a uri.

        Args:
            uri (str | Path): Uri to be parsed that contains the file prefix.

        Examples:
            >>> FileClient.parse_uri_prefix('s3://path/of/your/file')
            's3'

        Returns:
            str | None: Return the prefix of uri if the uri contains '://' else
            ``None``.
        """
        assert is_filepath(uri)
        uri = str(uri)
        if "://" not in uri:
            return None
        else:
            prefix, _ = uri.split("://")
            # In the case of PetrelBackend, the prefix may contains the cluster
            # name like clusterName:s3
            if ":" in prefix:
                _, prefix = prefix.split(":")
            return prefix

    @classmethod
    def infer_client(
        cls,
        file_client_args: Optional[dict] = None,
        uri: Optional[Union[str, Path]] = None,
    ) -> "FileClient":
        """Infer a suitable file client based on the URI and arguments.

        Args:
            file_client_args (dict, optional): Arguments to instantiate a
                FileClient. Defaults to None.
            uri (str | Path, optional): Uri to be parsed that contains the file
                prefix. Defaults to None.

        Examples:
            >>> uri = 's3://path/of/your/file'
            >>> file_client = FileClient.infer_client(uri=uri)
            >>> file_client_args = {'backend': 'petrel'}
            >>> file_client = FileClient.infer_client(file_client_args)

        Returns:
            FileClient: Instantiated FileClient object.
        """
        assert file_client_args is not None or uri is not None
        if file_client_args is None:
            file_prefix = cls.parse_uri_prefix(uri)  # type: ignore
            return cls(prefix=file_prefix)
        else:
            return cls(**file_client_args)

    @classmethod
    def _register_backend(cls, name, backend, force=False, prefixes=None):
        if not isinstance(name, str):
            raise TypeError(
                "the backend name should be a string, " f"but got {type(name)}"
            )
        if not inspect.isclass(backend):
            raise TypeError(f"backend should be a class but got {type(backend)}")
        if not issubclass(backend, BaseStorageBackend):
            raise TypeError(
                f"backend {backend} is not a subclass of BaseStorageBackend"
            )
        if not force and name in cls._backends:
            raise KeyError(
                f"{name} is already registered as a storage backend, "
                'add "force=True" if you want to override it'
            )

        if name in cls._backends and force:
            for arg_key, instance in list(cls._instances.items()):
                if isinstance(instance.client, cls._backends[name]):
                    cls._instances.pop(arg_key)
        cls._backends[name] = backend

        if prefixes is not None:
            if isinstance(prefixes, str):
                prefixes = [prefixes]
            else:
                assert isinstance(prefixes, (list, tuple))
            for prefix in prefixes:
                if prefix not in cls._prefix_to_backends:
                    cls._prefix_to_backends[prefix] = backend
                elif (prefix in cls._prefix_to_backends) and force:
                    overridden_backend = cls._prefix_to_backends[prefix]
                    for arg_key, instance in list(cls._instances.items()):
                        if isinstance(instance.client, overridden_backend):
                            cls._instances.pop(arg_key)
                else:
                    raise KeyError(
                        f"{prefix} is already registered as a storage backend,"
                        ' add "force=True" if you want to override it'
                    )

    @classmethod
    def register_backend(cls, name, backend=None, force=False, prefixes=None):
        """Register a backend to FileClient.

        This method can be used as a normal class method or a decorator.

        .. code-block:: python

            class NewBackend(BaseStorageBackend):

                def get(self, filepath):
                    return filepath

                def get_text(self, filepath):
                    return filepath

            FileClient.register_backend('new', NewBackend)

        or

        .. code-block:: python

            @FileClient.register_backend('new')
            class NewBackend(BaseStorageBackend):

                def get(self, filepath):
                    return filepath

                def get_text(self, filepath):
                    return filepath

        Args:
            name (str): The name of the registered backend.
            backend (class, optional): The backend class to be registered,
                which must be a subclass of :class:`BaseStorageBackend`.
                When this method is used as a decorator, backend is None.
                Defaults to None.
            force (bool, optional): Whether to override the backend if the name
                has already been registered. Defaults to False.
            prefixes (str or list[str] or tuple[str], optional): The prefixes
                of the registered storage backend. Defaults to None.
                `New in version 1.3.15.`
        """
        if backend is not None:
            cls._register_backend(name, backend, force=force, prefixes=prefixes)
            return

        def _register(backend_cls):
            cls._register_backend(name, backend_cls, force=force, prefixes=prefixes)
            return backend_cls

        return _register

    def get(self, filepath: Union[str, Path]) -> Union[bytes, memoryview]:
        """Read data from a given ``filepath`` with 'rb' mode.

        Note:
            There are two types of return values for ``get``, one is ``bytes``
            and the other is ``memoryview``. The advantage of using memoryview
            is that you can avoid copying, and if you want to convert it to
            ``bytes``, you can use ``.tobytes()``.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes | memoryview: Expected bytes object or a memory view of the
            bytes object.
        """
        return self.client.get(filepath)

    def get_text(self, filepath: Union[str, Path], encoding="utf-8") -> str:
        """Read data from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Defaults to 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.
        """
        return self.client.get_text(filepath, encoding)

    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        """Write data to a given ``filepath`` with 'wb' mode.

        Note:
            ``put`` should create a directory if the directory of ``filepath``
            does not exist.

        Args:
            obj (bytes): Data to be written.
            filepath (str or Path): Path to write data.
        """
        self.client.put(obj, filepath)

    def put_text(self, obj: str, filepath: Union[str, Path]) -> None:
        """Write data to a given ``filepath`` with 'w' mode.

        Note:
            ``put_text`` should create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str, optional): The encoding format used to open the
                `filepath`. Defaults to 'utf-8'.
        """
        self.client.put_text(obj, filepath)

    def remove(self, filepath: Union[str, Path]) -> None:
        """Remove a file.

        Args:
            filepath (str, Path): Path to be removed.
        """
        self.client.remove(filepath)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.

        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.
        """
        return self.client.exists(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
            ``False`` otherwise.
        """
        return self.client.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
            otherwise.
        """
        return self.client.isfile(filepath)

    def join_path(
        self, filepath: Union[str, Path], *filepaths: Union[str, Path]
    ) -> str:
        r"""Concatenate all file paths.

        Join one or more filepath components intelligently. The return value
        is the concatenation of filepath and any members of \*filepaths.

        Args:
            filepath (str or Path): Path to be concatenated.

        Returns:
            str: The result of concatenation.
        """
        return self.client.join_path(filepath, *filepaths)

    @contextmanager
    def get_local_path(
        self, filepath: Union[str, Path]
    ) -> Generator[Union[str, Path], None, None]:
        """Download data from ``filepath`` and write the data to local path.

        ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Note:
            If the ``filepath`` is a local path, just return itself.

        .. warning::
            ``get_local_path`` is an experimental interface that may change in
            the future.

        Args:
            filepath (str or Path): Path to be read data.

        Examples:
            >>> file_client = FileClient(prefix='s3')
            >>> with file_client.get_local_path('s3://bucket/abc.jpg') as path:
            ...     # do something here

        Yields:
            Iterable[str]: Only yield one path.
        """
        with self.client.get_local_path(str(filepath)) as local_path:
            yield local_path

    def list_dir_or_file(
        self,
        dir_path: Union[str, Path],
        list_dir: bool = True,
        list_file: bool = True,
        suffix: Optional[Union[str, Tuple[str]]] = None,
        recursive: bool = False,
    ) -> Iterator[str]:
        """Scan a directory to find the interested directories or files in
        arbitrary order.

        Note:
            :meth:`list_dir_or_file` returns the path relative to ``dir_path``.

        Args:
            dir_path (str | Path): Path of the directory.
            list_dir (bool): List the directories. Defaults to True.
            list_file (bool): List the path of files. Defaults to True.
            suffix (str or tuple[str], optional):  File suffix
                that we are interested in. Defaults to None.
            recursive (bool): If set to True, recursively scan the
                directory. Defaults to False.

        Yields:
            Iterable[str]: A relative path to ``dir_path``.
        """
        yield from self.client.list_dir_or_file(
            dir_path, list_dir, list_file, suffix, recursive
        )


def _register_backend(
    name: str,
    backend: Type[BaseStorageBackend],
    force: bool = False,
    prefixes: Union[str, list, tuple, None] = None,
):
    """Register a backend.

    Args:
        name (str): The name of the registered backend.
        backend (BaseStorageBackend): The backend class to be registered,
            which must be a subclass of :class:`BaseStorageBackend`.
        force (bool): Whether to override the backend if the name has already
            been registered. Defaults to False.
        prefixes (str or list[str] or tuple[str], optional): The prefix
            of the registered storage backend. Defaults to None.
    """
    global backends, prefix_to_backends

    if not isinstance(name, str):
        raise TypeError("the backend name should be a string, " f"but got {type(name)}")

    if not inspect.isclass(backend):
        raise TypeError(f"backend should be a class, but got {type(backend)}")
    if not issubclass(backend, BaseStorageBackend):
        raise TypeError(f"backend {backend} is not a subclass of BaseStorageBackend")

    if name in backends and not force:
        raise ValueError(
            f"{name} is already registered as a storage backend, "
            'add "force=True" if you want to override it'
        )
    backends[name] = backend

    if prefixes is not None:
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        else:
            assert isinstance(prefixes, (list, tuple))

        for prefix in prefixes:
            if prefix in prefix_to_backends and not force:
                raise ValueError(
                    f"{prefix} is already registered as a storage backend,"
                    ' add "force=True" if you want to override it'
                )

            prefix_to_backends[prefix] = backend


def register_backend(
    name: str,
    backend: Optional[Type[BaseStorageBackend]] = None,
    force: bool = False,
    prefixes: Union[str, list, tuple, None] = None,
):
    """Register a backend.

    Args:
        name (str): The name of the registered backend.
        backend (class, optional): The backend class to be registered,
            which must be a subclass of :class:`BaseStorageBackend`.
            When this method is used as a decorator, backend is None.
            Defaults to None.
        force (bool): Whether to override the backend if the name has already
            been registered. Defaults to False.
        prefixes (str or list[str] or tuple[str], optional): The prefix
            of the registered storage backend. Defaults to None.

    This method can be used as a normal method or a decorator.

    Examples:

        >>> class NewBackend(BaseStorageBackend):
        ...     def get(self, filepath):
        ...         return filepath
        ...
        ...     def get_text(self, filepath):
        ...         return filepath
        >>> register_backend('new', NewBackend)

        >>> @register_backend('new')
        ... class NewBackend(BaseStorageBackend):
        ...     def get(self, filepath):
        ...         return filepath
        ...
        ...     def get_text(self, filepath):
        ...         return filepath
    """
    if backend is not None:
        _register_backend(name, backend, force=force, prefixes=prefixes)
        return

    def _register(backend_cls):
        _register_backend(name, backend_cls, force=force, prefixes=prefixes)
        return backend_cls

    return _register


register_backend('local', LocalBackend, prefixes='')
register_backend('memcached', MemcachedBackend)
register_backend('lmdb', LmdbBackend)
# To avoid breaking backward Compatibility, 's3' is also used as a
# prefix for PetrelBackend
register_backend('petrel', PetrelBackend, prefixes=['petrel', 's3'])
register_backend('http', HTTPBackend, prefixes=['http', 'https'])


def _parse_uri_prefix(uri: Union[str, Path]) -> str:
    """Parse the prefix of uri.

    Args:
        uri (str or Path): Uri to be parsed that contains the file prefix.

    Examples:
        >>> _parse_uri_prefix('/home/path/of/your/file')
        ''
        >>> _parse_uri_prefix('s3://path/of/your/file')
        's3'
        >>> _parse_uri_prefix('clusterName:s3://path/of/your/file')
        's3'

    Returns:
        str: Return the prefix of uri if the uri contains '://'. Otherwise,
        return ''.
    """
    assert is_filepath(uri)
    uri = str(uri)
    # if uri does not contains '://', the uri will be handled by
    # LocalBackend by default
    if "://" not in uri:
        return ""
    else:
        prefix, _ = uri.split("://")
        # In the case of PetrelBackend, the prefix may contain the cluster
        # name like clusterName:s3://path/of/your/file
        if ":" in prefix:
            _, prefix = prefix.split(":")
        return prefix

backend_instances: dict = {}

def get_file_backend(
    uri: Union[str, Path, None] = None,
    *,
    backend_args: Optional[dict] = None,
    enable_singleton: bool = False,
):
    """Return a file backend based on the prefix of uri or backend_args.

    Args:
        uri (str or Path): Uri to be parsed that contains the file prefix.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        enable_singleton (bool): Whether to enable the singleton pattern.
            If it is True, the backend created will be reused if the
            signature is same with the previous one. Defaults to False.

    Returns:
        BaseStorageBackend: Instantiated Backend object.

    Examples:
        >>> # get file backend based on the prefix of uri
        >>> uri = 's3://path/of/your/file'
        >>> backend = get_file_backend(uri)
        >>> # get file backend based on the backend_args
        >>> backend = get_file_backend(backend_args={'backend': 'petrel'})
        >>> # backend name has a higher priority if 'backend' in backend_args
        >>> backend = get_file_backend(uri, backend_args={'backend': 'petrel'})
    """
    global backend_instances

    if backend_args is None:
        backend_args = {}

    if uri is None and "backend" not in backend_args:
        raise ValueError(
            'uri should not be None when "backend" does not exist in ' "backend_args"
        )

    if uri is not None:
        prefix = _parse_uri_prefix(uri)
    else:
        prefix = ""

    if enable_singleton:
        # TODO: whether to pass sort_key to json.dumps
        unique_key = f"{prefix}:{json.dumps(backend_args)}"
        if unique_key in backend_instances:
            return backend_instances[unique_key]

        backend = _get_file_backend(prefix, backend_args)
        backend_instances[unique_key] = backend
        return backend
    else:
        backend = _get_file_backend(prefix, backend_args)
        return backend


def load(file,
         file_format=None,
         file_client_args=None,
         backend_args=None,
         **kwargs):
    """Load data from json/yaml/pickle files.

    This method provides a unified api for loading data from serialized files.

    ``load`` supports loading data from serialized files those can be storaged
    in different backends.

    Args:
        file (str or :obj:`Path` or file-like object): Filename or a file-like
            object.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "yaml/yml" and
            "pickle/pkl".
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
        backend_args (dict, optional): Arguments to instantiate the
            prefix of uri corresponding backend. Defaults to None.
            New in v0.2.0.

    Examples:
        >>> load('/path/of/your/file')  # file is storaged in disk
        >>> load('https://path/of/your/file')  # file is storaged in Internet
        >>> load('s3://path/of/your/file')  # file is storaged in petrel

    Returns:
        The content from the file.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None and is_str(file):
        file_format = file.split('.')[-1]
    if file_format not in file_handlers:
        raise TypeError(f'Unsupported format: {file_format}')

    if file_client_args is not None:
        warnings.warn(
            '"file_client_args" will be deprecated in future. '
            'Please use "backend_args" instead', DeprecationWarning)
        if backend_args is not None:
            raise ValueError(
                '"file_client_args and "backend_args" cannot be set at the '
                'same time.')

    handler = file_handlers[file_format]
    if is_str(file):
        if file_client_args is not None:
            file_client = FileClient.infer_client(file_client_args, file)
            file_backend = file_client
        else:
            file_backend = get_file_backend(file, backend_args=backend_args)

        if handler.str_like:
            with StringIO(file_backend.get_text(file)) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
        else:
            with BytesIO(file_backend.get(file)) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
    elif hasattr(file, 'read'):
        obj = handler.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj

MODULE2PACKAGE = {
    "mmcls": "mmcls",
    "mmdet": "mmdet",
    "mmdet3d": "mmdet3d",
    "mmseg": "mmsegmentation",
    "mmaction": "mmaction2",
    "mmtrack": "mmtrack",
    "mmpose": "mmpose",
    "mmedit": "mmedit",
    "mmocr": "mmocr",
    "mmgen": "mmgen",
    "mmfewshot": "mmfewshot",
    "mmrazor": "mmrazor",
    "mmflow": "mmflow",
    "mmhuman3d": "mmhuman3d",
    "mmrotate": "mmrotate",
    "mmselfsup": "mmselfsup",
    "mmyolo": "mmyolo",
    "mmpretrain": "mmpretrain",
}


def _get_package_and_cfg_path(cfg_path: str) -> Tuple[str, str]:
    """Get package name and relative config path.

    Args:
        cfg_path (str): External relative config path with 'package::'.

    Returns:
        Tuple[str, str]: Package name and config path.
    """
    if re.match(r"\w*::\w*/\w*", cfg_path) is None:
        raise ValueError(
            "`_get_package_and_cfg_path` is used for get external package, "
            "please specify the package name and relative config path, just "
            "like `mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py`"
        )
    package_cfg = cfg_path.split("::")
    if len(package_cfg) > 2:
        raise ValueError(
            "`::` should only be used to separate package and "
            "config name, but found multiple `::` in "
            f"{cfg_path}"
        )
    package, cfg_path = package_cfg
    assert (
        package in MODULE2PACKAGE
    ), f"mmengine does not support to load {package} config."
    package = MODULE2PACKAGE[package]
    return package, cfg_path


def is_installed(package: str) -> bool:
    """Check package whether installed.

    Args:
        package (str): Name of package to be checked.
    """
    # When executing `import mmengine.runner`,
    # pkg_resources will be imported and it takes too much time.
    # Therefore, import it in function scope to save time.
    import pkg_resources
    from pkg_resources import get_distribution

    # refresh the pkg_resources
    # more datails at https://github.com/pypa/setuptools/issues/373
    importlib.reload(pkg_resources)
    try:
        get_distribution(package)
        return True
    except pkg_resources.DistributionNotFound:
        return False


def package2module(package: str):
    """Infer module name from package.

    Args:
        package (str): Package to infer module name.
    """
    from pkg_resources import get_distribution

    pkg = get_distribution(package)
    if pkg.has_metadata("top_level.txt"):
        module_name = pkg.get_metadata("top_level.txt").split("\n")[0]
        return module_name
    else:
        raise ValueError(f"can not infer the module name of {package}")


def get_installed_path(package: str) -> str:
    """Get installed path of package.

    Args:
        package (str): Name of package.

    Example:
        >>> get_installed_path('mmcls')
        >>> '.../lib/python3.7/site-packages/mmcls'
    """
    from pkg_resources import get_distribution

    # if the package name is not the same as module name, module name should be
    # inferred. For example, mmcv-full is the package name, but mmcv is module
    # name. If we want to get the installed path of mmcv-full, we should concat
    # the pkg.location and module name
    pkg = get_distribution(package)
    possible_path = osp.join(pkg.location, package)
    if osp.exists(possible_path):
        return possible_path
    else:
        return osp.join(pkg.location, package2module(package))

def _get_cfg_metainfo(package_path: str, cfg_path: str) -> dict:
    """Get target meta information from all 'metafile.yml' defined in `mode-
    index.yml` of external package.

    Args:
        package_path (str): Path of external package.
        cfg_path (str): Name of experiment config.

    Returns:
        dict: Meta information of target experiment.
    """
    meta_index_path = osp.join(package_path, '.mim', 'model-index.yml')
    meta_index = load(meta_index_path)
    cfg_dict = dict()
    for meta_path in meta_index['Import']:
        meta_path = osp.join(package_path, '.mim', meta_path)
        cfg_meta = load(meta_path)
        for model_cfg in cfg_meta['Models']:
            if 'Config' not in model_cfg:
                warnings.warn(f'There is not `Config` define in {model_cfg}')
                continue
            cfg_name = model_cfg['Config'].partition('/')[-1]
            # Some config could have multiple weights, we only pick the
            # first one.
            if cfg_name in cfg_dict:
                continue
            cfg_dict[cfg_name] = model_cfg
    if cfg_path not in cfg_dict:
        raise ValueError(f'Expected configs: {cfg_dict.keys()}, but got '
                         f'{cfg_path}')
    return cfg_dict[cfg_path]


def _get_external_cfg_path(package_path: str, cfg_file: str) -> str:
    """Get config path of external package.

    Args:
        package_path (str): Path of external package.
        cfg_file (str): Name of experiment config.

    Returns:
        str: Absolute config path from external package.
    """
    cfg_file = cfg_file.split('.')[0]
    model_cfg = _get_cfg_metainfo(package_path, cfg_file)
    cfg_path = osp.join(package_path, model_cfg['Config'])
    check_file_exist(cfg_path)
    return cfg_path


def _get_external_cfg_base_path(package_path: str, cfg_name: str) -> str:
    """Get base config path of external package.

    Args:
        package_path (str): Path of external package.
        cfg_name (str): External relative config path with 'package::'.

    Returns:
        str: Absolute config path from external package.
    """
    cfg_path = osp.join(package_path, ".mim", "configs", cfg_name)
    check_file_exist(cfg_path)
    return cfg_path


def add_args(parser: ArgumentParser, cfg: dict, prefix: str = "") -> ArgumentParser:
    """Add config fields into argument parser.

    Args:
        parser (ArgumentParser): Argument parser.
        cfg (dict): Config dictionary.
        prefix (str, optional): Prefix of parser argument.
            Defaults to ''.

    Returns:
        ArgumentParser: Argument parser containing config fields.
    """
    for k, v in cfg.items():
        if isinstance(v, str):
            parser.add_argument("--" + prefix + k)
        elif isinstance(v, bool):
            parser.add_argument("--" + prefix + k, action="store_true")
        elif isinstance(v, int):
            parser.add_argument("--" + prefix + k, type=int)
        elif isinstance(v, float):
            parser.add_argument("--" + prefix + k, type=float)
        elif isinstance(v, dict):
            add_args(parser, v, prefix + k + ".")
        elif isinstance(v, abc.Iterable):
            parser.add_argument("--" + prefix + k, type=type(next(iter(v))), nargs="+")
        else:
            print_log(
                f"cannot parse key {prefix + k} of type {type(v)}", logger="current"
            )
    return parser


def dump(
    obj, file=None, file_format=None, file_client_args=None, backend_args=None, **kwargs
):
    """Dump data to json/yaml/pickle strings or files.

    This method provides a unified api for dumping data as strings or to files,
    and also supports custom arguments for each file format.

    ``dump`` supports dumping data as strings or to files which is saved to
    different backends.

    Args:
        obj (any): The python object to be dumped.
        file (str or :obj:`Path` or file-like object, optional): If not
            specified, then the object is dumped to a str, otherwise to a file
            specified by the filename or file-like object.
        file_format (str, optional): Same as :func:`load`.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
        backend_args (dict, optional): Arguments to instantiate the
            prefix of uri corresponding backend. Defaults to None.
            New in v0.2.0.

    Examples:
        >>> dump('hello world', '/path/of/your/file')  # disk
        >>> dump('hello world', 's3://path/of/your/file')  # ceph or petrel

    Returns:
        bool: True for success, False otherwise.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None:
        if is_str(file):
            file_format = file.split(".")[-1]
        elif file is None:
            raise ValueError("file_format must be specified since file is None")
    if file_format not in file_handlers:
        raise TypeError(f"Unsupported format: {file_format}")

    if file_client_args is not None:
        warnings.warn(
            '"file_client_args" will be deprecated in future. '
            'Please use "backend_args" instead',
            DeprecationWarning,
        )
        if backend_args is not None:
            raise ValueError(
                '"file_client_args" and "backend_args" cannot be set at the '
                "same time."
            )

    handler = file_handlers[file_format]
    if file is None:
        return handler.dump_to_str(obj, **kwargs)
    elif is_str(file):
        if file_client_args is not None:
            file_client = FileClient.infer_client(file_client_args, file)
            file_backend = file_client
        else:
            file_backend = get_file_backend(file, backend_args=backend_args)

        if handler.str_like:
            with StringIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                file_backend.put_text(f.getvalue(), file)
        else:
            with BytesIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                file_backend.put(f.getvalue(), file)
    elif hasattr(file, "write"):
        handler.dump_to_fileobj(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


class Config:
    """A facility for config and config files.

    It supports common file formats as configs: python/json/yaml.
    ``Config.fromfile`` can parse a dictionary from a config file, then
    build a ``Config`` instance with the dictionary.
    The interface is the same as a dict object and also allows access config
    values as attributes.

    Args:
        cfg_dict (dict, optional): A config dictionary. Defaults to None.
        cfg_text (str, optional): Text of config. Defaults to None.
        filename (str or Path, optional): Name of config file.
            Defaults to None.

    Examples:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/config/a.py')
        >>> cfg.filename
        "/home/username/projects/mmengine/tests/data/config/a.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/username/projects/mmengine/tests/data/config/a.py]
        :"
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"
    """

    def __init__(
        self,
        cfg_dict: dict = None,
        cfg_text: Optional[str] = None,
        filename: Optional[Union[str, Path]] = None,
        env_variables: Optional[dict] = None,
    ):
        filename = str(filename) if isinstance(filename, Path) else filename
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError("cfg_dict must be a dict, but " f"got {type(cfg_dict)}")
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f"{key} is reserved for config file")

        super().__setattr__("_cfg_dict", ConfigDict(cfg_dict))
        super().__setattr__("_filename", filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, encoding="utf-8") as f:
                text = f.read()
        else:
            text = ""
        super().__setattr__("_text", text)
        if env_variables is None:
            env_variables = dict()
        super().__setattr__("_env_variables", env_variables)

    @staticmethod
    def fromfile(
        filename: Union[str, Path],
        use_predefined_variables: bool = True,
        import_custom_modules: bool = True,
        use_environment_variables: bool = True,
    ) -> "Config":
        """Build a Config instance from config file.

        Args:
            filename (str or Path): Name of config file.
            use_predefined_variables (bool, optional): Whether to use
                predefined variables. Defaults to True.
            import_custom_modules (bool, optional): Whether to support
                importing custom modules in config. Defaults to True.

        Returns:
            Config: Config instance built from config file.
        """
        filename = str(filename) if isinstance(filename, Path) else filename
        cfg_dict, cfg_text, env_variables = Config._file2dict(
            filename, use_predefined_variables, use_environment_variables
        )
        if import_custom_modules and cfg_dict.get("custom_imports", None):
            try:
                import_modules_from_strings(**cfg_dict["custom_imports"])
            except ImportError as e:
                raise ImportError("Failed to custom import!") from e
        return Config(
            cfg_dict, cfg_text=cfg_text, filename=filename, env_variables=env_variables
        )

    @staticmethod
    def fromstring(cfg_str: str, file_format: str) -> "Config":
        """Build a Config instance from config text.

        Args:
            cfg_str (str): Config text.
            file_format (str): Config file format corresponding to the
               config str. Only py/yml/yaml/json type are supported now!

        Returns:
            Config: Config object generated from ``cfg_str``.
        """
        if file_format not in [".py", ".json", ".yaml", ".yml"]:
            raise OSError("Only py/yml/yaml/json type are supported now!")
        if file_format != ".py" and "dict(" in cfg_str:
            # check if users specify a wrong suffix for python
            warnings.warn('Please check "file_format", the file format may be .py')

        # A temporary file can not be opened a second time on Windows.
        # See https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile for more details. # noqa
        # `temp_file` is opened first in `tempfile.NamedTemporaryFile` and
        #  second in `Config.from_file`.
        # In addition, a named temporary file will be removed after closed.
        # As a workaround we set `delete=False` and close the temporary file
        # before opening again.

        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=file_format, delete=False
        ) as temp_file:
            temp_file.write(cfg_str)

        cfg = Config.fromfile(temp_file.name)
        os.remove(temp_file.name)  # manually delete the temporary file
        return cfg

    @staticmethod
    def _validate_py_syntax(filename: str):
        """Validate syntax of python config.

        Args:
            filename (str): Filename of python config file.
        """
        with open(filename, encoding="utf-8") as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError(
                "There are syntax errors in config " f"file {filename}: {e}"
            )

    @staticmethod
    def _substitute_predefined_vars(filename: str, temp_config_name: str):
        """Substitute predefined variables in config with actual values.

        Sometimes we want some variables in the config to be related to the
        current path or file name, etc.

        Here is an example of a typical usage scenario. When training a model,
        we define a working directory in the config that save the models and
        logs. For different configs, we expect to define different working
        directories. A common way for users is to use the config file name
        directly as part of the working directory name, e.g. for the config
        ``config_setting1.py``, the working directory is
        ``. /work_dir/config_setting1``.

        This can be easily achieved using predefined variables, which can be
        written in the config `config_setting1.py` as follows

        .. code-block:: python

           work_dir = '. /work_dir/{{ fileBasenameNoExtension }}'


        Here `{{ fileBasenameNoExtension }}` indicates the file name of the
        config (without the extension), and when the config class reads the
        config file, it will automatically parse this double-bracketed string
        to the corresponding actual value.

        .. code-block:: python

           cfg = Config.fromfile('. /config_setting1.py')
           cfg.work_dir # ". /work_dir/config_setting1"


        For details, Please refer to docs/zh_cn/advanced_tutorials/config.md .

        Args:
            filename (str): Filename of config.
            temp_config_name (str): Temporary filename to save substituted
                config.
        """
        file_dirname = osp.dirname(filename)
        file_basename = osp.basename(filename)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(filename)[1]
        support_templates = dict(
            fileDirname=file_dirname,
            fileBasename=file_basename,
            fileBasenameNoExtension=file_basename_no_extension,
            fileExtname=file_extname,
        )
        with open(filename, encoding="utf-8") as f:
            config_file = f.read()
        for key, value in support_templates.items():
            regexp = r"\{\{\s*" + str(key) + r"\s*\}\}"
            value = value.replace("\\", "/")
            config_file = re.sub(regexp, value, config_file)
        with open(temp_config_name, "w", encoding="utf-8") as tmp_config_file:
            tmp_config_file.write(config_file)

    @staticmethod
    def _substitute_env_variables(filename: str, temp_config_name: str):
        """Substitute environment variables in config with actual values.

        Sometimes, we want to change some items in the config with environment
        variables. For examples, we expect to change dataset root by setting
        ``DATASET_ROOT=/dataset/root/path`` in the command line. This can be
        easily achieved by writing lines in the config as follows

        .. code-block:: python

           data_root = '{{$DATASET_ROOT:/default/dataset}}/images'


        Here, ``{{$DATASET_ROOT:/default/dataset}}`` indicates using the
        environment variable ``DATASET_ROOT`` to replace the part between
        ``{{}}``. If the ``DATASET_ROOT`` is not set, the default value
        ``/default/dataset`` will be used.

        Environment variables not only can replace items in the string, they
        can also substitute other types of data in config. In this situation,
        we can write the config as below

        .. code-block:: python

           model = dict(
               bbox_head = dict(num_classes={{'$NUM_CLASSES:80'}}))


        For details, Please refer to docs/zh_cn/tutorials/config.md .

        Args:
            filename (str): Filename of config.
            temp_config_name (str): Temporary filename to save substituted
                config.
        """
        with open(filename, encoding="utf-8") as f:
            config_file = f.read()
        regexp = r"\{\{[\'\"]?\s*\$(\w+)\s*\:\s*(\S*?)\s*[\'\"]?\}\}"
        keys = re.findall(regexp, config_file)
        env_variables = dict()
        for var_name, value in keys:
            regexp = (
                r"\{\{[\'\"]?\s*\$" + var_name + r"\s*\:\s*" + value + r"\s*[\'\"]?\}\}"
            )
            if var_name in os.environ:
                value = os.environ[var_name]
                env_variables[var_name] = value
                print_log(
                    f"Using env variable `{var_name}` with value of "
                    f"{value} to replace item in config.",
                    logger="current",
                )
            if not value:
                raise KeyError(
                    f"`{var_name}` cannot be found in `os.environ`."
                    f" Please set `{var_name}` in environment or "
                    "give a default value."
                )
            config_file = re.sub(regexp, value, config_file)

        with open(temp_config_name, "w", encoding="utf-8") as tmp_config_file:
            tmp_config_file.write(config_file)
        return env_variables

    @staticmethod
    def _pre_substitute_base_vars(filename: str, temp_config_name: str) -> dict:
        """Preceding step for substituting variables in base config with actual
        value.

        Args:
            filename (str): Filename of config.
            temp_config_name (str): Temporary filename to save substituted
                config.

        Returns:
            dict: A dictionary contains variables in base config.
        """
        with open(filename, encoding="utf-8") as f:
            config_file = f.read()
        base_var_dict = {}
        regexp = r"\{\{\s*" + BASE_KEY + r"\.([\w\.]+)\s*\}\}"
        base_vars = set(re.findall(regexp, config_file))
        for base_var in base_vars:
            randstr = f"_{base_var}_{uuid.uuid4().hex.lower()[:6]}"
            base_var_dict[randstr] = base_var
            regexp = r"\{\{\s*" + BASE_KEY + r"\." + base_var + r"\s*\}\}"
            config_file = re.sub(regexp, f'"{randstr}"', config_file)
        with open(temp_config_name, "w", encoding="utf-8") as tmp_config_file:
            tmp_config_file.write(config_file)
        return base_var_dict

    @staticmethod
    def _substitute_base_vars(cfg: Any, base_var_dict: dict, base_cfg: dict) -> Any:
        """Substitute base variables from strings to their actual values.

        Args:
            Any : Config dictionary.
            base_var_dict (dict): A dictionary contains variables in base
                config.
            base_cfg (dict): Base config dictionary.

        Returns:
            Any : A dictionary with origin base variables
                substituted with actual values.
        """
        cfg = copy.deepcopy(cfg)

        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if isinstance(v, str) and v in base_var_dict:
                    new_v = base_cfg
                    for new_k in base_var_dict[v].split("."):
                        new_v = new_v[new_k]
                    cfg[k] = new_v
                elif isinstance(v, (list, tuple, dict)):
                    cfg[k] = Config._substitute_base_vars(v, base_var_dict, base_cfg)
        elif isinstance(cfg, tuple):
            cfg = tuple(
                Config._substitute_base_vars(c, base_var_dict, base_cfg) for c in cfg
            )
        elif isinstance(cfg, list):
            cfg = [
                Config._substitute_base_vars(c, base_var_dict, base_cfg) for c in cfg
            ]
        elif isinstance(cfg, str) and cfg in base_var_dict:
            new_v = base_cfg
            for new_k in base_var_dict[cfg].split("."):
                new_v = new_v[new_k]
            cfg = new_v

        return cfg

    @staticmethod
    def _file2dict(
        filename: str,
        use_predefined_variables: bool = True,
        use_environment_variables: bool = True,
    ) -> Tuple[dict, str, dict]:
        """Transform file to variables dictionary.

        Args:
            filename (str): Name of config file.
            use_predefined_variables (bool, optional): Whether to use
                predefined variables. Defaults to True.

        Returns:
            Tuple[dict, str]: Variables dictionary and text of Config.
        """
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in [".py", ".json", ".yaml", ".yml"]:
            raise OSError("Only py/yml/yaml/json type are supported now!")

        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(
                dir=temp_config_dir, suffix=fileExtname
            )
            if platform.system() == "Windows":
                temp_config_file.close()

            # Substitute predefined variables
            if use_predefined_variables:
                Config._substitute_predefined_vars(filename, temp_config_file.name)
            else:
                shutil.copyfile(filename, temp_config_file.name)
            # Substitute environment variables
            env_variables = dict()
            if use_environment_variables:
                env_variables = Config._substitute_env_variables(
                    temp_config_file.name, temp_config_file.name
                )
            # Substitute base variables from placeholders to strings
            base_var_dict = Config._pre_substitute_base_vars(
                temp_config_file.name, temp_config_file.name
            )

            # Handle base files
            base_cfg_dict = ConfigDict()
            cfg_text_list = list()
            for base_cfg_path in Config._get_base_files(temp_config_file.name):
                base_cfg_path, scope = Config._get_cfg_path(base_cfg_path, filename)
                _cfg_dict, _cfg_text, _env_variables = Config._file2dict(
                    filename=base_cfg_path,
                    use_predefined_variables=use_predefined_variables,
                    use_environment_variables=use_environment_variables,
                )
                cfg_text_list.append(_cfg_text)
                env_variables.update(_env_variables)
                duplicate_keys = base_cfg_dict.keys() & _cfg_dict.keys()
                if len(duplicate_keys) > 0:
                    raise KeyError(
                        "Duplicate key is not allowed among bases. "
                        f"Duplicate keys: {duplicate_keys}"
                    )

                # _dict_to_config_dict will do the following things:
                # 1. Recursively converts ``dict`` to :obj:`ConfigDict`.
                # 2. Set `_scope_` for the outer dict variable for the base
                # config.
                # 3. Set `scope` attribute for each base variable. Different
                # from `_scope_`， `scope` is not a key of base dict,
                # `scope` attribute will be parsed to key `_scope_` by
                # function `_parse_scope` only if the base variable is
                # accessed by the current config.
                _cfg_dict = Config._dict_to_config_dict(_cfg_dict, scope)
                base_cfg_dict.update(_cfg_dict)

            if filename.endswith(".py"):
                with open(temp_config_file.name, encoding="utf-8") as f:
                    codes = ast.parse(f.read())
                    codes = RemoveAssignFromAST(BASE_KEY).visit(codes)
                codeobj = compile(codes, "", mode="exec")
                # Support load global variable in nested function of the
                # config.
                global_locals_var = {"_base_": base_cfg_dict}
                ori_keys = set(global_locals_var.keys())
                eval(codeobj, global_locals_var, global_locals_var)
                cfg_dict = {
                    key: value
                    for key, value in global_locals_var.items()
                    if (key not in ori_keys and not key.startswith("__"))
                }
            elif filename.endswith((".yml", ".yaml", ".json")):
                cfg_dict = load(temp_config_file.name)
            # close temp file
            for key, value in list(cfg_dict.items()):
                if isinstance(value, (types.FunctionType, types.ModuleType)):
                    cfg_dict.pop(key)
            temp_config_file.close()

            # If the current config accesses a base variable of base
            # configs, The ``scope`` attribute of corresponding variable
            # will be converted to the `_scope_`.
            Config._parse_scope(cfg_dict)

        # check deprecation information
        if DEPRECATION_KEY in cfg_dict:
            deprecation_info = cfg_dict.pop(DEPRECATION_KEY)
            warning_msg = (
                f"The config file {filename} will be deprecated " "in the future."
            )
            if "expected" in deprecation_info:
                warning_msg += f' Please use {deprecation_info["expected"]} ' "instead."
            if "reference" in deprecation_info:
                warning_msg += (
                    " More information can be found at "
                    f'{deprecation_info["reference"]}'
                )
            warnings.warn(warning_msg, DeprecationWarning)

        cfg_text = filename + "\n"
        with open(filename, encoding="utf-8") as f:
            # Setting encoding explicitly to resolve coding issue on windows
            cfg_text += f.read()

        # Substitute base variables from strings to their actual values
        cfg_dict = Config._substitute_base_vars(cfg_dict, base_var_dict, base_cfg_dict)
        cfg_dict.pop(BASE_KEY, None)

        cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
        cfg_dict = {k: v for k, v in cfg_dict.items() if not k.startswith("__")}

        # merge cfg_text
        cfg_text_list.append(cfg_text)
        cfg_text = "\n".join(cfg_text_list)

        return cfg_dict, cfg_text, env_variables

    @staticmethod
    def _dict_to_config_dict(cfg: dict, scope: Optional[str] = None, has_scope=True):
        """Recursively converts ``dict`` to :obj:`ConfigDict`.

        Args:
            cfg (dict): Config dict.
            scope (str, optional): Scope of instance.
            has_scope (bool): Whether to add `_scope_` key to config dict.

        Returns:
            ConfigDict: Converted dict.
        """
        # Only the outer dict with key `type` should have the key `_scope_`.
        if isinstance(cfg, dict):
            if has_scope and "type" in cfg:
                has_scope = False
                if scope is not None and cfg.get("_scope_", None) is None:
                    cfg._scope_ = scope  # type: ignore
            cfg = ConfigDict(cfg)
            dict.__setattr__(cfg, "scope", scope)
            for key, value in cfg.items():
                cfg[key] = Config._dict_to_config_dict(
                    value, scope=scope, has_scope=has_scope
                )
        elif isinstance(cfg, tuple):
            cfg = tuple(
                Config._dict_to_config_dict(_cfg, scope, has_scope=has_scope)
                for _cfg in cfg
            )
        elif isinstance(cfg, list):
            cfg = [
                Config._dict_to_config_dict(_cfg, scope, has_scope=has_scope)
                for _cfg in cfg
            ]
        return cfg

    @staticmethod
    def _parse_scope(cfg: dict) -> None:
        """Adds ``_scope_`` to :obj:`ConfigDict` instance, which means a base
        variable.

        If the config dict already has the scope, scope will not be
        overwritten.

        Args:
            cfg (dict): Config needs to be parsed with scope.
        """
        if isinstance(cfg, ConfigDict):
            cfg._scope_ = cfg.scope
        elif isinstance(cfg, (tuple, list)):
            [Config._parse_scope(value) for value in cfg]
        else:
            return

    @staticmethod
    def _get_base_files(filename: str) -> list:
        """Get the base config file.

        Args:
            filename (str): The config file.

        Raises:
            TypeError: Name of config file.

        Returns:
            list: A list of base config.
        """
        file_format = osp.splitext(filename)[1]
        if file_format == ".py":
            Config._validate_py_syntax(filename)
            with open(filename, encoding="utf-8") as f:
                codes = ast.parse(f.read()).body

                def is_base_line(c):
                    return (
                        isinstance(c, ast.Assign)
                        and isinstance(c.targets[0], ast.Name)
                        and c.targets[0].id == BASE_KEY
                    )

                base_code = next((c for c in codes if is_base_line(c)), None)
                if base_code is not None:
                    base_code = ast.Expression(  # type: ignore
                        body=base_code.value
                    )  # type: ignore
                    base_files = eval(compile(base_code, "", mode="eval"))
                else:
                    base_files = []
        elif file_format in (".yml", ".yaml", ".json"):
            # import mmengine

            cfg_dict = load(filename) #mmengine
            base_files = cfg_dict.get(BASE_KEY, [])
        else:
            raise TypeError(
                "The config type should be py, json, yaml or "
                f"yml, but got {file_format}"
            )
        base_files = base_files if isinstance(base_files, list) else [base_files]
        return base_files

    @staticmethod
    def _get_cfg_path(cfg_path: str, filename: str) -> Tuple[str, Optional[str]]:
        """Get the config path from the current or external package.

        Args:
            cfg_path (str): Relative path of config.
            filename (str): The config file being parsed.

        Returns:
            Tuple[str, str or None]: Path and scope of config. If the config
            is not an external config, the scope will be `None`.
        """
        if "::" in cfg_path:
            # `cfg_path` startswith '::' means an external config path.
            # Get package name and relative config path.
            scope = cfg_path.partition("::")[0]
            package, cfg_path = _get_package_and_cfg_path(cfg_path)

            if not is_installed(package):
                raise ModuleNotFoundError(
                    f"{package} is not installed, please install {package} " f"manually"
                )

            # Get installed package path.
            package_path = get_installed_path(package)
            try:
                # Get config path from meta file.
                cfg_path = _get_external_cfg_path(package_path, cfg_path)
            except ValueError:
                # Since base config does not have a metafile, it should be
                # concatenated with package path and relative config path.
                cfg_path = _get_external_cfg_base_path(package_path, cfg_path)
            except FileNotFoundError as e:
                raise e
            return cfg_path, scope
        else:
            # Get local config path.
            cfg_dir = osp.dirname(filename)
            cfg_path = osp.join(cfg_dir, cfg_path)
            return cfg_path, None

    @staticmethod
    def _merge_a_into_b(a: dict, b: dict, allow_list_keys: bool = False) -> dict:
        """merge dict ``a`` into dict ``b`` (non-inplace).

        Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid
        in-place modifications.

        Args:
            a (dict): The source dict to be merged into ``b``.
            b (dict): The origin dict to be fetch keys from ``a``.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in source ``a`` and will replace the element of the
              corresponding index in b if b is a list. Defaults to False.

        Returns:
            dict: The modified dict of ``b`` using ``a``.

        Examples:
            # Normally merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # Delete b first and merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(_delete_=True, a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # b is a list
            >>> Config._merge_a_into_b(
            ...     {'0': dict(a=2)}, [dict(a=1), dict(b=2)], True)
            [{'a': 2}, {'b': 2}]
        """
        b = b.copy()
        for k, v in a.items():
            if allow_list_keys and k.isdigit() and isinstance(b, list):
                k = int(k)
                if len(b) <= k:
                    raise KeyError(f"Index {k} exceeds the length of list {b}")
                b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
            elif isinstance(v, dict):
                if k in b and not v.pop(DELETE_KEY, False):
                    allowed_types: Union[Tuple, type] = (
                        (dict, list) if allow_list_keys else dict
                    )
                    if not isinstance(b[k], allowed_types):
                        raise TypeError(
                            f"{k}={v} in child config cannot inherit from "
                            f"base because {k} is a dict in the child config "
                            f"but is of type {type(b[k])} in base config. "
                            f"You may set `{DELETE_KEY}=True` to ignore the "
                            f"base config."
                        )
                    b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
                else:
                    b[k] = ConfigDict(v)
            else:
                b[k] = v
        return b

    @staticmethod
    def auto_argparser(description=None):
        """Generate argparser from config file automatically (experimental)"""
        partial_parser = ArgumentParser(description=description)
        partial_parser.add_argument("config", help="config file path")
        cfg_file = partial_parser.parse_known_args()[0].config
        cfg = Config.fromfile(cfg_file)
        parser = ArgumentParser(description=description)
        parser.add_argument("config", help="config file path")
        add_args(parser, cfg)
        return parser, cfg

    @property
    def filename(self) -> str:
        """get file name of config."""
        return self._filename

    @property
    def text(self) -> str:
        """get config text."""
        return self._text

    @property
    def env_variables(self) -> dict:
        """get used environment variables."""
        return self._env_variables

    @property
    def pretty_text(self) -> str:
        """get formatted python config text."""

        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = repr(v)
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f"{k_str}: {v_str}"
            else:
                attr_str = f"{str(k)}={v_str}"
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list(k, v, use_mapping=False):
            # check if all items in the list are dict
            if all(isinstance(_, dict) for _ in v):
                v_str = "[\n"
                v_str += "\n".join(
                    f"dict({_indent(_format_dict(v_), indent)})," for v_ in v
                ).rstrip(",")
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f"{k_str}: {v_str}"
                else:
                    attr_str = f"{str(k)}={v_str}"
                attr_str = _indent(attr_str, indent) + "]"
            else:
                attr_str = _format_basic_types(k, v, use_mapping)
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= not str(key_name).isidentifier()
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ""
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += "{"
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = "" if outest_level or is_last else ","
                if isinstance(v, dict):
                    v_str = "\n" + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f"{k_str}: dict({v_str}"
                    else:
                        attr_str = f"{str(k)}=dict({v_str}"
                    attr_str = _indent(attr_str, indent) + ")" + end
                elif isinstance(v, list):
                    attr_str = _format_list(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += "\n".join(s)
            if use_mapping:
                r += "}"
            return r

        cfg_dict = self._cfg_dict.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        # copied from setup.cfg
        yapf_style = dict(
            based_on_style="pep8",
            blank_line_before_nested_class_or_def=True,
            split_before_expression_after_opening_paren=True,
        )
        text, _ = FormatCode(text, style_config=yapf_style, verify=True)

        return text

    def __repr__(self):
        return f"Config (path: {self.filename}): {self._cfg_dict.__repr__()}"

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def __getstate__(self) -> Tuple[dict, Optional[str], Optional[str], dict]:
        return (self._cfg_dict, self._filename, self._text, self._env_variables)

    def __deepcopy__(self, memo):
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other

        for key, value in self.__dict__.items():
            super(Config, other).__setattr__(key, copy.deepcopy(value, memo))

        return other

    def __copy__(self):
        cls = self.__class__
        other = cls.__new__(cls)
        other.__dict__.update(self.__dict__)

        return other

    def __setstate__(self, state: Tuple[dict, Optional[str], Optional[str], dict]):
        _cfg_dict, _filename, _text, _env_variables = state
        super().__setattr__("_cfg_dict", _cfg_dict)
        super().__setattr__("_filename", _filename)
        super().__setattr__("_text", _text)
        super().__setattr__("_text", _env_variables)

    def dump(self, file: Optional[Union[str, Path]] = None):
        """Dump config to file or return config text.

        Args:
            file (str or Path, optional): If not specified, then the object
            is dumped to a str, otherwise to a file specified by the filename.
            Defaults to None.

        Returns:
            str or None: Config text.
        """
        file = str(file) if isinstance(file, Path) else file
        cfg_dict = super().__getattribute__("_cfg_dict").to_dict()
        if file is None:
            if self.filename is None or self.filename.endswith(".py"):
                return self.pretty_text
            else:
                file_format = self.filename.split(".")[-1]
                return dump(cfg_dict, file_format=file_format)
        elif file.endswith(".py"):
            with open(file, "w", encoding="utf-8") as f:
                f.write(self.pretty_text)
        else:
            file_format = file.split(".")[-1]
            return dump(cfg_dict, file=file, file_format=file_format)

    def merge_from_dict(self, options: dict, allow_list_keys: bool = True) -> None:
        """Merge list into cfg_dict.

        Merge the dict parsed by MultipleKVAction into this cfg.

        Args:
            options (dict): dict of configs to merge from.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
                are allowed in ``options`` and will replace the element of the
                corresponding index in the config if the config is a list.
                Defaults to True.

        Examples:
            >>> from mmengine import Config
            >>> #  Merge dictionary element
            >>> options = {'model.backbone.depth': 50, 'model.backbone.with_cp': True}
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet'))))
            >>> cfg.merge_from_dict(options)
            >>> cfg._cfg_dict
            {'model': {'backbone': {'type': 'ResNet', 'depth': 50, 'with_cp': True}}}
            >>> # Merge list element
            >>> cfg = Config(
            >>>     dict(pipeline=[dict(type='LoadImage'),
            >>>                    dict(type='LoadAnnotations')]))
            >>> options = dict(pipeline={'0': dict(type='SelfLoadImage')})
            >>> cfg.merge_from_dict(options, allow_list_keys=True)
            >>> cfg._cfg_dict
            {'pipeline': [{'type': 'SelfLoadImage'}, {'type': 'LoadAnnotations'}]}
        """  # noqa: E501
        option_cfg_dict: dict = {}
        for full_key, v in options.items():
            d = option_cfg_dict
            key_list = full_key.split(".")
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict = super().__getattribute__("_cfg_dict")
        super().__setattr__(
            "_cfg_dict",
            Config._merge_a_into_b(
                option_cfg_dict, cfg_dict, allow_list_keys=allow_list_keys
            ),
        )


class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val: str) -> Union[int, float, bool, Any]:
        """parse int/float/bool value in the string."""
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ["true", "false"]:
            return True if val.lower() == "true" else False
        if val == "None":
            return None
        return val

    @staticmethod
    def _parse_iterable(val: str) -> Union[list, tuple, Any]:
        """Parse iterable values in the string.

        All elements inside '()' or '[]' are treated as iterable values.

        Args:
            val (str): Value string.

        Returns:
            list | tuple | Any: The expanded list or tuple from the string,
            or single value if no iterable values are found.

        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.

            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count("(") == string.count(")")) and (
                string.count("[") == string.count("]")
            ), f"Imbalanced brackets exist in {string}"
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if (
                    (char == ",")
                    and (pre.count("(") == pre.count(")"))
                    and (pre.count("[") == pre.count("]"))
                ):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip("'\"").replace(" ", "")
        is_tuple = False
        if val.startswith("(") and val.endswith(")"):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith("[") and val.endswith("]"):
            val = val[1:-1]
        elif "," not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1 :]

        if is_tuple:
            return tuple(values)

        return values

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: Union[str, Sequence[Any], None],
        option_string: str = None,
    ):
        """Parse Variables in string and add them into argparser.

        Args:
            parser (ArgumentParser): Argument parser.
            namespace (Namespace): Argument namespace.
            values (Union[str, Sequence[Any], None]): Argument string.
            option_string (list[str], optional): Option string.
                Defaults to None.
        """
        # Copied behavior from `argparse._ExtendAction`.
        options = copy.copy(getattr(namespace, self.dest, None) or {})
        if values is not None:
            for kv in values:
                key, val = kv.split("=", maxsplit=1)
                options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)


class DefaultScope(ManagerMixin):
    """Scope of current task used to reset the current registry, which can be
    accessed globally.

    Consider the case of resetting the current ``Registry`` by
    ``default_scope`` in the internal module which cannot access runner
    directly, it is difficult to get the ``default_scope`` defined in
    ``Runner``. However, if ``Runner`` created ``DefaultScope`` instance
    by given ``default_scope``, the internal module can get
    ``default_scope`` by ``DefaultScope.get_current_instance`` everywhere.

    Args:
        name (str): Name of default scope for global access.
        scope_name (str): Scope of current task.

    Examples:
        >>> from mmengine.model import MODELS
        >>> # Define default scope in runner.
        >>> DefaultScope.get_instance('task', scope_name='mmdet')
        >>> # Get default scope globally.
        >>> scope_name = DefaultScope.get_instance('task').scope_name
    """

    def __init__(self, name: str, scope_name: str):
        super().__init__(name)
        assert isinstance(
            scope_name, str
        ), f"scope_name should be a string, but got {scope_name}"
        self._scope_name = scope_name

    @property
    def scope_name(self) -> str:
        """
        Returns:
            str: Get current scope.
        """
        return self._scope_name

    @classmethod
    def get_current_instance(cls) -> Optional["DefaultScope"]:
        """Get latest created default scope.

        Since default_scope is an optional argument for ``Registry.build``.
        ``get_current_instance`` should return ``None`` if there is no
        ``DefaultScope`` created.

        Examples:
            >>> default_scope = DefaultScope.get_current_instance()
            >>> # There is no `DefaultScope` created yet,
            >>> # `get_current_instance` return `None`.
            >>> default_scope = DefaultScope.get_instance(
            >>>     'instance_name', scope_name='mmengine')
            >>> default_scope.scope_name
            mmengine
            >>> default_scope = DefaultScope.get_current_instance()
            >>> default_scope.scope_name
            mmengine

        Returns:
            Optional[DefaultScope]: Return None If there has not been
            ``DefaultScope`` instance created yet, otherwise return the
            latest created DefaultScope instance.
        """
        _accquire_lock()
        if cls._instance_dict:
            instance = super().get_current_instance()
        else:
            instance = None
        _release_lock()
        return instance

    @classmethod
    @contextmanager
    def overwrite_default_scope(cls, scope_name: Optional[str]) -> Generator:
        """overwrite the current default scope with `scope_name`"""
        if scope_name is None:
            yield
        else:
            tmp = copy.deepcopy(cls._instance_dict)
            # To avoid create an instance with the same name.
            time.sleep(1e-6)
            cls.get_instance(f"overwrite-{time.time()}", scope_name=scope_name)
            try:
                yield
            finally:
                cls._instance_dict = tmp


def init_default_scope(scope: str) -> None:
    """Initialize the given default scope.

    Args:
        scope (str): The name of the default scope.
    """
    never_created = (
        DefaultScope.get_current_instance() is None
        or not DefaultScope.check_instance_created(scope)
    )
    if never_created:
        DefaultScope.get_instance(scope, scope_name=scope)
        return
    current_scope = DefaultScope.get_current_instance()  # type: ignore
    if current_scope.scope_name != scope:  # type: ignore
        print_log(
            "The current default scope "  # type: ignore
            f'"{current_scope.scope_name}" is not "{scope}", '
            "`init_default_scope` will force set the current"
            f'default scope to "{scope}".',
            logger="current",
            level=logging.WARNING,
        )
        # avoid name conflict
        new_instance_name = f"{scope}-{datetime.datetime.now()}"
        DefaultScope.get_instance(new_instance_name, scope_name=scope)


def build_from_cfg(
    cfg: Union[dict, ConfigDict, Config],
    registry,
    default_args: Optional[Union[dict, ConfigDict, Config]] = None,
) -> Any:
    """Build a module from config dict when it is a class configuration, or
    call a function from config dict when it is a function configuration.

    If the global variable default scope (:obj:`DefaultScope`) exists,
    :meth:`build` will firstly get the responding registry and then call
    its own :meth:`build`.

    At least one of the ``cfg`` and ``default_args`` contains the key "type",
    which should be either str or class. If they all contain it, the key
    in ``cfg`` will be used because ``cfg`` has a high priority than
    ``default_args`` that means if a key exists in both of them, the value of
    the key will be ``cfg[key]``. They will be merged first and the key "type"
    will be popped up and the remaining keys will be used as initialization
    arguments.

    Examples:
        >>> from mmengine import Registry, build_from_cfg
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     def __init__(self, depth, stages=4):
        >>>         self.depth = depth
        >>>         self.stages = stages
        >>> cfg = dict(type='ResNet', depth=50)
        >>> model = build_from_cfg(cfg, MODELS)
        >>> # Returns an instantiated object
        >>> @MODELS.register_module()
        >>> def resnet50():
        >>>     pass
        >>> resnet = build_from_cfg(dict(type='resnet50'), MODELS)
        >>> # Return a result of the calling function

    Args:
        cfg (dict or ConfigDict or Config): Config dict. It should at least
            contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict or ConfigDict or Config, optional): Default
            initialization arguments. Defaults to None.

    Returns:
        object: The constructed object.
    """
    # Avoid circular import

    if not isinstance(cfg, (dict, ConfigDict, Config)):
        raise TypeError(
            f"cfg should be a dict, ConfigDict or Config, but got {type(cfg)}"
        )

    if "type" not in cfg:
        if default_args is None or "type" not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f"but got {cfg}\n{default_args}"
            )

    if not isinstance(registry, Registry):
        raise TypeError(
            "registry must be a mmengine.Registry object, " f"but got {type(registry)}"
        )

    if not (
        isinstance(default_args, (dict, ConfigDict, Config)) or default_args is None
    ):
        raise TypeError(
            "default_args should be a dict, ConfigDict, Config or None, "
            f"but got {type(default_args)}"
        )

    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    # Instance should be built under target scope, if `_scope_` is defined
    # in cfg, current default scope should switch to specified scope
    # temporarily.
    scope = args.pop("_scope_", None)
    with registry.switch_scope_and_registry(scope) as registry:
        obj_type = args.pop("type")
        if isinstance(obj_type, str):
            obj_cls = registry.get(obj_type)
            if obj_cls is None:
                raise KeyError(
                    f"{obj_type} is not in the {registry.name} registry. "
                    f"Please check whether the value of `{obj_type}` is "
                    "correct or it was registered as expected. More details "
                    "can be found at "
                    "https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#import-the-custom-module"  # noqa: E501
                )
        elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f"type must be a str or valid type, but got {type(obj_type)}"
            )

        try:
            # If `obj_cls` inherits from `ManagerMixin`, it should be
            # instantiated by `ManagerMixin.get_instance` to ensure that it
            # can be accessed globally.
            if inspect.isclass(obj_cls) and issubclass(
                obj_cls, ManagerMixin
            ):  # type: ignore
                obj = obj_cls.get_instance(**args)  # type: ignore
            else:
                obj = obj_cls(**args)  # type: ignore

            print_log(
                f"An `{obj_cls.__name__}` instance is built from "  # type: ignore # noqa: E501
                "registry, its implementation can be found in "
                f"{obj_cls.__module__}",  # type: ignore
                logger="current",
                level=logging.DEBUG,
            )
            return obj

        except Exception as e:
            # Normal TypeError does not print class name.
            cls_location = "/".join(obj_cls.__module__.split("."))  # type: ignore
            raise type(e)(
                f"class `{obj_cls.__name__}` in "  # type: ignore
                f"{cls_location}.py: {e}"
            )


def is_seq_of(
    seq: Any, expected_type: Union[Type, tuple], seq_type: Type = None
) -> bool:
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type or tuple): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Defaults to None.

    Returns:
        bool: Return True if ``seq`` is valid else False.

    Examples:
        >>> from mmengine.utils import is_seq_of
        >>> seq = ['a', 'b', 'c']
        >>> is_seq_of(seq, str)
        True
        >>> is_seq_of(seq, int)
        False
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


class Registry:
    """A registry to map strings to classes or functions.

    Registered object could be built from registry. Meanwhile, registered
    functions could be called from registry.

    Args:
        name (str): Registry name.
        build_func (callable, optional): A function to construct instance
            from Registry. :func:`build_from_cfg` is used if neither ``parent``
            or ``build_func`` is specified. If ``parent`` is specified and
            ``build_func`` is not given,  ``build_func`` will be inherited
            from ``parent``. Defaults to None.
        parent (:obj:`Registry`, optional): Parent registry. The class
            registered in children registry could be built from parent.
            Defaults to None.
        scope (str, optional): The scope of registry. It is the key to search
            for children registry. If not specified, scope will be the name of
            the package where class is defined, e.g. mmdet, mmcls, mmseg.
            Defaults to None.
        locations (list): The locations to import the modules registered
            in this registry. Defaults to [].
            New in version 0.4.0.

    Examples:
        >>> # define a registry
        >>> MODELS = Registry('models')
        >>> # registry the `ResNet` to `MODELS`
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> # build model from `MODELS`
        >>> resnet = MODELS.build(dict(type='ResNet'))
        >>> @MODELS.register_module()
        >>> def resnet50():
        >>>     pass
        >>> resnet = MODELS.build(dict(type='resnet50'))

        >>> # hierarchical registry
        >>> DETECTORS = Registry('detectors', parent=MODELS, scope='det')
        >>> @DETECTORS.register_module()
        >>> class FasterRCNN:
        >>>     pass
        >>> fasterrcnn = DETECTORS.build(dict(type='FasterRCNN'))

        >>> # add locations to enable auto import
        >>> DETECTORS = Registry('detectors', parent=MODELS,
        >>>     scope='det', locations=['det.models.detectors'])
        >>> # define this class in 'det.models.detectors'
        >>> @DETECTORS.register_module()
        >>> class MaskRCNN:
        >>>     pass
        >>> # The registry will auto import det.models.detectors.MaskRCNN
        >>> fasterrcnn = DETECTORS.build(dict(type='det.MaskRCNN'))

    More advanced usages can be found at
    https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html.
    """

    def __init__(
        self,
        name: str,
        build_func: Optional[Callable] = None,
        parent: Optional["Registry"] = None,
        scope: Optional[str] = None,
        locations: List = [],
    ):

        self._name = name
        self._module_dict: Dict[str, Type] = dict()
        self._children: Dict[str, "Registry"] = dict()
        self._locations = locations
        self._imported = False

        if scope is not None:
            assert isinstance(scope, str)
            self._scope = scope
        else:
            self._scope = self.infer_scope()

        # See https://mypy.readthedocs.io/en/stable/common_issues.html#
        # variables-vs-type-aliases for the use
        self.parent: Optional["Registry"]
        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_child(self)
            self.parent = parent
        else:
            self.parent = None

        # self.build_func will be set with the following priority:
        # 1. build_func
        # 2. parent.build_func
        # 3. build_from_cfg
        self.build_func: Callable
        if build_func is None:
            if self.parent is not None:
                self.build_func = self.parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        table = Table(title=f"Registry of {self._name}")
        table.add_column("Names", justify="left", style="cyan")
        table.add_column("Objects", justify="left", style="green")

        for name, obj in sorted(self._module_dict.items()):
            table.add_row(name, str(obj))

        console = Console()
        with console.capture() as capture:
            console.print(table, end="")

        return capture.get()

    @staticmethod
    def infer_scope() -> str:
        """Infer the scope of registry.

        The name of the package where registry is defined will be returned.

        Returns:
            str: The inferred scope name.

        Examples:
            >>> # in mmdet/models/backbone/resnet.py
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            >>> # The scope of ``ResNet`` will be ``mmdet``.
        """

        # `sys._getframe` returns the frame object that many calls below the
        # top of the stack. The call stack for `infer_scope` can be listed as
        # follow:
        # frame-0: `infer_scope` itself
        # frame-1: `__init__` of `Registry` which calls the `infer_scope`
        # frame-2: Where the `Registry(...)` is called
        module = inspect.getmodule(sys._getframe(2))
        if module is not None:
            filename = module.__name__
            split_filename = filename.split(".")
            scope = split_filename[0]
        else:
            # use "mmengine" to handle some cases which can not infer the scope
            # like initializing Registry in interactive mode
            scope = "mmengine"
            print_log(
                'set scope as "mmengine" when scope can not be inferred. You '
                'can silence this warning by passing a "scope" argument to '
                'Registry like `Registry(name, scope="toy")`',
                logger="current",
                level=logging.WARNING,
            )

        return scope

    @staticmethod
    def split_scope_key(key: str) -> Tuple[Optional[str], str]:
        """Split scope and key.

        The first scope will be split from key.

        Return:
            tuple[str | None, str]: The former element is the first scope of
            the key, which can be ``None``. The latter is the remaining key.

        Examples:
            >>> Registry.split_scope_key('mmdet.ResNet')
            'mmdet', 'ResNet'
            >>> Registry.split_scope_key('ResNet')
            None, 'ResNet'
        """
        split_index = key.find(".")
        if split_index != -1:
            return key[:split_index], key[split_index + 1 :]
        else:
            return None, key

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def children(self):
        return self._children

    @property
    def root(self):
        return self._get_root_registry()

    @contextmanager
    def switch_scope_and_registry(self, scope: Optional[str]) -> Generator:
        """Temporarily switch default scope to the target scope, and get the
        corresponding registry.

        If the registry of the corresponding scope exists, yield the
        registry, otherwise yield the current itself.

        Args:
            scope (str, optional): The target scope.

        Examples:
            >>> from mmengine.registry import Registry, DefaultScope, MODELS
            >>> import time
            >>> # External Registry
            >>> MMDET_MODELS = Registry('mmdet_model', scope='mmdet',
            >>>     parent=MODELS)
            >>> MMCLS_MODELS = Registry('mmcls_model', scope='mmcls',
            >>>     parent=MODELS)
            >>> # Local Registry
            >>> CUSTOM_MODELS = Registry('custom_model', scope='custom',
            >>>     parent=MODELS)
            >>>
            >>> # Initiate DefaultScope
            >>> DefaultScope.get_instance(f'scope_{time.time()}',
            >>>     scope_name='custom')
            >>> # Check default scope
            >>> DefaultScope.get_current_instance().scope_name
            custom
            >>> # Switch to mmcls scope and get `MMCLS_MODELS` registry.
            >>> with CUSTOM_MODELS.switch_scope_and_registry(scope='mmcls') as registry:
            >>>     DefaultScope.get_current_instance().scope_name
            mmcls
            >>>     registry.scope
            mmcls
            >>> # Nested switch scope
            >>> with CUSTOM_MODELS.switch_scope_and_registry(scope='mmdet') as mmdet_registry:
            >>>     DefaultScope.get_current_instance().scope_name
            mmdet
            >>>     mmdet_registry.scope
            mmdet
            >>>     with CUSTOM_MODELS.switch_scope_and_registry(scope='mmcls') as mmcls_registry:
            >>>         DefaultScope.get_current_instance().scope_name
            mmcls
            >>>         mmcls_registry.scope
            mmcls
            >>>
            >>> # Check switch back to original scope.
            >>> DefaultScope.get_current_instance().scope_name
            custom
        """  # noqa: E501

        # Switch to the given scope temporarily. If the corresponding registry
        # can be found in root registry, return the registry under the scope,
        # otherwise return the registry itself.
        with DefaultScope.overwrite_default_scope(scope):
            # Get the global default scope
            default_scope = DefaultScope.get_current_instance()
            # Get registry by scope
            if default_scope is not None:
                scope_name = default_scope.scope_name
                try:
                    import_module(f"{scope_name}.registry")
                except (ImportError, AttributeError, ModuleNotFoundError):
                    if scope in MODULE2PACKAGE:
                        print_log(
                            f"{scope} is not installed and its "
                            "modules will not be registered. If you "
                            "want to use modules defined in "
                            f"{scope}, Please install {scope} by "
                            f"`pip install {MODULE2PACKAGE[scope]}.",
                            logger="current",
                            level=logging.WARNING,
                        )
                    else:
                        print_log(
                            f"Failed to import `{scope}.registry` "
                            f"make sure the registry.py exists in `{scope}` "
                            "package.",
                            logger="current",
                            level=logging.WARNING,
                        )
                root = self._get_root_registry()
                registry = root._search_child(scope_name)
                if registry is None:
                    # if `default_scope` can not be found, fallback to argument
                    # `registry`
                    print_log(
                        f'Failed to search registry with scope "{scope_name}" '
                        f'in the "{root.name}" registry tree. '
                        f'As a workaround, the current "{self.name}" registry '
                        f'in "{self.scope}" is used to build instance. This '
                        "may cause unexpected failure when running the built "
                        f'modules. Please check whether "{scope_name}" is a '
                        "correct scope, or whether the registry is "
                        "initialized.",
                        logger="current",
                        level=logging.WARNING,
                    )
                    registry = self
            # If there is no built default scope, just return current registry.
            else:
                registry = self
            yield registry

    def _get_root_registry(self) -> "Registry":
        """Return the root registry."""
        root = self
        while root.parent is not None:
            root = root.parent
        return root

    def import_from_location(self) -> None:
        """import modules from the pre-defined locations in self._location."""
        if not self._imported:
            #     # Avoid circular import

            # avoid BC breaking
            if len(self._locations) == 0 and self.scope in MODULE2PACKAGE:
                print_log(
                    f'The "{self.name}" registry in {self.scope} did not '
                    "set import location. Fallback to call "
                    f"`{self.scope}.utils.register_all_modules` "
                    "instead.",
                    logger="current",
                    level=logging.DEBUG,
                )
                try:
                    module = import_module(f"{self.scope}.utils")
                except (ImportError, AttributeError, ModuleNotFoundError):
                    if self.scope in MODULE2PACKAGE:
                        print_log(
                            f"{self.scope} is not installed and its "
                            "modules will not be registered. If you "
                            "want to use modules defined in "
                            f"{self.scope}, Please install {self.scope} by "
                            f"`pip install {MODULE2PACKAGE[self.scope]}.",
                            logger="current",
                            level=logging.WARNING,
                        )
                    else:
                        print_log(
                            f"Failed to import {self.scope} and register "
                            "its modules, please make sure you "
                            "have registered the module manually.",
                            logger="current",
                            level=logging.WARNING,
                        )
                else:
                    # The import errors triggered during the registration
                    # may be more complex, here just throwing
                    # the error to avoid causing more implicit registry errors
                    # like `xxx`` not found in `yyy` registry.
                    module.register_all_modules(False)  # type: ignore

            for loc in self._locations:
                import_module(loc)
                print_log(
                    f"Modules of {self.scope}'s {self.name} registry have "
                    f"been automatically imported from {loc}",
                    logger="current",
                    level=logging.DEBUG,
                )
            self._imported = True

    def get(self, key: str) -> Optional[Type]:
        """Get the registry record.

        The method will first parse :attr:`key` and check whether it contains
        a scope name. The logic to search for :attr:`key`:

        - ``key`` does not contain a scope name, i.e., it is purely a module
          name like "ResNet": :meth:`get` will search for ``ResNet`` from the
          current registry to its parent or ancestors until finding it.

        - ``key`` contains a scope name and it is equal to the scope of the
          current registry (e.g., "mmcls"), e.g., "mmcls.ResNet": :meth:`get`
          will only search for ``ResNet`` in the current registry.

        - ``key`` contains a scope name and it is not equal to the scope of
          the current registry (e.g., "mmdet"), e.g., "mmcls.FCNet": If the
          scope exists in its children, :meth:`get` will get "FCNet" from
          them. If not, :meth:`get` will first get the root registry and root
          registry call its own :meth:`get` method.

        Args:
            key (str): Name of the registered item, e.g., the class name in
                string format.

        Returns:
            Type or None: Return the corresponding class if ``key`` exists,
            otherwise return None.

        Examples:
            >>> # define a registry
            >>> MODELS = Registry('models')
            >>> # register `ResNet` to `MODELS`
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            >>> resnet_cls = MODELS.get('ResNet')

            >>> # hierarchical registry
            >>> DETECTORS = Registry('detector', parent=MODELS, scope='det')
            >>> # `ResNet` does not exist in `DETECTORS` but `get` method
            >>> # will try to search from its parents or ancestors
            >>> resnet_cls = DETECTORS.get('ResNet')
            >>> CLASSIFIER = Registry('classifier', parent=MODELS, scope='cls')
            >>> @CLASSIFIER.register_module()
            >>> class MobileNet:
            >>>     pass
            >>> # `get` from its sibling registries
            >>> mobilenet_cls = DETECTORS.get('cls.MobileNet')
        """
        # Avoid circular import

        scope, real_key = self.split_scope_key(key)
        obj_cls = None
        registry_name = self.name
        scope_name = self.scope

        # lazy import the modules to register them into the registry
        self.import_from_location()

        if scope is None or scope == self._scope:
            # get from self
            if real_key in self._module_dict:
                obj_cls = self._module_dict[real_key]
            elif scope is None:
                # try to get the target from its parent or ancestors
                parent = self.parent
                while parent is not None:
                    if real_key in parent._module_dict:
                        obj_cls = parent._module_dict[real_key]
                        registry_name = parent.name
                        scope_name = parent.scope
                        break
                    parent = parent.parent
        else:
            # import the registry to add the nodes into the registry tree
            try:
                import_module(f"{scope}.registry")
                print_log(
                    f"Registry node of {scope} has been automatically " "imported.",
                    logger="current",
                    level=logging.DEBUG,
                )
            except (ImportError, AttributeError, ModuleNotFoundError):
                print_log(
                    f"Cannot auto import {scope}.registry, please check "
                    f'whether the package "{scope}" is installed correctly '
                    "or import the registry manually.",
                    logger="current",
                    level=logging.DEBUG,
                )
            # get from self._children
            if scope in self._children:
                obj_cls = self._children[scope].get(real_key)
                registry_name = self._children[scope].name
                scope_name = scope
            else:
                root = self._get_root_registry()

                if scope != root._scope and scope not in root._children:
                    # If not skip directly, `root.get(key)` will recursively
                    # call itself until RecursionError is thrown.
                    pass
                else:
                    obj_cls = root.get(key)

        if obj_cls is not None:
            print_log(
                f'Get class `{obj_cls.__name__}` from "{registry_name}"'
                f' registry in "{scope_name}"',
                logger="current",
                level=logging.DEBUG,
            )
        return obj_cls

    def _search_child(self, scope: str) -> Optional["Registry"]:
        """Depth-first search for the corresponding registry in its children.

        Note that the method only search for the corresponding registry from
        the current registry. Therefore, if we want to search from the root
        registry, :meth:`_get_root_registry` should be called to get the
        root registry first.

        Args:
            scope (str): The scope name used for searching for its
                corresponding registry.

        Returns:
            Registry or None: Return the corresponding registry if ``scope``
            exists, otherwise return None.
        """
        if self._scope == scope:
            return self

        for child in self._children.values():
            registry = child._search_child(scope)
            if registry is not None:
                return registry

        return None

    def build(self, cfg: dict, *args, **kwargs) -> Any:
        """Build an instance.

        Build an instance by calling :attr:`build_func`.

        Args:
            cfg (dict): Config dict needs to be built.

        Returns:
            Any: The constructed object.

        Examples:
            >>> from mmengine import Registry
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     def __init__(self, depth, stages=4):
            >>>         self.depth = depth
            >>>         self.stages = stages
            >>> cfg = dict(type='ResNet', depth=50)
            >>> model = MODELS.build(cfg)
        """
        return self.build_func(cfg, *args, **kwargs, registry=self)

    def _add_child(self, registry: "Registry") -> None:
        """Add a child for a registry.

        Args:
            registry (:obj:`Registry`): The ``registry`` will be added as a
                child of the ``self``.
        """

        assert isinstance(registry, Registry)
        assert registry.scope is not None
        assert (
            registry.scope not in self.children
        ), f"scope {registry.scope} exists in {self.name} registry"
        self.children[registry.scope] = registry

    def _register_module(
        self,
        module: Type,
        module_name: Optional[Union[str, List[str]]] = None,
        force: bool = False,
    ) -> None:
        """Register a module.

        Args:
            module (type): Module class or function to be registered.
            module_name (str or list of str, optional): The module name to be
                registered. If not specified, the class name will be used.
                Defaults to None.
            force (bool): Whether to override an existing class with the same
                name. Defaults to False.
        """
        if not inspect.isclass(module) and not inspect.isfunction(module):
            raise TypeError(
                "module must be a class or a function, " f"but got {type(module)}"
            )

        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                existed_module = self.module_dict[name]
                raise KeyError(
                    f"{name} is already registered in {self.name} "
                    f"at {existed_module.__module__}"
                )
            self._module_dict[name] = module

    def register_module(
        self,
        name: Optional[Union[str, List[str]]] = None,
        force: bool = False,
        module: Optional[Type] = None,
    ) -> Union[type, Callable]:
        """Register a module.

        A record will be added to ``self._module_dict``, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        Args:
            name (str or list of str, optional): The module name to be
                registered. If not specified, the class name will be used.
            force (bool): Whether to override an existing class with the same
                name. Defaults to False.
            module (type, optional): Module class or function to be registered.
                Defaults to None.

        Examples:
            >>> backbones = Registry('backbone')
            >>> # as a decorator
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass

            >>> # as a normal function
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(module=ResNet)
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        # raise the error ahead of time
        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(
                "name must be None, an instance of str, or a sequence of str, "
                f"but got {type(name)}"
            )

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            self._register_module(module=module, module_name=name, force=force)
            return module

        return _register




def master_only(func: Callable) -> Callable:
    """Decorate those methods which should be executed in master process.

    Args:
        func (callable): Function to be decorated.

    Returns:
        callable: Return decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)

    return wrapper




class BaseModule(nn.Module, metaclass=ABCMeta):
    """Base module for all modules in openmmlab. ``BaseModule`` is a wrapper of
    ``torch.nn.Module`` with additional functionality of parameter
    initialization. Compared with ``torch.nn.Module``, ``BaseModule`` mainly
    adds three attributes.

    - ``init_cfg``: the config to control the initialization.
    - ``init_weights``: The function of parameter initialization and recording
      initialization information.
    - ``_params_init_info``: Used to track the parameter initialization
      information. This attribute only exists during executing the
      ``init_weights``.

    Note:
        :obj:`PretrainedInit` has a higher priority than any other
        initializer. The loaded pretrained weights will overwrite
        the previous initialized weights.

    Args:
        init_cfg (dict or List[dict], optional): Initialization config dict.
    """

    def __init__(self, init_cfg: Union[dict, List[dict], None] = None):
        """Initialize BaseModule, inherited from `torch.nn.Module`"""

        # NOTE init_cfg can be defined in different levels, but init_cfg
        # in low levels has a higher priority.

        super().__init__()
        # define default value of init_cfg instead of hard code
        # in init_weights() function
        self._is_init = False

        self.init_cfg = copy.deepcopy(init_cfg)

        # Backward compatibility in derived classes
        # if pretrained is not None:
        #     warnings.warn('DeprecationWarning: pretrained is a deprecated \
        #         key, please consider using init_cfg')
        #     self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    @property
    def is_init(self):
        return self._is_init

    def init_weights(self):
        """Initialize the weights."""

        is_top_level_module = False
        # check if it is top-level module
        if not hasattr(self, "_params_init_info"):
            # The `_params_init_info` is used to record the initialization
            # information of the parameters
            # the key should be the obj:`nn.Parameter` of model and the value
            # should be a dict containing
            # - init_info (str): The string that describes the initialization.
            # - tmp_mean_value (FloatTensor): The mean of the parameter,
            #       which indicates whether the parameter has been modified.
            # this attribute would be deleted after all parameters
            # is initialized.
            self._params_init_info = defaultdict(dict)
            is_top_level_module = True

            # Initialize the `_params_init_info`,
            # When detecting the `tmp_mean_value` of
            # the corresponding parameter is changed, update related
            # initialization information
            for name, param in self.named_parameters():
                self._params_init_info[param]["init_info"] = (
                    f"The value is the same before and "
                    f"after calling `init_weights` "
                    f"of {self.__class__.__name__} "
                )
                self._params_init_info[param][
                    "tmp_mean_value"
                ] = param.data.mean().cpu()

            # pass `params_init_info` to all submodules
            # All submodules share the same `params_init_info`,
            # so it will be updated when parameters are
            # modified at any level of the model.
            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info

        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(
                    f"initialize {module_name} with init_cfg {self.init_cfg}",
                    logger="current",
                    level=logging.DEBUG,
                )

                init_cfgs = self.init_cfg
                if isinstance(self.init_cfg, dict):
                    init_cfgs = [self.init_cfg]

                # PretrainedInit has higher priority than any other init_cfg.
                # Therefore we initialize `pretrained_cfg` last to overwrite
                # the previous initialized weights.
                # See details in https://github.com/open-mmlab/mmengine/issues/691 # noqa E501
                other_cfgs = []
                pretrained_cfg = []
                for init_cfg in init_cfgs:
                    assert isinstance(init_cfg, dict)
                    if init_cfg["type"] == "Pretrained":
                        pretrained_cfg.append(init_cfg)
                    else:
                        other_cfgs.append(init_cfg)

                initialize(self, other_cfgs)

            for m in self.children():
                if hasattr(m, "init_weights"):
                    m.init_weights()
                    # users may overload the `init_weights`
                    update_init_info(
                        m,
                        init_info=f"Initialized by "
                        f"user-defined `init_weights`"
                        f" in {m.__class__.__name__} ",
                    )
            if self.init_cfg and pretrained_cfg:
                initialize(self, pretrained_cfg)
            self._is_init = True
        else:
            print_log(
                f"init_weights of {self.__class__.__name__} has "
                f"been called more than once.",
                logger="current",
                level=logging.WARNING,
            )

        if is_top_level_module:
            self._dump_init_info()

            for sub_module in self.modules():
                del sub_module._params_init_info

    @master_only
    def _dump_init_info(self):
        """Dump the initialization information to a file named
        `initialization.log.json` in workdir."""

        logger = MMLogger.get_current_instance()
        with_file_handler = False
        # dump the information to the logger file if there is a `FileHandler`
        for handler in logger.handlers:
            if isinstance(handler, FileHandler):
                handler.stream.write("Name of parameter - Initialization information\n")
                for name, param in self.named_parameters():
                    handler.stream.write(
                        f"\n{name} - {param.shape}: "
                        f"\n{self._params_init_info[param]['init_info']} \n"
                    )
                handler.stream.flush()
                with_file_handler = True
        if not with_file_handler:
            for name, param in self.named_parameters():
                logger.info(
                    f"\n{name} - {param.shape}: "
                    f"\n{self._params_init_info[param]['init_info']} \n "
                )

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f"\ninit_cfg={self.init_cfg}"
        return s


class BaseModel(BaseModule):
    """Base class for all algorithmic models.

    BaseModel implements the basic functions of the algorithmic model, such as
    weights initialize, batch inputs preprocess(see more information in
    :class:`BaseDataPreprocessor`), parse losses, and update model parameters.

    Subclasses inherit from BaseModel only need to implement the forward
    method, which implements the logic to calculate loss and predictions,
    then can be trained in the runner.

    Examples:
        >>> @MODELS.register_module()
        >>> class ToyModel(BaseModel):
        >>>
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.backbone = nn.Sequential()
        >>>         self.backbone.add_module('conv1', nn.Conv2d(3, 6, 5))
        >>>         self.backbone.add_module('pool', nn.MaxPool2d(2, 2))
        >>>         self.backbone.add_module('conv2', nn.Conv2d(6, 16, 5))
        >>>         self.backbone.add_module('fc1', nn.Linear(16 * 5 * 5, 120))
        >>>         self.backbone.add_module('fc2', nn.Linear(120, 84))
        >>>         self.backbone.add_module('fc3', nn.Linear(84, 10))
        >>>
        >>>         self.criterion = nn.CrossEntropyLoss()
        >>>
        >>>     def forward(self, batch_inputs, data_samples, mode='tensor'):
        >>>         data_samples = torch.stack(data_samples)
        >>>         if mode == 'tensor':
        >>>             return self.backbone(batch_inputs)
        >>>         elif mode == 'predict':
        >>>             feats = self.backbone(batch_inputs)
        >>>             predictions = torch.argmax(feats, 1)
        >>>             return predictions
        >>>         elif mode == 'loss':
        >>>             feats = self.backbone(batch_inputs)
        >>>             loss = self.criterion(feats, data_samples)
        >>>             return dict(loss=loss)

    Args:
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.

    Attributes:
        data_preprocessor (:obj:`BaseDataPreprocessor`): Used for
            pre-processing data sampled by dataloader to the format accepted by
            :meth:`forward`.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(
        self,
        data_preprocessor: Optional[Union[dict, nn.Module]] = None,
        init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        if data_preprocessor is None:
            data_preprocessor = dict(type="BaseDataPreprocessor")
        if isinstance(data_preprocessor, nn.Module):
            self.data_preprocessor = data_preprocessor
        elif isinstance(data_preprocessor, dict):
            self.data_preprocessor = MODELS.build(data_preprocessor)
        else:
            raise TypeError(
                "data_preprocessor should be a `dict` or "
                f"`nn.Module` instance, but got "
                f"{type(data_preprocessor)}"
            )

    def train_step(
        self, data: Union[dict, tuple, list], optim_wrapper
    ) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode="loss")  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data: Union[tuple, dict, list]) -> list:
        """Gets the predictions of given data.

        Calls ``self.data_preprocessor(data, False)`` and
        ``self(inputs, data_sample, mode='predict')`` in order. Return the
        predictions which will be passed to evaluator.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode="predict")  # type: ignore

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode="predict")  # type: ignore

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append([loss_name, sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(value for key, value in log_vars if "loss" in key)
        log_vars.insert(0, ["loss", loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        return loss, log_vars  # type: ignore

    def to(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.to`
        additionally.

        Returns:
            nn.Module: The model itself.
        """

        # Since Torch has not officially merged
        # the npu-related fields, using the _parse_to function
        # directly will cause the NPU to not be found.
        # Here, the input parameters are processed to avoid errors.
        if args and isinstance(args[0], str) and "npu" in args[0]:
            args = tuple([list(args)[0].replace("npu", torch.npu.native_device)])
        if kwargs and "npu" in str(kwargs.get("device", "")):
            kwargs["device"] = kwargs["device"].replace("npu", torch.npu.native_device)

        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._set_device(torch.device(device))
        return super().to(*args, **kwargs)

    def cuda(
        self,
        device: Optional[Union[int, str, torch.device]] = None,
    ) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.cuda`
        additionally.

        Returns:
            nn.Module: The model itself.
        """
        if device is None or isinstance(device, int):
            device = torch.device("cuda", index=device)
        self._set_device(torch.device(device))
        return super().cuda(device)

    def npu(
        self,
        device: Union[int, str, torch.device, None] = None,
    ) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.npu`
        additionally.

        Returns:
            nn.Module: The model itself.

        Note:
            This generation of NPU(Ascend910) does not support
            the use of multiple cards in a single process,
            so the index here needs to be consistent with the default device
        """
        device = torch.npu.current_device()
        self._set_device(device)
        return super().npu()

    def cpu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.cpu`
        additionally.

        Returns:
            nn.Module: The model itself.
        """
        self._set_device(torch.device("cpu"))
        return super().cpu()

    def _set_device(self, device: torch.device) -> None:
        """Recursively set device for `BaseDataPreprocessor` instance.

        Args:
            device (torch.device): the desired device of the parameters and
                buffers in this module.
        """

        def apply_fn(module):
            if not isinstance(module, BaseDataPreprocessor):
                return
            if device is not None:
                module._device = device

        self.apply(apply_fn)

    @abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[list] = None,
        mode: str = "tensor",
    ) -> Union[Dict[str, torch.Tensor], list]:
        """Returns losses or predictions of training, validation, testing, and
        simple inference process.

        ``forward`` method of BaseModel is an abstract method, its subclasses
        must implement this method.

        Accepts ``batch_inputs`` and ``data_sample`` processed by
        :attr:`data_preprocessor`, and returns results according to mode
        arguments.

        During non-distributed training, validation, and testing process,
        ``forward`` will be called by ``BaseModel.train_step``,
        ``BaseModel.val_step`` and ``BaseModel.val_step`` directly.

        During distributed data parallel training process,
        ``MMSeparateDistributedDataParallel.train_step`` will first call
        ``DistributedDataParallel.forward`` to enable automatic
        gradient synchronization, and then call ``forward`` to get training
        loss.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (list, optional):
                data samples collated by :attr:`data_preprocessor`.
            mode (str): mode should be one of ``loss``, ``predict`` and
                ``tensor``

                - ``loss``: Called by ``train_step`` and return loss ``dict``
                  used for logging
                - ``predict``: Called by ``val_step`` and ``test_step``
                  and return list of results used for computing metric.
                - ``tensor``: Called by custom use to get ``Tensor`` type
                  results.

        Returns:
            dict or list:
                - If ``mode == loss``, return a ``dict`` of loss tensor used
                  for backward and logging.
                - If ``mode == predict``, return a ``list`` of inference
                  results.
                - If ``mode == tensor``, return a tensor or ``tuple`` of tensor
                  or ``dict`` of tensor for custom use.
        """

    def _run_forward(
        self, data: Union[dict, tuple, list], mode: str
    ) -> Union[Dict[str, torch.Tensor], list]:
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            results = self(**data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode)
        else:
            raise TypeError(
                "Output of `data_preprocessor` should be "
                f"list, tuple or dict, but got {type(data)}"
            )
        return results

class Sequential(BaseModule, nn.Sequential):
    """Sequential module in openmmlab.

    Ensures that all modules in ``Sequential`` have a different initialization
    strategy than the outer model

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, *args, init_cfg: Optional[dict] = None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)

def build_model_from_cfg(
    cfg: Union[dict, ConfigDict, Config],
    registry: Registry,
    default_args: Optional[Union[dict, "ConfigDict", "Config"]] = None,
) -> "nn.Module":
    """Build a PyTorch model from config dict(s). Different from
    ``build_from_cfg``, if cfg is a list, a ``nn.Sequential`` will be built.

    Args:
        cfg (dict, list[dict]): The config of modules, which is either a config
            dict or a list of config dicts. If cfg is a list, the built
            modules will be wrapped with ``nn.Sequential``.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn.Module.
    """

    if isinstance(cfg, list):
        modules = [build_from_cfg(_cfg, registry, default_args) for _cfg in cfg]
        return Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


MODELS = Registry('model', build_model_from_cfg)



class SyncBatchNormFunction(Function):

    @staticmethod
    def symbolic(
        g,
        input,
        running_mean,
        running_var,
        weight,
        bias,
        momentum,
        eps,
        group,
        group_size,
        stats_mode,
    ):
        return g.op(
            "mmcv::MMCVSyncBatchNorm",
            input,
            running_mean,
            running_var,
            weight,
            bias,
            momentum_f=momentum,
            eps_f=eps,
            group_i=group,
            group_size_i=group_size,
            stats_mode=stats_mode,
        )

    @staticmethod
    def forward(
        self,
        input: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        momentum: float,
        eps: float,
        group: int,
        group_size: int,
        stats_mode: str,
    ) -> torch.Tensor:
        self.momentum = momentum
        self.eps = eps
        self.group = group
        self.group_size = group_size
        self.stats_mode = stats_mode

        assert isinstance(
            input,
            (
                torch.HalfTensor,
                torch.FloatTensor,
                torch.cuda.HalfTensor,
                torch.cuda.FloatTensor,
            ),
        ), f"only support Half or Float Tensor, but {input.type()}"
        output = torch.zeros_like(input)
        input3d = input.flatten(start_dim=2)
        output3d = output.view_as(input3d)
        num_channels = input3d.size(1)

        # ensure mean/var/norm/std are initialized as zeros
        # ``torch.empty()`` does not guarantee that
        mean = torch.zeros(num_channels, dtype=torch.float, device=input3d.device)
        var = torch.zeros(num_channels, dtype=torch.float, device=input3d.device)
        norm = torch.zeros_like(input3d, dtype=torch.float, device=input3d.device)
        std = torch.zeros(num_channels, dtype=torch.float, device=input3d.device)

        batch_size = input3d.size(0)
        if batch_size > 0:
            ext_module.sync_bn_forward_mean(input3d, mean)
            batch_flag = torch.ones([1], device=mean.device, dtype=mean.dtype)
        else:
            # skip updating mean and leave it as zeros when the input is empty
            batch_flag = torch.zeros([1], device=mean.device, dtype=mean.dtype)

        # synchronize mean and the batch flag
        vec = torch.cat([mean, batch_flag])
        if self.stats_mode == "N":
            vec *= batch_size
        if self.group_size > 1:
            dist.all_reduce(vec, group=self.group)
        total_batch = vec[-1].detach()
        mean = vec[:num_channels]

        if self.stats_mode == "default":
            mean = mean / self.group_size
        elif self.stats_mode == "N":
            mean = mean / total_batch.clamp(min=1)
        else:
            raise NotImplementedError

        # leave var as zeros when the input is empty
        if batch_size > 0:
            ext_module.sync_bn_forward_var(input3d, mean, var)

        if self.stats_mode == "N":
            var *= batch_size
        if self.group_size > 1:
            dist.all_reduce(var, group=self.group)

        if self.stats_mode == "default":
            var /= self.group_size
        elif self.stats_mode == "N":
            var /= total_batch.clamp(min=1)
        else:
            raise NotImplementedError

        # if the total batch size over all the ranks is zero,
        # we should not update the statistics in the current batch
        update_flag = total_batch.clamp(max=1)
        momentum = update_flag * self.momentum
        ext_module.sync_bn_forward_output(
            input3d,
            mean,
            var,
            weight,
            bias,
            running_mean,
            running_var,
            norm,
            std,
            output3d,
            eps=self.eps,
            momentum=momentum,
            group_size=self.group_size,
        )
        self.save_for_backward(norm, std, weight)
        return output

    @staticmethod
    @once_differentiable
    def backward(self, grad_output: torch.Tensor) -> tuple:
        norm, std, weight = self.saved_tensors
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(weight)
        grad_input = torch.zeros_like(grad_output)
        grad_output3d = grad_output.flatten(start_dim=2)
        grad_input3d = grad_input.view_as(grad_output3d)

        batch_size = grad_input3d.size(0)
        if batch_size > 0:
            ext_module.sync_bn_backward_param(
                grad_output3d, norm, grad_weight, grad_bias
            )

        # all reduce
        if self.group_size > 1:
            dist.all_reduce(grad_weight, group=self.group)
            dist.all_reduce(grad_bias, group=self.group)
            grad_weight /= self.group_size
            grad_bias /= self.group_size

        if batch_size > 0:
            ext_module.sync_bn_backward_data(
                grad_output3d, weight, grad_weight, grad_bias, norm, std, grad_input3d
            )

        return (
            grad_input,
            None,
            None,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
        )


@MODELS.register_module(name="MMSyncBN")
class SyncBatchNorm(Module):
    """Synchronized Batch Normalization.

    Args:
        num_features (int): number of features/chennels in input tensor
        eps (float, optional): a value added to the denominator for numerical
            stability. Defaults to 1e-5.
        momentum (float, optional): the value used for the running_mean and
            running_var computation. Defaults to 0.1.
        affine (bool, optional): whether to use learnable affine parameters.
            Defaults to True.
        track_running_stats (bool, optional): whether to track the running
            mean and variance during training. When set to False, this
            module does not track such statistics, and initializes statistics
            buffers ``running_mean`` and ``running_var`` as ``None``. When
            these buffers are ``None``, this module always uses batch
            statistics in both training and eval modes. Defaults to True.
        group (int, optional): synchronization of stats happen within
            each process group individually. By default it is synchronization
            across the whole world. Defaults to None.
        stats_mode (str, optional): The statistical mode. Available options
            includes ``'default'`` and ``'N'``. Defaults to 'default'.
            When ``stats_mode=='default'``, it computes the overall statistics
            using those from each worker with equal weight, i.e., the
            statistics are synchronized and simply divied by ``group``. This
            mode will produce inaccurate statistics when empty tensors occur.
            When ``stats_mode=='N'``, it compute the overall statistics using
            the total number of batches in each worker ignoring the number of
            group, i.e., the statistics are synchronized and then divied by
            the total batch ``N``. This mode is beneficial when empty tensors
            occur during training, as it average the total mean by the real
            number of batch.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        group: Optional[int] = None,
        stats_mode: str = "default",
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        group = dist.group.WORLD if group is None else group
        self.group = group
        self.group_size = dist.get_world_size(group)
        assert stats_mode in [
            "default",
            "N",
        ], f'"stats_mode" only accepts "default" and "N", got "{stats_mode}"'
        self.stats_mode = stats_mode
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()  # pytorch use ones_()
            self.bias.data.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() < 2:
            raise ValueError(f"expected at least 2D input, got {input.dim()}D input")
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or not self.track_running_stats:
            return SyncBatchNormFunction.apply(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                exponential_average_factor,
                self.eps,
                self.group,
                self.group_size,
                self.stats_mode,
            )
        else:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                exponential_average_factor,
                self.eps,
            )

    def __repr__(self):
        s = self.__class__.__name__
        s += f"({self.num_features}, "
        s += f"eps={self.eps}, "
        s += f"momentum={self.momentum}, "
        s += f"affine={self.affine}, "
        s += f"track_running_stats={self.track_running_stats}, "
        s += f"group_size={self.group_size},"
        s += f"stats_mode={self.stats_mode})"
        return s


class _BatchNormXd(nn.modules.batchnorm._BatchNorm):
    """A general BatchNorm layer without input dimension check.

    Reproduced from @kapily's work:
    (https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547)
    The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
    is `_check_input_dim` that is designed for tensor sanity checks.
    The check has been bypassed in this class for the convenience of converting
    SyncBatchNorm.
    """

    def _check_input_dim(self, input: torch.Tensor):
        return


def revert_sync_batchnorm(module: nn.Module) -> nn.Module:
    """Helper function to convert all `SyncBatchNorm` (SyncBN) and
    `mmcv.ops.sync_bn.SyncBatchNorm`(MMSyncBN) layers in the model to
    `BatchNormXd` layers.

    Adapted from @kapily's work:
    (https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547)

    Args:
        module (nn.Module): The module containing `SyncBatchNorm` layers.

    Returns:
        module_output: The converted module with `BatchNormXd` layers.
    """
    module_output = module
    module_checklist = [torch.nn.modules.batchnorm.SyncBatchNorm]

    if isinstance(module, tuple(module_checklist)):
        module_output = _BatchNormXd(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            # no_grad() may not be needed here but
            # just to be consistent with `convert_sync_batchnorm()`
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        module_output.training = module.training
        # qconfig exists in quantized models
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        # Some custom modules or 3rd party implemented modules may raise an
        # error when calling `add_module`. Therefore, try to catch the error
        # and do not raise it. See https://github.com/open-mmlab/mmengine/issues/638 # noqa: E501
        # for more details.
        try:
            module_output.add_module(name, revert_sync_batchnorm(child))
        except Exception:
            print_log(
                f"Failed to convert {child} from SyncBN to BN!",
                logger="current",
                level=logging.WARNING,
            )
    del module
    return module_output

dataset_aliases = {
    "voc": ["voc", "pascal_voc", "voc07", "voc12"],
    "imagenet_det": ["det", "imagenet_det", "ilsvrc_det"],
    "imagenet_vid": ["vid", "imagenet_vid", "ilsvrc_vid"],
    "coco": ["coco", "mscoco", "ms_coco"],
    "coco_panoptic": ["coco_panoptic", "panoptic"],
    "wider_face": ["WIDERFaceDataset", "wider_face", "WIDERFace"],
    "cityscapes": ["cityscapes"],
    "oid_challenge": ["oid_challenge", "openimages_challenge"],
    "oid_v6": ["oid_v6", "openimages_v6"],
    "objects365v1": ["objects365v1", "obj365v1"],
    "objects365v2": ["objects365v2", "obj365v2"],
}

def get_classes(dataset) -> list:
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + "_classes()")
        else:
            raise ValueError(f"Unrecognized dataset: {dataset}")
    else:
        raise TypeError(f"dataset must a str, but got {type(dataset)}")
    return labels


class CheckpointLoader:
    """A general checkpoint loader to manage all schemes."""

    _schemes: Dict[str, Callable] = {}

    @classmethod
    def _register_scheme(cls, prefixes, loader, force=False):
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        else:
            assert isinstance(prefixes, (list, tuple))
        for prefix in prefixes:
            if (prefix not in cls._schemes) or force:
                cls._schemes[prefix] = loader
            else:
                raise KeyError(
                    f"{prefix} is already registered as a loader backend, "
                    'add "force=True" if you want to override it'
                )
        # sort, longer prefixes take priority
        cls._schemes = OrderedDict(
            sorted(cls._schemes.items(), key=lambda t: t[0], reverse=True)
        )

    @classmethod
    def register_scheme(cls, prefixes, loader=None, force=False):
        """Register a loader to CheckpointLoader.

        This method can be used as a normal class method or a decorator.

        Args:
            prefixes (str or list[str] or tuple[str]):
            The prefix of the registered loader.
            loader (function, optional): The loader function to be registered.
                When this method is used as a decorator, loader is None.
                Defaults to None.
            force (bool, optional): Whether to override the loader
                if the prefix has already been registered. Defaults to False.
        """

        if loader is not None:
            cls._register_scheme(prefixes, loader, force=force)
            return

        def _register(loader_cls):
            cls._register_scheme(prefixes, loader_cls, force=force)
            return loader_cls

        return _register

    @classmethod
    def _get_checkpoint_loader(cls, path):
        """Finds a loader that supports the given path. Falls back to the local
        loader if no other loader is found.

        Args:
            path (str): checkpoint path

        Returns:
            callable: checkpoint loader
        """
        for p in cls._schemes:
            # use regular match to handle some cases that where the prefix of
            # loader has a prefix. For example, both 's3://path' and
            # 'open-mmlab:s3://path' should return `load_from_ceph`
            if re.match(p, path) is not None:
                return cls._schemes[p]

    @classmethod
    def load_checkpoint(cls, filename, map_location=None, logger="current"):
        """load checkpoint through URL scheme path.

        Args:
            filename (str): checkpoint file name with given prefix
            map_location (str, optional): Same as :func:`torch.load`.
                Defaults to None
            logger (str): The logger for message. Defaults to 'current'.

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """

        checkpoint_loader = cls._get_checkpoint_loader(filename)
        class_name = checkpoint_loader.__name__
        print_log(
            f"Loads checkpoint by {class_name[10:]} backend from path: " f"{filename}",
            logger=logger,
        )
        return checkpoint_loader(filename, map_location)


def _load_checkpoint(filename, map_location=None, logger=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str, optional): Same as :func:`torch.load`.
           Defaults to None.
        logger (:mod:`logging.Logger`, optional): The logger for error message.
           Defaults to None

    Returns:
        dict or OrderedDict: The loaded checkpoint. It can be either an
        OrderedDict storing model weights or a dict containing other
        information, which depends on the checkpoint.
    """
    return CheckpointLoader.load_checkpoint(filename, map_location, logger)


@CheckpointLoader.register_scheme(prefixes="")
def load_from_local(filename, map_location):
    """load checkpoint by local file path.

    Args:
        filename (str): local checkpoint file path
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f"{filename} can not be found.")
    checkpoint = torch.load(filename, map_location=map_location)
    if "state_dict" in checkpoint:
        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            # Replace bn with batch_norm2d in keys
            new_k = k.replace("bn.", "batch_norm2d.")
            new_state_dict[new_k] = v
        checkpoint["state_dict"] = new_state_dict
    return checkpoint


WEIGHT_INITIALIZERS = Registry('weight initializer')
def _initialize(module, cfg, wholemodule=False):
    func = build_from_cfg(cfg, WEIGHT_INITIALIZERS)
    # wholemodule flag is for override mode, there is no layer key in override
    # and initializer will give init values for the whole module with the name
    # in override.
    func.wholemodule = wholemodule
    func(module)


def _initialize_override(module, override, cfg):
    if not isinstance(override, (dict, list)):
        raise TypeError(
            f"override must be a dict or a list of dict, \
                but got {type(override)}"
        )

    override = [override] if isinstance(override, dict) else override

    for override_ in override:

        cp_override = copy.deepcopy(override_)
        name = cp_override.pop("name", None)
        if name is None:
            raise ValueError(
                '`override` must contain the key "name",' f"but got {cp_override}"
            )
        # if override only has name key, it means use args in init_cfg
        if not cp_override:
            cp_override.update(cfg)
        # if override has name key and other args except type key, it will
        # raise error
        elif "type" not in cp_override.keys():
            raise ValueError(f'`override` need "type" key, but got {cp_override}')

        if hasattr(module, name):
            _initialize(getattr(module, name), cp_override, wholemodule=True)
        else:
            raise RuntimeError(
                f"module did not have attribute {name}, "
                f"but init_cfg is {cp_override}."
            )


def initialize(module, init_cfg):
    r"""Initialize a module.

    Args:
        module (``torch.nn.Module``): the module will be initialized.
        init_cfg (dict | list[dict]): initialization configuration dict to
            define initializer. OpenMMLab has implemented 6 initializers
            including ``Constant``, ``Xavier``, ``Normal``, ``Uniform``,
            ``Kaiming``, and ``Pretrained``.

    Example:
        >>> module = nn.Linear(2, 3, bias=True)
        >>> init_cfg = dict(type='Constant', layer='Linear', val =1 , bias =2)
        >>> initialize(module, init_cfg)
        >>> module = nn.Sequential(nn.Conv1d(3, 1, 3), nn.Linear(1,2))
        >>> # define key ``'layer'`` for initializing layer with different
        >>> # configuration
        >>> init_cfg = [dict(type='Constant', layer='Conv1d', val=1),
                dict(type='Constant', layer='Linear', val=2)]
        >>> initialize(module, init_cfg)
        >>> # define key``'override'`` to initialize some specific part in
        >>> # module
        >>> class FooNet(nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.feat = nn.Conv2d(3, 16, 3)
        >>>         self.reg = nn.Conv2d(16, 10, 3)
        >>>         self.cls = nn.Conv2d(16, 5, 3)
        >>> model = FooNet()
        >>> init_cfg = dict(type='Constant', val=1, bias=2, layer='Conv2d',
        >>>     override=dict(type='Constant', name='reg', val=3, bias=4))
        >>> initialize(model, init_cfg)
        >>> model = ResNet(depth=50)
        >>> # Initialize weights with the pretrained model.
        >>> init_cfg = dict(type='Pretrained',
                checkpoint='torchvision://resnet50')
        >>> initialize(model, init_cfg)
        >>> # Initialize weights of a sub-module with the specific part of
        >>> # a pretrained model by using "prefix".
        >>> url = 'http://download.openmmlab.com/mmdetection/v2.0/retinanet/'\
        >>>     'retinanet_r50_fpn_1x_coco/'\
        >>>     'retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
        >>> init_cfg = dict(type='Pretrained',
                checkpoint=url, prefix='backbone.')
    """
    if not isinstance(init_cfg, (dict, list)):
        raise TypeError(
            f"init_cfg must be a dict or a list of dict, \
                but got {type(init_cfg)}"
        )

    if isinstance(init_cfg, dict):
        init_cfg = [init_cfg]

    for cfg in init_cfg:
        # should deeply copy the original config because cfg may be used by
        # other modules, e.g., one init_cfg shared by multiple bottleneck
        # blocks, the expected cfg will be changed after pop and will change
        # the initialization behavior of other modules
        cp_cfg = copy.deepcopy(cfg)
        override = cp_cfg.pop("override", None)
        _initialize(module, cp_cfg)

        if override is not None:
            cp_cfg.pop("layer", None)
            _initialize_override(module, override, cp_cfg)
        else:
            # All attributes in module have same initialization.
            pass


def update_init_info(module, init_info):
    """Update the `_params_init_info` in the module if the value of parameters
    are changed.

    Args:
        module (obj:`nn.Module`): The module of PyTorch with a user-defined
            attribute `_params_init_info` which records the initialization
            information.
        init_info (str): The string that describes the initialization.
    """
    assert hasattr(
        module, "_params_init_info"
    ), f"Can not find `_params_init_info` in {module}"
    for name, param in module.named_parameters():

        assert param in module._params_init_info, (
            f"Find a new :obj:`Parameter` "
            f"named `{name}` during executing the "
            f"`init_weights` of "
            f"`{module.__class__.__name__}`. "
            f"Please do not add or "
            f"replace parameters during executing "
            f"the `init_weights`. "
        )

        # The parameter has been changed during executing the
        # `init_weights` of module
        mean_value = param.data.mean().cpu()
        if module._params_init_info[param]["tmp_mean_value"] != mean_value:
            module._params_init_info[param]["init_info"] = init_info
            module._params_init_info[param]["tmp_mean_value"] = mean_value


def is_main_process(group: Optional[ProcessGroup] = None) -> bool:
    """Whether the current rank of the given process group is equal to 0.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        bool: Return True if the current rank of the given process group is
        equal to 0, otherwise False.
    """
    return get_rank(group) == 0


class HistoryBuffer:
    """Unified storage format for different log types.

    ``HistoryBuffer`` records the history of log for further statistics.

    Examples:
        >>> history_buffer = HistoryBuffer()
        >>> # Update history_buffer.
        >>> history_buffer.update(1)
        >>> history_buffer.update(2)
        >>> history_buffer.min()  # minimum of (1, 2)
        1
        >>> history_buffer.max()  # maximum of (1, 2)
        2
        >>> history_buffer.mean()  # mean of (1, 2)
        1.5
        >>> history_buffer.statistics('mean')  # access method by string.
        1.5

    Args:
        log_history (Sequence): History logs. Defaults to [].
        count_history (Sequence): Counts of history logs. Defaults to [].
        max_length (int): The max length of history logs. Defaults to 1000000.
    """

    _statistics_methods: dict = dict()

    def __init__(
        self,
        log_history: Sequence = [],
        count_history: Sequence = [],
        max_length: int = 1000000,
    ):

        self.max_length = max_length
        self._set_default_statistics()
        assert len(log_history) == len(
            count_history
        ), "The lengths of log_history and count_histroy should be equal"
        if len(log_history) > max_length:
            warnings.warn(
                f"The length of history buffer({len(log_history)}) "
                f"exceeds the max_length({max_length}), the first "
                "few elements will be ignored."
            )
            self._log_history = np.array(log_history[-max_length:])
            self._count_history = np.array(count_history[-max_length:])
        else:
            self._log_history = np.array(log_history)
            self._count_history = np.array(count_history)

    def _set_default_statistics(self) -> None:
        """Register default statistic methods: min, max, current and mean."""
        self._statistics_methods.setdefault("min", HistoryBuffer.min)
        self._statistics_methods.setdefault("max", HistoryBuffer.max)
        self._statistics_methods.setdefault("current", HistoryBuffer.current)
        self._statistics_methods.setdefault("mean", HistoryBuffer.mean)

    def update(self, log_val: Union[int, float], count: int = 1) -> None:
        """update the log history.

        If the length of the buffer exceeds ``self._max_length``, the oldest
        element will be removed from the buffer.

        Args:
            log_val (int or float): The value of log.
            count (int): The accumulation times of log, defaults to 1.
            ``count`` will be used in smooth statistics.
        """
        if not isinstance(log_val, (int, float)) or not isinstance(count, (int, float)):
            raise TypeError(
                f"log_val must be int or float but got "
                f"{type(log_val)}, count must be int but got "
                f"{type(count)}"
            )
        self._log_history = np.append(self._log_history, log_val)
        self._count_history = np.append(self._count_history, count)
        if len(self._log_history) > self.max_length:
            self._log_history = self._log_history[-self.max_length :]
            self._count_history = self._count_history[-self.max_length :]

    @property
    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the ``_log_history`` and ``_count_history``.

        Returns:
            Tuple[np.ndarray, np.ndarray]: History logs and the counts of
            the history logs.
        """
        return self._log_history, self._count_history

    @classmethod
    def register_statistics(cls, method: Callable) -> Callable:
        """Register custom statistics method to ``_statistics_methods``.

        The registered method can be called by ``history_buffer.statistics``
        with corresponding method name and arguments.

        Examples:
            >>> @HistoryBuffer.register_statistics
            >>> def weighted_mean(self, window_size, weight):
            >>>     assert len(weight) == window_size
            >>>     return (self._log_history[-window_size:] *
            >>>             np.array(weight)).sum() / \
            >>>             self._count_history[-window_size:]

            >>> log_buffer = HistoryBuffer([1, 2], [1, 1])
            >>> log_buffer.statistics('weighted_mean', 2, [2, 1])
            2

        Args:
            method (Callable): Custom statistics method.
        Returns:
            Callable: Original custom statistics method.
        """
        method_name = method.__name__
        assert (
            method_name not in cls._statistics_methods
        ), "method_name cannot be registered twice!"
        cls._statistics_methods[method_name] = method
        return method

    def statistics(self, method_name: str, *arg, **kwargs) -> Any:
        """Access statistics method by name.

        Args:
            method_name (str): Name of method.

        Returns:
            Any: Depends on corresponding method.
        """
        if method_name not in self._statistics_methods:
            raise KeyError(
                f"{method_name} has not been registered in "
                "HistoryBuffer._statistics_methods"
            )
        method = self._statistics_methods[method_name]
        # Provide self arguments for registered functions.
        return method(self, *arg, **kwargs)

    def mean(self, window_size: Optional[int] = None) -> np.ndarray:
        """Return the mean of the latest ``window_size`` values in log
        histories.

        If ``window_size is None`` or ``window_size > len(self._log_history)``,
        return the global mean value of history logs.

        Args:
            window_size (int, optional): Size of statistics window.
        Returns:
            np.ndarray: Mean value within the window.
        """
        if window_size is not None:
            assert isinstance(window_size, int), (
                "The type of window size should be int, but got " f"{type(window_size)}"
            )
        else:
            window_size = len(self._log_history)
        logs_sum = self._log_history[-window_size:].sum()
        counts_sum = self._count_history[-window_size:].sum()
        return logs_sum / counts_sum

    def max(self, window_size: Optional[int] = None) -> np.ndarray:
        """Return the maximum value of the latest ``window_size`` values in log
        histories.

        If ``window_size is None`` or ``window_size > len(self._log_history)``,
        return the global maximum value of history logs.

        Args:
            window_size (int, optional): Size of statistics window.
        Returns:
            np.ndarray: The maximum value within the window.
        """
        if window_size is not None:
            assert isinstance(window_size, int), (
                "The type of window size should be int, but got " f"{type(window_size)}"
            )
        else:
            window_size = len(self._log_history)
        return self._log_history[-window_size:].max()

    def min(self, window_size: Optional[int] = None) -> np.ndarray:
        """Return the minimum value of the latest ``window_size`` values in log
        histories.

        If ``window_size is None`` or ``window_size > len(self._log_history)``,
        return the global minimum value of history logs.

        Args:
            window_size (int, optional): Size of statistics window.
        Returns:
            np.ndarray: The minimum value within the window.
        """
        if window_size is not None:
            assert isinstance(window_size, int), (
                "The type of window size should be int, but got " f"{type(window_size)}"
            )
        else:
            window_size = len(self._log_history)
        return self._log_history[-window_size:].min()

    def current(self) -> np.ndarray:
        """Return the recently updated values in log histories.

        Returns:
            np.ndarray: Recently updated values in log histories.
        """
        if len(self._log_history) == 0:
            raise ValueError(
                "HistoryBuffer._log_history is an empty array! "
                "please call update first"
            )
        return self._log_history[-1]


class MessageHub(ManagerMixin):
    """Message hub for component interaction. MessageHub is created and
    accessed in the same way as ManagerMixin.

    ``MessageHub`` will record log information and runtime information. The
    log information refers to the learning rate, loss, etc. of the model
    during training phase, which will be stored as ``HistoryBuffer``. The
    runtime information refers to the iter times, meta information of
    runner etc., which will be overwritten by next update.

    Args:
        name (str): Name of message hub used to get corresponding instance
            globally.
        log_scalars (OrderedDict, optional): Each key-value pair in the
            dictionary is the name of the log information such as "loss", "lr",
            "metric" and their corresponding values. The type of value must be
            HistoryBuffer. Defaults to None.
        runtime_info (OrderedDict, optional): Each key-value pair in the
            dictionary is the name of the runtime information and their
            corresponding values. Defaults to None.
        resumed_keys (OrderedDict, optional): Each key-value pair in the
            dictionary decides whether the key in :attr:`_log_scalars` and
            :attr:`_runtime_info` will be serialized.

    Note:
        Key in :attr:`_resumed_keys` belongs to :attr:`_log_scalars` or
        :attr:`_runtime_info`. The corresponding value cannot be set
        repeatedly.

    Examples:
        >>> # create empty `MessageHub`.
        >>> message_hub1 = MessageHub()
        >>> log_scalars = OrderedDict(loss=HistoryBuffer())
        >>> runtime_info = OrderedDict(task='task')
        >>> resumed_keys = dict(loss=True)
        >>> # create `MessageHub` from data.
        >>> message_hub2 = MessageHub(
        >>>     name='name',
        >>>     log_scalars=log_scalars,
        >>>     runtime_info=runtime_info,
        >>>     resumed_keys=resumed_keys)
    """

    def __init__(
        self,
        name: str,
        log_scalars: Optional[OrderedDict] = None,
        runtime_info: Optional[OrderedDict] = None,
        resumed_keys: Optional[OrderedDict] = None,
    ):
        super().__init__(name)
        self._log_scalars = log_scalars if log_scalars is not None else OrderedDict()
        self._runtime_info = runtime_info if runtime_info is not None else OrderedDict()
        self._resumed_keys = resumed_keys if resumed_keys is not None else OrderedDict()

        assert isinstance(self._log_scalars, OrderedDict)
        assert isinstance(self._runtime_info, OrderedDict)
        assert isinstance(self._resumed_keys, OrderedDict)

        for value in self._log_scalars.values():
            assert isinstance(value, HistoryBuffer), (
                "The type of log_scalars'value must be HistoryBuffer, but "
                f"got {type(value)}"
            )

        for key in self._resumed_keys.keys():
            assert key in self._log_scalars or key in self._runtime_info, (
                "Key in `resumed_keys` must contained in `log_scalars` or "
                f"`runtime_info`, but got {key}"
            )

    @classmethod
    def get_current_instance(cls) -> "MessageHub":
        """Get latest created ``MessageHub`` instance.

        :obj:`MessageHub` can call :meth:`get_current_instance` before any
        instance has been created, and return a message hub with the instance
        name "mmengine".

        Returns:
            MessageHub: Empty ``MessageHub`` instance.
        """
        if not cls._instance_dict:
            cls.get_instance("mmengine")
        return super().get_current_instance()

    def update_scalar(
        self,
        key: str,
        value: Union[int, float, np.ndarray, "torch.Tensor"],
        count: int = 1,
        resumed: bool = True,
    ) -> None:
        """Update :attr:_log_scalars.

        Update ``HistoryBuffer`` in :attr:`_log_scalars`. If corresponding key
        ``HistoryBuffer`` has been created, ``value`` and ``count`` is the
        argument of ``HistoryBuffer.update``, Otherwise, ``update_scalar``
        will create an ``HistoryBuffer`` with value and count via the
        constructor of ``HistoryBuffer``.

        Examples:
            >>> message_hub = MessageHub
            >>> # create loss `HistoryBuffer` with value=1, count=1
            >>> message_hub.update_scalar('loss', 1)
            >>> # update loss `HistoryBuffer` with value
            >>> message_hub.update_scalar('loss', 3)
            >>> message_hub.update_scalar('loss', 3, resumed=False)
            AssertionError: loss used to be true, but got false now. resumed
            keys cannot be modified repeatedly'

        Note:
            The ``resumed`` argument needs to be consistent for the same
            ``key``.

        Args:
            key (str): Key of ``HistoryBuffer``.
            value (torch.Tensor or np.ndarray or int or float): Value of log.
            count (torch.Tensor or np.ndarray or int or float): Accumulation
                times of log, defaults to 1. `count` will be used in smooth
                statistics.
            resumed (str): Whether the corresponding ``HistoryBuffer``
                could be resumed. Defaults to True.
        """
        self._set_resumed_keys(key, resumed)
        checked_value = self._get_valid_value(value)
        assert isinstance(
            count, int
        ), f"The type of count must be int. but got {type(count): {count}}"
        if key in self._log_scalars:
            self._log_scalars[key].update(checked_value, count)
        else:
            self._log_scalars[key] = HistoryBuffer([checked_value], [count])

    def update_scalars(self, log_dict: dict, resumed: bool = True) -> None:
        """Update :attr:`_log_scalars` with a dict.

        ``update_scalars`` iterates through each pair of log_dict key-value,
        and calls ``update_scalar``. If type of value is dict, the value should
        be ``dict(value=xxx) or dict(value=xxx, count=xxx)``. Item in
        ``log_dict`` has the same resume option.

        Note:
            The ``resumed`` argument needs to be consistent for the same
            ``log_dict``.

        Args:
            log_dict (str): Used for batch updating :attr:`_log_scalars`.
            resumed (bool): Whether all ``HistoryBuffer`` referred in
                log_dict should be resumed. Defaults to True.

        Examples:
            >>> message_hub = MessageHub.get_instance('mmengine')
            >>> log_dict = dict(a=1, b=2, c=3)
            >>> message_hub.update_scalars(log_dict)
            >>> # The default count of  `a`, `b` and `c` is 1.
            >>> log_dict = dict(a=1, b=2, c=dict(value=1, count=2))
            >>> message_hub.update_scalars(log_dict)
            >>> # The count of `c` is 2.
        """
        assert isinstance(log_dict, dict), (
            "`log_dict` must be a dict!, " f"but got {type(log_dict)}"
        )
        for log_name, log_val in log_dict.items():
            if isinstance(log_val, dict):
                assert "value" in log_val, f"value must be defined in {log_val}"
                count = self._get_valid_value(log_val.get("count", 1))
                value = log_val["value"]
            else:
                count = 1
                value = log_val
            assert isinstance(count, int), (
                "The type of count must be int. but got " f"{type(count): {count}}"
            )
            self.update_scalar(log_name, value, count, resumed)

    def update_info(self, key: str, value: Any, resumed: bool = True) -> None:
        """Update runtime information.

        The key corresponding runtime information will be overwritten each
        time calling ``update_info``.

        Note:
            The ``resumed`` argument needs to be consistent for the same
            ``key``.

        Examples:
            >>> message_hub = MessageHub()
            >>> message_hub.update_info('iter', 100)

        Args:
            key (str): Key of runtime information.
            value (Any): Value of runtime information.
            resumed (bool): Whether the corresponding ``HistoryBuffer``
                could be resumed.
        """
        self._set_resumed_keys(key, resumed)
        self._runtime_info[key] = value

    def update_info_dict(self, info_dict: dict, resumed: bool = True) -> None:
        """Update runtime information with dictionary.

        The key corresponding runtime information will be overwritten each
        time calling ``update_info``.

        Note:
            The ``resumed`` argument needs to be consistent for the same
            ``info_dict``.

        Examples:
            >>> message_hub = MessageHub()
            >>> message_hub.update_info({'iter': 100})

        Args:
            info_dict (str): Runtime information dictionary.
            resumed (bool): Whether the corresponding ``HistoryBuffer``
                could be resumed.
        """
        assert isinstance(info_dict, dict), (
            "`log_dict` must be a dict!, " f"but got {type(info_dict)}"
        )
        for key, value in info_dict.items():
            self.update_info(key, value, resumed=resumed)

    def _set_resumed_keys(self, key: str, resumed: bool) -> None:
        """Set corresponding resumed keys.

        This method is called by ``update_scalar``, ``update_scalars`` and
        ``update_info`` to set the corresponding key is true or false in
        :attr:`_resumed_keys`.

        Args:
            key (str): Key of :attr:`_log_scalrs` or :attr:`_runtime_info`.
            resumed (bool): Whether the corresponding ``HistoryBuffer``
                could be resumed.
        """
        if key not in self._resumed_keys:
            self._resumed_keys[key] = resumed
        else:
            assert self._resumed_keys[key] == resumed, (
                f"{key} used to be {self._resumed_keys[key]}, but got "
                "{resumed} now. resumed keys cannot be modified repeatedly."
            )

    @property
    def log_scalars(self) -> OrderedDict:
        """Get all ``HistoryBuffer`` instances.

        Note:
            Considering the large memory footprint of history buffers in the
            post-training, :meth:`get_scalar` will return a reference of
            history buffer rather than a copy.

        Returns:
            OrderedDict: All ``HistoryBuffer`` instances.
        """
        return self._log_scalars

    @property
    def runtime_info(self) -> OrderedDict:
        """Get all runtime information.

        Returns:
            OrderedDict: A copy of all runtime information.
        """
        # return copy.deepcopy(self._runtime_info)
        return self._runtime_info

    def get_scalar(self, key: str) -> HistoryBuffer:
        """Get ``HistoryBuffer`` instance by key.

        Note:
            Considering the large memory footprint of history buffers in the
            post-training, :meth:`get_scalar` will not return a reference of
            history buffer rather than a copy.

        Args:
            key (str): Key of ``HistoryBuffer``.

        Returns:
            HistoryBuffer: Corresponding ``HistoryBuffer`` instance if the
            key exists.
        """
        if key not in self.log_scalars:
            raise KeyError(
                f"{key} is not found in Messagehub.log_buffers: "
                f"instance name is: {MessageHub.instance_name}"
            )
        return self.log_scalars[key]

    def get_info(self, key: str) -> Any:
        """Get runtime information by key.

        Args:
            key (str): Key of runtime information.

        Returns:
            Any: A copy of corresponding runtime information if the key exists.
        """
        if key not in self.runtime_info:
            raise KeyError(
                f"{key} is not found in Messagehub.log_buffers: "
                f"instance name is: {MessageHub.instance_name}"
            )

        # TODO： There are restrictions on objects that can be saved
        # return copy.deepcopy(self._runtime_info[key])
        return self._runtime_info[key]

    def _get_valid_value(
        self,
        value: Union["torch.Tensor", np.ndarray, np.number, int, float],
    ) -> Union[int, float]:
        """Convert value to python built-in type.

        Args:
            value (torch.Tensor or np.ndarray or np.number or int or float):
                value of log.

        Returns:
            float or int: python built-in type value.
        """
        if isinstance(value, (np.ndarray, np.number)):
            assert value.size == 1
            value = value.item()
        elif isinstance(value, (int, float)):
            value = value
        else:
            # check whether value is torch.Tensor but don't want
            # to import torch in this file
            assert hasattr(value, "numel") and value.numel() == 1
            value = value.item()
        return value  # type: ignore

    def state_dict(self) -> dict:
        """Returns a dictionary containing log scalars, runtime information and
        resumed keys, which should be resumed.

        The returned ``state_dict`` can be loaded by :meth:`load_state_dict`.

        Returns:
            dict: A dictionary contains ``log_scalars``, ``runtime_info`` and
            ``resumed_keys``.
        """
        saved_scalars = OrderedDict()
        saved_info = OrderedDict()

        for key, value in self._log_scalars.items():
            if self._resumed_keys.get(key, False):
                saved_scalars[key] = copy.deepcopy(value)

        for key, value in self._runtime_info.items():
            if self._resumed_keys.get(key, False):
                try:
                    saved_info[key] = copy.deepcopy(value)
                except:  # noqa: E722
                    print_log(
                        f"{key} in message_hub cannot be copied, "
                        f"just return its reference. ",
                        logger="current",
                        level=logging.WARNING,
                    )
                    saved_info[key] = value
        return dict(
            log_scalars=saved_scalars,
            runtime_info=saved_info,
            resumed_keys=self._resumed_keys,
        )

    def load_state_dict(self, state_dict: Union["MessageHub", dict]) -> None:
        """Loads log scalars, runtime information and resumed keys from
        ``state_dict`` or ``message_hub``.

        If ``state_dict`` is a dictionary returned by :meth:`state_dict`, it
        will only make copies of data which should be resumed from the source
        ``message_hub``.

        If ``state_dict`` is a ``message_hub`` instance, it will make copies of
        all data from the source message_hub. We suggest to load data from
        ``dict`` rather than a ``MessageHub`` instance.

        Args:
            state_dict (dict or MessageHub): A dictionary contains key
                ``log_scalars`` ``runtime_info`` and ``resumed_keys``, or a
                MessageHub instance.
        """
        if isinstance(state_dict, dict):
            for key in ("log_scalars", "runtime_info", "resumed_keys"):
                assert key in state_dict, (
                    "The loaded `state_dict` of `MessageHub` must contain "
                    f"key: `{key}`"
                )
            # The old `MessageHub` could save non-HistoryBuffer `log_scalars`,
            # therefore the loaded `log_scalars` needs to be filtered.
            for key, value in state_dict["log_scalars"].items():
                if not isinstance(value, HistoryBuffer):
                    print_log(
                        f"{key} in message_hub is not HistoryBuffer, "
                        f"just skip resuming it.",
                        logger="current",
                        level=logging.WARNING,
                    )
                    continue
                self.log_scalars[key] = value

            for key, value in state_dict["runtime_info"].items():
                try:
                    self._runtime_info[key] = copy.deepcopy(value)
                except:  # noqa: E722
                    print_log(
                        f"{key} in message_hub cannot be copied, "
                        f"just return its reference.",
                        logger="current",
                        level=logging.WARNING,
                    )
                    self._runtime_info[key] = value

            for key, value in state_dict["resumed_keys"].items():
                if key not in set(self.log_scalars.keys()) | set(
                    self._runtime_info.keys()
                ):
                    print_log(
                        f"resumed key: {key} is not defined in message_hub, "
                        f"just skip resuming this key.",
                        logger="current",
                        level=logging.WARNING,
                    )
                    continue
                elif not value:
                    print_log(
                        f"Although resumed key: {key} is False, {key} "
                        "will still be loaded this time. This key will "
                        "not be saved by the next calling of "
                        "`MessageHub.state_dict()`",
                        logger="current",
                        level=logging.WARNING,
                    )
                self._resumed_keys[key] = value

        # Since some checkpoints saved serialized `message_hub` instance,
        # `load_state_dict` support loading `message_hub` instance for
        # compatibility
        else:
            self._log_scalars = copy.deepcopy(state_dict._log_scalars)
            self._runtime_info = copy.deepcopy(state_dict._runtime_info)
            self._resumed_keys = copy.deepcopy(state_dict._resumed_keys)


def _get_norm() -> tuple:
    """A wrapper to obtain base classes of normalization layers from PyTorch or
    Parrots."""
    if TORCH_VERSION == 'parrots':
        from parrots.nn.modules.batchnorm import _BatchNorm, _InstanceNorm
        SyncBatchNorm_ = torch.nn.SyncBatchNorm2d
    else:
        from torch.nn.modules.batchnorm import _BatchNorm
        from torch.nn.modules.instancenorm import _InstanceNorm
        SyncBatchNorm_ = torch.nn.SyncBatchNorm
    return _BatchNorm, _InstanceNorm, SyncBatchNorm_


def _get_conv() -> tuple:
    """A wrapper to obtain base classes of Conv layers from PyTorch or
    Parrots."""
    if TORCH_VERSION == "parrots":
        from parrots.nn.modules.conv import _ConvNd, _ConvTransposeMixin
    else:
        from torch.nn.modules.conv import _ConvNd, _ConvTransposeMixin
    return _ConvNd, _ConvTransposeMixin

def _get_dataloader() -> tuple:
    """A wrapper to obtain DataLoader class from PyTorch or Parrots."""
    if TORCH_VERSION == 'parrots':
        from torch.utils.data import DataLoader, PoolDataLoader
    else:
        from torch.utils.data import DataLoader
        PoolDataLoader = DataLoader
    return DataLoader, PoolDataLoader


def _get_pool() -> tuple:
    """A wrapper to obtain base classes of pooling layers from PyTorch or
    Parrots."""
    if TORCH_VERSION == 'parrots':
        from parrots.nn.modules.pool import (_AdaptiveAvgPoolNd,
                                             _AdaptiveMaxPoolNd, _AvgPoolNd,
                                             _MaxPoolNd)
    else:
        from torch.nn.modules.pooling import (_AdaptiveAvgPoolNd,
                                              _AdaptiveMaxPoolNd, _AvgPoolNd,
                                              _MaxPoolNd)
    return _AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd, _AvgPoolNd, _MaxPoolNd

_ConvNd, _ConvTransposeMixin = _get_conv()
DataLoader, PoolDataLoader = _get_dataloader()
_BatchNorm, _InstanceNorm, SyncBatchNorm_ = _get_norm()
_AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd, _AvgPoolNd, _MaxPoolNd = _get_pool()


_ConvNd, _ConvTransposeMixin = _get_conv()
DataLoader, PoolDataLoader = _get_dataloader()
_BatchNorm, _InstanceNorm, SyncBatchNorm_ = _get_norm()
_AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd, _AvgPoolNd, _MaxPoolNd = _get_pool()

def has_batch_norm(model: nn.Module) -> bool:
    """Detect whether model has a BatchNormalization layer.

    Args:
        model (nn.Module): training model.

    Returns:
        bool: whether model has a BatchNormalization layer
    """
    if isinstance(model, _BatchNorm):
        return True
    for m in model.children():
        if has_batch_norm(m):
            return True
    return False



OPTIM_WRAPPERS = Registry("optim_wrapper")
@OPTIM_WRAPPERS.register_module()
class OptimWrapper:
    """Optimizer wrapper provides a common interface for updating parameters.

    Optimizer wrapper provides a unified interface for single precision
    training and automatic mixed precision training with different hardware.
    OptimWrapper encapsulates optimizer to provide simplified interfaces
    for commonly used training techniques such as gradient accumulative and
    grad clips. ``OptimWrapper`` implements the basic logic of gradient
    accumulation and gradient clipping based on ``torch.optim.Optimizer``.
    The subclasses only need to override some methods to implement the mixed
    precision training. See more information in :class:`AmpOptimWrapper`.

    Args:
        optimizer (Optimizer): Optimizer used to update model parameters.
        accumulative_counts (int): The number of iterations to accumulate
            gradients. The parameters will be updated per
            ``accumulative_counts``.
        clip_grad (dict, optional): If ``clip_grad`` is not None, it will be
            the arguments of :func:`torch.nn.utils.clip_grad_norm_` or
            :func:`torch.nn.utils.clip_grad_value_`. ``clip_grad`` should be a
            dict, and the keys could be set as follows:

            If the key ``type`` is not set, or ``type`` is "norm",
            the accepted keys are as follows:

            - max_norm (float or int): Max norm of the gradients.
            - norm_type (float or int): Type of the used p-norm. Can be
              ``'inf'`` for infinity norm.
            - error_if_nonfinite (bool): If True, an error is thrown if
              the total norm of the gradients from :attr:`parameters` is
              ``nan``, ``inf``, or ``-inf``. Defaults to False (will switch
              to True in the future)

            If the key ``type`` is set to "value", the accepted keys are as
            follows:

            - clip_value (float or int): maximum allowed value of the
              gradients. The gradients are clipped in the range
              ``(-clip_value, +clip_value)``.

    Note:
        If ``accumulative_counts`` is larger than 1, perform
        :meth:`update_params` under the context of  ``optim_context``
        could avoid unnecessary gradient synchronization.

    Note:
        If you use ``IterBasedRunner`` and enable gradient accumulation,
        the original `max_iters` should be multiplied by
        ``accumulative_counts``.

    Note:
        The subclass should ensure that once :meth:`update_params` is called,
        ``_inner_count += 1`` is automatically performed.

    Examples:
        >>> # Config sample of OptimWrapper and enable clipping gradient by
        >>> # norm.
        >>> optim_wrapper_cfg = dict(
        >>>     type='OptimWrapper',
        >>>     _accumulative_counts=1,
        >>>     clip_grad=dict(max_norm=0.2))
        >>> # Config sample of OptimWrapper and enable clipping gradient by
        >>> # value.
        >>> optim_wrapper_cfg = dict(
        >>>     type='OptimWrapper',
        >>>     _accumulative_counts=1,
        >>>     clip_grad=dict(type='value', clip_value=0.2))
        >>> # Use OptimWrapper to update model.
        >>> import torch.nn as nn
        >>> import torch
        >>> from torch.optim import SGD
        >>> from torch.utils.data import DataLoader
        >>> from mmengine.optim import OptimWrapper
        >>>
        >>> model = nn.Linear(1, 1)
        >>> dataset = torch.randn(10, 1, 1)
        >>> dataloader = DataLoader(dataset)
        >>> optimizer = SGD(model.parameters(), lr=0.1)
        >>> optim_wrapper = OptimWrapper(optimizer)
        >>>
        >>> for data in dataloader:
        >>>     loss = model(data)
        >>>     optim_wrapper.update_params(loss)
        >>> # Enable gradient accumulation
        >>> optim_wrapper_cfg = dict(
        >>>     type='OptimWrapper',
        >>>     _accumulative_counts=3,
        >>>     clip_grad=dict(max_norm=0.2))
        >>> ddp_model = DistributedDataParallel(model)
        >>> optimizer = SGD(ddp_model.parameters(), lr=0.1)
        >>> optim_wrapper = OptimWrapper(optimizer)
        >>> optim_wrapper.initialize_count_status(0, len(dataloader))
        >>> # If model is a subclass instance of DistributedDataParallel,
        >>> # `optim_context` context manager can avoid unnecessary gradient
        >>> #  synchronize.
        >>> for iter, data in enumerate(dataloader):
        >>>     with optim_wrapper.optim_context(ddp_model):
        >>>         loss = model(data)
        >>>     optim_wrapper.update_params(loss)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        accumulative_counts: int = 1,
        clip_grad: Optional[dict] = None,
    ):
        assert (
            accumulative_counts > 0
        ), "_accumulative_counts at least greater than or equal to 1"
        self._accumulative_counts = accumulative_counts

        assert isinstance(optimizer, Optimizer), (
            "optimizer must be a `torch.optim.Optimizer` instance, but got "
            f"{type(optimizer)}"
        )
        self.optimizer = optimizer

        if clip_grad is not None:
            # clip_grad_kwargs should not be non-empty dict.
            assert isinstance(clip_grad, dict) and clip_grad, (
                "If `clip_grad` is not None, it should be a `dict` "
                "which is the arguments of `torch.nn.utils.clip_grad_norm_` "
                "or clip_grad_value_`."
            )
            clip_type = clip_grad.pop("type", "norm")
            if clip_type == "norm":
                self.clip_func = torch.nn.utils.clip_grad_norm_
                self.grad_name = "grad_norm"
            elif clip_type == "value":
                self.clip_func = torch.nn.utils.clip_grad_value_
                self.grad_name = "grad_value"
            else:
                raise ValueError(
                    'type of clip_grad should be "norm" or '
                    f'"value" but got {clip_type}'
                )
            assert clip_grad, (
                "`clip_grad` should contain other arguments "
                "besides `type`. The arguments should match "
                "with the `torch.nn.utils.clip_grad_norm_` or "
                "clip_grad_value_`"
            )
        self.clip_grad_kwargs = clip_grad
        # Used to update `grad_norm` log message.
        self.message_hub = MessageHub.get_current_instance()
        self._inner_count = 0
        # `_max_counts` means the total number of parameter updates.  It
        # ensures that the gradient of the last few iterations will not be
        # lost when the `_max_counts` is not divisible by
        # `accumulative_counts`.
        self._max_counts = -1
        # The `_remainder_iter` is used for calculating loss factor at the
        # last few iterations. If `_max_counts` has not been initialized,
        # the loss factor will always be the same as `_accumulative_counts`.
        self._remainder_counts = -1

    def update_params(
        self,
        loss: torch.Tensor,
        step_kwargs: Optional[Dict] = None,
        zero_kwargs: Optional[Dict] = None,
    ) -> None:
        """Update parameters in :attr:`optimizer`.

        Args:
            loss (torch.Tensor): A tensor for back propagation.
            step_kwargs (dict): Arguments for optimizer.step.
                Defaults to None.
                New in version v0.4.0.
            zero_kwargs (dict): Arguments for optimizer.zero_grad.
                Defaults to None.
                New in version v0.4.0.
        """
        if step_kwargs is None:
            step_kwargs = {}
        if zero_kwargs is None:
            zero_kwargs = {}
        loss = self.scale_loss(loss)
        self.backward(loss)
        # Update parameters only if `self._inner_count` is divisible by
        # `self._accumulative_counts` or `self._inner_count` equals to
        # `self._max_counts`
        if self.should_update():
            self.step(**step_kwargs)
            self.zero_grad(**zero_kwargs)

    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        """Perform gradient back propagation.

        Provide unified ``backward`` interface compatible with automatic mixed
        precision training. Subclass can overload this method to implement the
        required logic. For example, ``torch.cuda.amp`` require some extra
        operation on GradScaler during backward process.

        Note:
            If subclasses inherit from ``OptimWrapper`` override
            ``backward``, ``_inner_count +=1`` must be implemented.

        Args:
            loss (torch.Tensor): The loss of current iteration.
            kwargs: Keyword arguments passed to :meth:`torch.Tensor.backward`.
        """
        loss.backward(**kwargs)
        self._inner_count += 1

    def zero_grad(self, **kwargs) -> None:
        """A wrapper of ``Optimizer.zero_grad``.

        Provide unified ``zero_grad`` interface compatible with automatic mixed
        precision training. Subclass can overload this method to implement the
        required logic.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`torch.optim.Optimizer.zero_grad`.
        """
        self.optimizer.zero_grad(**kwargs)

    def step(self, **kwargs) -> None:
        """A wrapper of ``Optimizer.step``.

        Provide unified ``step`` interface compatible with automatic mixed
        precision training. Subclass can overload this method to implement the
        required logic. For example, ``torch.cuda.amp`` require some extra
        operation on ``GradScaler`` during step process.

        Clip grad if :attr:`clip_grad_kwargs` is not None, and then update
        parameters.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`torch.optim.Optimizer.step`.
        """
        if self.clip_grad_kwargs:
            self._clip_grad()
        self.optimizer.step(**kwargs)

    def state_dict(self) -> dict:
        """A wrapper of ``Optimizer.state_dict``.

        Provide unified ``state_dict`` interface compatible with automatic
        mixed precision training. Subclass can overload this method to
        implement the required logic. For example, the state dictionary of
        GradScaler should be saved when training with ``torch.cuda.amp``.

        Returns:
            dict: The state dictionary of :attr:`optimizer`.
        """
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """A wrapper of ``Optimizer.load_state_dict``. load the state dict of
        :attr:`optimizer`.

        Provide unified ``load_state_dict`` interface compatible with automatic
        mixed precision training. Subclass can overload this method to
        implement the required logic. For example, the state dictionary of
        GradScaler should be loaded when training with ``torch.cuda.amp``.

        Args:
            state_dict (dict): The state dictionary of :attr:`optimizer`.
        """
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self) -> List[dict]:
        """A wrapper of ``Optimizer.param_groups``.

        Make OptimizeWrapper compatible with :class:`_ParamScheduler`.

        Returns:
             dict: the ``param_groups`` of :attr:`optimizer`.
        """
        return self.optimizer.param_groups

    @property
    def defaults(self) -> dict:
        """A wrapper of ``Optimizer.defaults``.

        Make OptimizeWrapper compatible with :class:`_ParamScheduler`.

        Returns:
             dict: the ``param_groups`` of :attr:`optimizer`.
        """
        return self.optimizer.defaults

    def get_lr(self) -> Dict[str, List[float]]:
        """Get the learning rate of the optimizer.

        Provide unified interface to get learning rate of optimizer.

        Returns:
            Dict[str, List[float]]: Learning rate of the optimizer.
        """
        lr = [group["lr"] for group in self.param_groups]
        return dict(lr=lr)

    def get_momentum(self) -> Dict[str, List[float]]:
        """Get the momentum of the optimizer.

        Provide unified interface to get momentum of optimizer.

        Returns:
            Dict[str, List[float]]: Momentum of the optimizer.
        """
        momentum = []
        for group in self.param_groups:
            # Get momentum of SGD.
            if "momentum" in group.keys():
                momentum.append(group["momentum"])
            # Get momentum of Adam.
            elif "betas" in group.keys():
                momentum.append(group["betas"][0])
            else:
                momentum.append(0)
        return dict(momentum=momentum)

    @contextmanager
    def optim_context(self, model: nn.Module):
        """A Context for gradient accumulation and automatic mix precision
        training.

        If subclasses need to enable the context for mix precision training,
        e.g., ``:class:`AmpOptimWrapper``,  the corresponding context should be
        enabled in `optim_context`. Since ``OptimWrapper`` uses default fp32
        training, ``optim_context`` will only enable the context for
        blocking the unnecessary gradient synchronization during gradient
        accumulation

        If model is an instance with ``no_sync`` method (which means
        blocking the gradient synchronization) and
        ``self._accumulative_counts != 1``. The model will not automatically
        synchronize gradients if ``cur_iter`` is divisible by
        ``self._accumulative_counts``. Otherwise, this method will enable an
        empty context.

        Args:
            model (nn.Module): The training model.
        """
        # During gradient accumulation process, the gradient synchronize
        # should only happen before updating parameters.
        if not self.should_sync() and hasattr(model, "no_sync"):
            with model.no_sync():
                yield
        else:
            yield

    def _clip_grad(self) -> None:
        """Clip the gradients of parameters."""
        params: List[torch.Tensor] = []
        for param_group in self.optimizer.param_groups:
            params.extend(param_group["params"])

        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            grad = self.clip_func(params, **self.clip_grad_kwargs)
            # `torch.nn.utils.clip_grad_value_` will return None.
            if grad is not None:
                self.message_hub.update_scalar(f"train/{self.grad_name}", float(grad))

    def initialize_count_status(
        self, model: nn.Module, init_counts: int, max_counts: int
    ) -> None:
        """Initialize gradient accumulation related attributes.

        ``OptimWrapper`` can be used without calling
        ``initialize_iter_status``. However, Consider the case of  ``len(
        dataloader) == 10``, and the ``accumulative_iter == 3``. Since 10 is
        not divisible by 3, the last iteration will not trigger
        ``optimizer.step()``, resulting in one less parameter updating.

        Args:
            model (nn.Module): Training model
            init_counts (int): The initial value of the inner count.
            max_counts (int): The maximum value of the inner count.
        """
        self._inner_count = init_counts
        self._max_counts = max_counts
        if self._inner_count % self._accumulative_counts != 0:
            print_log(
                "Resumed iteration number is not divisible by "
                "`_accumulative_counts` in `GradientCumulativeOptimizerHook`, "
                "which means the gradient of some iterations is lost and the "
                "result may be influenced slightly.",
                logger="current",
                level=logging.WARNING,
            )

        if has_batch_norm(model) and self._accumulative_counts > 1:
            print_log(
                "Gradient accumulative may slightly decrease "
                "performance because the model has BatchNorm layers.",
                logger="current",
                level=logging.WARNING,
            )
        # Remainder of `_max_counts` divided by `_accumulative_counts`
        self._remainder_counts = self._max_counts % self._accumulative_counts

    def should_update(self) -> bool:
        """Decide whether the parameters should be updated at the current
        iteration.

        Called by :meth:`update_params` and check whether the optimizer
        wrapper should update parameters at current iteration.

        Returns:
            bool: Whether to update parameters.
        """
        return (
            self._inner_count % self._accumulative_counts == 0
            or self._inner_count == self._max_counts
        )

    def should_sync(self) -> bool:
        """Decide whether the automatic gradient synchronization should be
        allowed at the current iteration.

        It takes effect when gradient accumulation is used to skip
        synchronization at the iterations where the parameter is not updated.

        Since ``should_sync`` is called by :meth:`optim_context`, and it is
        called before :meth:`backward` which means ``self._inner_count += 1``
        has not happened yet. Therefore, ``self._inner_count += 1`` should be
        performed manually here.

        Returns:
            bool: Whether to block the automatic gradient synchronization.
        """
        return (self._inner_count + 1) % self._accumulative_counts == 0 or (
            self._inner_count + 1
        ) == self._max_counts

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Get scaled loss according to ``_accumulative_counts``,
        ``_inner_count`` and max_counts.

        Args:
            loss (torch.Tensor): Original loss calculated by model.

        Returns:
            loss (torch.Tensor): Scaled loss.
        """
        if self._accumulative_counts == 1:
            # update parameters without gradient accumulation. The gradient
            # should not be rescaled and `loss_factor=1`.
            loss_factor = 1
        elif self._max_counts == -1:
            loss_factor = self._accumulative_counts
        else:
            # if `self._accumulative_counts > 1`, the gradient needs to be
            # rescaled and accumulated. In most cases, `loss_factor` equals to
            # `self._accumulative_counts`. However, `self._max_counts` may not
            # be divisible by `self._accumulative_counts`, so the
            # `loss_scale` for the last few iterations needs to be
            # recalculated.
            if self._inner_count < self._max_counts - self._remainder_counts:
                loss_factor = self._accumulative_counts
            else:
                loss_factor = self._remainder_counts
            assert loss_factor > 0, (
                "loss_factor should be larger than zero! This error could "
                "happened when initialize_iter_status called with an "
                "error `init_counts` or `max_counts`"
            )

        loss = loss / loss_factor
        return loss

    @property
    def inner_count(self):
        """Get the number of updating parameters of optimizer wrapper."""
        return self._inner_count

    def __repr__(self):
        wrapper_info = (
            f"Type: {type(self).__name__}\n"
            f"_accumulative_counts: {self._accumulative_counts}\n"
            "optimizer: \n"
        )
        optimizer_str = repr(self.optimizer) + "\n"
        return wrapper_info + optimizer_str


def is_list_of(seq, expected_type):
    """Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)


class BaseDataElement:
    """A base data interface that supports Tensor-like and dict-like
    operations.

    A typical data elements refer to predicted results or ground truth labels
    on a task, such as predicted bboxes, instance masks, semantic
    segmentation masks, etc. Because groundtruth labels and predicted results
    often have similar properties (for example, the predicted bboxes and the
    groundtruth bboxes), MMEngine uses the same abstract data interface to
    encapsulate predicted results and groundtruth labels, and it is recommended
    to use different name conventions to distinguish them, such as using
    ``gt_instances`` and ``pred_instances`` to distinguish between labels and
    predicted results. Additionally, we distinguish data elements at instance
    level, pixel level, and label level. Each of these types has its own
    characteristics. Therefore, MMEngine defines the base class
    ``BaseDataElement``, and implement ``InstanceData``, ``PixelData``, and
    ``LabelData`` inheriting from ``BaseDataElement`` to represent different
    types of ground truth labels or predictions.

    Another common data element is sample data. A sample data consists of input
    data (such as an image) and its annotations and predictions. In general,
    an image can have multiple types of annotations and/or predictions at the
    same time (for example, both pixel-level semantic segmentation annotations
    and instance-level detection bboxes annotations). All labels and
    predictions of a training sample are often passed between Dataset, Model,
    Visualizer, and Evaluator components. In order to simplify the interface
    between components, we can treat them as a large data element and
    encapsulate them. Such data elements are generally called XXDataSample in
    the OpenMMLab. Therefore, Similar to `nn.Module`, the `BaseDataElement`
    allows `BaseDataElement` as its attribute. Such a class generally
    encapsulates all the data of a sample in the algorithm library, and its
    attributes generally are various types of data elements. For example,
    MMDetection is assigned by the BaseDataElement to encapsulate all the data
    elements of the sample labeling and prediction of a sample in the
    algorithm library.

    The attributes in ``BaseDataElement`` are divided into two parts,
    the ``metainfo`` and the ``data`` respectively.

        - ``metainfo``: Usually contains the
          information about the image such as filename,
          image_shape, pad_shape, etc. The attributes can be accessed or
          modified by dict-like or object-like operations, such as
          ``.`` (for data access and modification), ``in``, ``del``,
          ``pop(str)``, ``get(str)``, ``metainfo_keys()``,
          ``metainfo_values()``, ``metainfo_items()``, ``set_metainfo()`` (for
          set or change key-value pairs in metainfo).

        - ``data``: Annotations or model predictions are
          stored. The attributes can be accessed or modified by
          dict-like or object-like operations, such as
          ``.``, ``in``, ``del``, ``pop(str)``, ``get(str)``, ``keys()``,
          ``values()``, ``items()``. Users can also apply tensor-like
          methods to all :obj:`torch.Tensor` in the ``data_fields``,
          such as ``.cuda()``, ``.cpu()``, ``.numpy()``, ``.to()``,
          ``to_tensor()``, ``.detach()``.

    Args:
        metainfo (dict, optional): A dict contains the meta information
            of single image, such as ``dict(img_shape=(512, 512, 3),
            scale_factor=(1, 1, 1, 1))``. Defaults to None.
        kwargs (dict, optional): A dict contains annotations of single image or
            model predictions. Defaults to None.

    Examples:
        >>> import torch
        >>> from mmengine.structures import BaseDataElement
        >>> gt_instances = BaseDataElement()
        >>> bboxes = torch.rand((5, 4))
        >>> scores = torch.rand((5,))
        >>> img_id = 0
        >>> img_shape = (800, 1333)
        >>> gt_instances = BaseDataElement(
        ...     metainfo=dict(img_id=img_id, img_shape=img_shape),
        ...     bboxes=bboxes, scores=scores)
        >>> gt_instances = BaseDataElement(
        ...     metainfo=dict(img_id=img_id, img_shape=(640, 640)))

        >>> # new
        >>> gt_instances1 = gt_instances.new(
        ...     metainfo=dict(img_id=1, img_shape=(640, 640)),
        ...                   bboxes=torch.rand((5, 4)),
        ...                   scores=torch.rand((5,)))
        >>> gt_instances2 = gt_instances1.new()

        >>> # add and process property
        >>> gt_instances = BaseDataElement()
        >>> gt_instances.set_metainfo(dict(img_id=9, img_shape=(100, 100)))
        >>> assert 'img_shape' in gt_instances.metainfo_keys()
        >>> assert 'img_shape' in gt_instances
        >>> assert 'img_shape' not in gt_instances.keys()
        >>> assert 'img_shape' in gt_instances.all_keys()
        >>> print(gt_instances.img_shape)
        (100, 100)
        >>> gt_instances.scores = torch.rand((5,))
        >>> assert 'scores' in gt_instances.keys()
        >>> assert 'scores' in gt_instances
        >>> assert 'scores' in gt_instances.all_keys()
        >>> assert 'scores' not in gt_instances.metainfo_keys()
        >>> print(gt_instances.scores)
        tensor([0.5230, 0.7885, 0.2426, 0.3911, 0.4876])
        >>> gt_instances.bboxes = torch.rand((5, 4))
        >>> assert 'bboxes' in gt_instances.keys()
        >>> assert 'bboxes' in gt_instances
        >>> assert 'bboxes' in gt_instances.all_keys()
        >>> assert 'bboxes' not in gt_instances.metainfo_keys()
        >>> print(gt_instances.bboxes)
        tensor([[0.0900, 0.0424, 0.1755, 0.4469],
                [0.8648, 0.0592, 0.3484, 0.0913],
                [0.5808, 0.1909, 0.6165, 0.7088],
                [0.5490, 0.4209, 0.9416, 0.2374],
                [0.3652, 0.1218, 0.8805, 0.7523]])

        >>> # delete and change property
        >>> gt_instances = BaseDataElement(
        ...     metainfo=dict(img_id=0, img_shape=(640, 640)),
        ...     bboxes=torch.rand((6, 4)), scores=torch.rand((6,)))
        >>> gt_instances.set_metainfo(dict(img_shape=(1280, 1280)))
        >>> gt_instances.img_shape  # (1280, 1280)
        >>> gt_instances.bboxes = gt_instances.bboxes * 2
        >>> gt_instances.get('img_shape', None)  # (1280, 1280)
        >>> gt_instances.get('bboxes', None)  # 6x4 tensor
        >>> del gt_instances.img_shape
        >>> del gt_instances.bboxes
        >>> assert 'img_shape' not in gt_instances
        >>> assert 'bboxes' not in gt_instances
        >>> gt_instances.pop('img_shape', None)  # None
        >>> gt_instances.pop('bboxes', None)  # None

        >>> # Tensor-like
        >>> cuda_instances = gt_instances.cuda()
        >>> cuda_instances = gt_instances.to('cuda:0')
        >>> cpu_instances = cuda_instances.cpu()
        >>> cpu_instances = cuda_instances.to('cpu')
        >>> fp16_instances = cuda_instances.to(
        ...     device=None, dtype=torch.float16, non_blocking=False,
        ...     copy=False, memory_format=torch.preserve_format)
        >>> cpu_instances = cuda_instances.detach()
        >>> np_instances = cpu_instances.numpy()

        >>> # print
        >>> metainfo = dict(img_shape=(800, 1196, 3))
        >>> gt_instances = BaseDataElement(
        ...     metainfo=metainfo, det_labels=torch.LongTensor([0, 1, 2, 3]))
        >>> sample = BaseDataElement(metainfo=metainfo,
        ...                          gt_instances=gt_instances)
        >>> print(sample)
        <BaseDataElement(
            META INFORMATION
            img_shape: (800, 1196, 3)
            DATA FIELDS
            gt_instances: <BaseDataElement(
                    META INFORMATION
                    img_shape: (800, 1196, 3)
                    DATA FIELDS
                    det_labels: tensor([0, 1, 2, 3])
                ) at 0x7f0ec5eadc70>
        ) at 0x7f0fea49e130>

        >>> # inheritance
        >>> class DetDataSample(BaseDataElement):
        ...     @property
        ...     def proposals(self):
        ...         return self._proposals
        ...     @proposals.setter
        ...     def proposals(self, value):
        ...         self.set_field(value, '_proposals', dtype=BaseDataElement)
        ...     @proposals.deleter
        ...     def proposals(self):
        ...         del self._proposals
        ...     @property
        ...     def gt_instances(self):
        ...         return self._gt_instances
        ...     @gt_instances.setter
        ...     def gt_instances(self, value):
        ...         self.set_field(value, '_gt_instances',
        ...                        dtype=BaseDataElement)
        ...     @gt_instances.deleter
        ...     def gt_instances(self):
        ...         del self._gt_instances
        ...     @property
        ...     def pred_instances(self):
        ...         return self._pred_instances
        ...     @pred_instances.setter
        ...     def pred_instances(self, value):
        ...         self.set_field(value, '_pred_instances',
        ...                        dtype=BaseDataElement)
        ...     @pred_instances.deleter
        ...     def pred_instances(self):
        ...         del self._pred_instances
        >>> det_sample = DetDataSample()
        >>> proposals = BaseDataElement(bboxes=torch.rand((5, 4)))
        >>> det_sample.proposals = proposals
        >>> assert 'proposals' in det_sample
        >>> assert det_sample.proposals == proposals
        >>> del det_sample.proposals
        >>> assert 'proposals' not in det_sample
        >>> with self.assertRaises(AssertionError):
        ...     det_sample.proposals = torch.rand((5, 4))
    """

    def __init__(self, *, metainfo: Optional[dict] = None, **kwargs) -> None:

        self._metainfo_fields: set = set()
        self._data_fields: set = set()

        if metainfo is not None:
            self.set_metainfo(metainfo=metainfo)
        if kwargs:
            self.set_data(kwargs)

    def set_metainfo(self, metainfo: dict) -> None:
        """Set or change key-value pairs in ``metainfo_field`` by parameter
        ``metainfo``.

        Args:
            metainfo (dict): A dict contains the meta information
                of image, such as ``img_shape``, ``scale_factor``, etc.
        """
        assert isinstance(
            metainfo, dict
        ), f"metainfo should be a ``dict`` but got {type(metainfo)}"
        meta = copy.deepcopy(metainfo)
        for k, v in meta.items():
            self.set_field(name=k, value=v, field_type="metainfo", dtype=None)

    def set_data(self, data: dict) -> None:
        """Set or change key-value pairs in ``data_field`` by parameter
        ``data``.

        Args:
            data (dict): A dict contains annotations of image or
                model predictions.
        """
        assert isinstance(data, dict), f"data should be a `dict` but got {data}"
        for k, v in data.items():
            # Use `setattr()` rather than `self.set_field` to allow `set_data`
            # to set property method.
            setattr(self, k, v)

    def update(self, instance: "BaseDataElement") -> None:
        """The update() method updates the BaseDataElement with the elements
        from another BaseDataElement object.

        Args:
            instance (BaseDataElement): Another BaseDataElement object for
                update the current object.
        """
        assert isinstance(
            instance, BaseDataElement
        ), f"instance should be a `BaseDataElement` but got {type(instance)}"
        self.set_metainfo(dict(instance.metainfo_items()))
        self.set_data(dict(instance.items()))

    def new(self, *, metainfo: Optional[dict] = None, **kwargs) -> "BaseDataElement":
        """Return a new data element with same type. If ``metainfo`` and
        ``data`` are None, the new data element will have same metainfo and
        data. If metainfo or data is not None, the new result will overwrite it
        with the input value.

        Args:
            metainfo (dict, optional): A dict contains the meta information
                of image, such as ``img_shape``, ``scale_factor``, etc.
                Defaults to None.
            kwargs (dict): A dict contains annotations of image or
                model predictions.

        Returns:
            BaseDataElement: A new data element with same type.
        """
        new_data = self.__class__()

        if metainfo is not None:
            new_data.set_metainfo(metainfo)
        else:
            new_data.set_metainfo(dict(self.metainfo_items()))
        if kwargs:
            new_data.set_data(kwargs)
        else:
            new_data.set_data(dict(self.items()))
        return new_data

    def clone(self):
        """Deep copy the current data element.

        Returns:
            BaseDataElement: The copy of current data element.
        """
        clone_data = self.__class__()
        clone_data.set_metainfo(dict(self.metainfo_items()))
        clone_data.set_data(dict(self.items()))
        return clone_data

    def keys(self) -> list:
        """
        Returns:
            list: Contains all keys in data_fields.
        """
        # We assume that the name of the attribute related to property is
        # '_' + the name of the property. We use this rule to filter out
        # private keys.
        # TODO: Use a more robust way to solve this problem
        private_keys = {
            "_" + key
            for key in self._data_fields
            if isinstance(getattr(type(self), key, None), property)
        }
        return list(self._data_fields - private_keys)

    def metainfo_keys(self) -> list:
        """
        Returns:
            list: Contains all keys in metainfo_fields.
        """
        return list(self._metainfo_fields)

    def values(self) -> list:
        """
        Returns:
            list: Contains all values in data.
        """
        return [getattr(self, k) for k in self.keys()]

    def metainfo_values(self) -> list:
        """
        Returns:
            list: Contains all values in metainfo.
        """
        return [getattr(self, k) for k in self.metainfo_keys()]

    def all_keys(self) -> list:
        """
        Returns:
            list: Contains all keys in metainfo and data.
        """
        return self.metainfo_keys() + self.keys()

    def all_values(self) -> list:
        """
        Returns:
            list: Contains all values in metainfo and data.
        """
        return self.metainfo_values() + self.values()

    def all_items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``metainfo`` and ``data``.
        """
        for k in self.all_keys():
            yield (k, getattr(self, k))

    def items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``data``.
        """
        for k in self.keys():
            yield (k, getattr(self, k))

    def metainfo_items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``metainfo``.
        """
        for k in self.metainfo_keys():
            yield (k, getattr(self, k))

    @property
    def metainfo(self) -> dict:
        """dict: A dict contains metainfo of current data element."""
        return dict(self.metainfo_items())

    def __setattr__(self, name: str, value: Any):
        """setattr is only used to set data."""
        if name in ("_metainfo_fields", "_data_fields"):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(
                    f"{name} has been used as a "
                    "private attribute, which is immutable."
                )
        else:
            self.set_field(name=name, value=value, field_type="data", dtype=None)

    def __delattr__(self, item: str):
        """Delete the item in dataelement.

        Args:
            item (str): The key to delete.
        """
        if item in ("_metainfo_fields", "_data_fields"):
            raise AttributeError(
                f"{item} has been used as a " "private attribute, which is immutable."
            )
        super().__delattr__(item)
        if item in self._metainfo_fields:
            self._metainfo_fields.remove(item)
        elif item in self._data_fields:
            self._data_fields.remove(item)

    # dict-like methods
    __delitem__ = __delattr__

    def get(self, key, default=None) -> Any:
        """Get property in data and metainfo as the same as python."""
        # Use `getattr()` rather than `self.__dict__.get()` to allow getting
        # properties.
        return getattr(self, key, default)

    def pop(self, *args) -> Any:
        """Pop property in data and metainfo as the same as python."""
        assert len(args) < 3, "``pop`` get more than 2 arguments"
        name = args[0]
        if name in self._metainfo_fields:
            self._metainfo_fields.remove(args[0])
            return self.__dict__.pop(*args)

        elif name in self._data_fields:
            self._data_fields.remove(args[0])
            return self.__dict__.pop(*args)

        # with default value
        elif len(args) == 2:
            return args[1]
        else:
            # don't just use 'self.__dict__.pop(*args)' for only popping key in
            # metainfo or data
            raise KeyError(f"{args[0]} is not contained in metainfo or data")

    def __contains__(self, item: str) -> bool:
        """Whether the item is in dataelement.

        Args:
            item (str): The key to inquire.
        """
        return item in self._data_fields or item in self._metainfo_fields

    def set_field(
        self,
        value: Any,
        name: str,
        dtype: Optional[Union[Type, Tuple[Type, ...]]] = None,
        field_type: str = "data",
    ) -> None:
        """Special method for set union field, used as property.setter
        functions."""
        assert field_type in ["metainfo", "data"]
        if dtype is not None:
            assert isinstance(
                value, dtype
            ), f"{value} should be a {dtype} but got {type(value)}"

        if field_type == "metainfo":
            if name in self._data_fields:
                raise AttributeError(
                    f"Cannot set {name} to be a field of metainfo "
                    f"because {name} is already a data field"
                )
            self._metainfo_fields.add(name)
        else:
            if name in self._metainfo_fields:
                raise AttributeError(
                    f"Cannot set {name} to be a field of data "
                    f"because {name} is already a metainfo field"
                )
            self._data_fields.add(name)
        super().__setattr__(name, value)

    # Tensor-like methods
    def to(self, *args, **kwargs) -> "BaseDataElement":
        """Apply same name function to all tensors in data_fields."""
        new_data = self.new()
        for k, v in self.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def cpu(self) -> "BaseDataElement":
        """Convert all tensors to CPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cpu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def cuda(self) -> "BaseDataElement":
        """Convert all tensors to GPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cuda()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def npu(self) -> "BaseDataElement":
        """Convert all tensors to NPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.npu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def detach(self) -> "BaseDataElement":
        """Detach all tensors in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.detach()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def numpy(self) -> "BaseDataElement":
        """Convert all tensors to np.ndarray in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.detach().cpu().numpy()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def to_tensor(self) -> "BaseDataElement":
        """Convert all np.ndarray to tensor in data."""
        new_data = self.new()
        for k, v in self.items():
            data = {}
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
                data[k] = v
            elif isinstance(v, BaseDataElement):
                v = v.to_tensor()
                data[k] = v
            new_data.set_data(data)
        return new_data

    def to_dict(self) -> dict:
        """Convert BaseDataElement to dict."""
        return {
            k: v.to_dict() if isinstance(v, BaseDataElement) else v
            for k, v in self.all_items()
        }

    def __repr__(self) -> str:
        """Represent the object."""

        def _addindent(s_: str, num_spaces: int) -> str:
            """This func is modified from `pytorch` https://github.com/pytorch/
            pytorch/blob/b17b2b1cc7b017c3daaeff8cc7ec0f514d42ec37/torch/nn/modu
            les/module.py#L29.

            Args:
                s_ (str): The string to add spaces.
                num_spaces (int): The num of space to add.

            Returns:
                str: The string after add indent.
            """
            s = s_.split("\n")
            # don't do anything for single-line stuff
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)  # type: ignore
            s = first + "\n" + s  # type: ignore
            return s  # type: ignore

        def dump(obj: Any) -> str:
            """Represent the object.

            Args:
                obj (Any): The obj to represent.

            Returns:
                str: The represented str.
            """
            _repr = ""
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _repr += f"\n{k}: {_addindent(dump(v), 4)}"
            elif isinstance(obj, BaseDataElement):
                _repr += "\n\n    META INFORMATION"
                metainfo_items = dict(obj.metainfo_items())
                _repr += _addindent(dump(metainfo_items), 4)
                _repr += "\n\n    DATA FIELDS"
                items = dict(obj.items())
                _repr += _addindent(dump(items), 4)
                classname = obj.__class__.__name__
                _repr = f"<{classname}({_repr}\n) at {hex(id(obj))}>"
            else:
                _repr += repr(obj)
            return _repr

        return dump(self)

# multi-batch inputs processed by different augmentations from the same batch.
EnhancedBatchInputs = List[Union[torch.Tensor, List[torch.Tensor]]]
# multi-batch data samples processed by different augmentations from the same
# batch. The inner list stands for different augmentations and the outer list
# stands for batch.
EnhancedBatchDataSamples = List[List[BaseDataElement]]
DATA_BATCH = Union[
    Dict[str, Union[EnhancedBatchInputs, EnhancedBatchDataSamples]], tuple, dict
]
MergedDataSamples = List[BaseDataElement]
CastData = Union[tuple, dict, BaseDataElement, torch.Tensor, list, bytes, str,
                 None]
@MODELS.register_module()
class BaseDataPreprocessor(nn.Module):
    """Base data pre-processor used for copying data to the target device.

    Subclasses inherit from ``BaseDataPreprocessor`` could override the
    forward method to implement custom data pre-processing, such as
    batch-resize, MixUp, or CutMix.

    Args:
        non_blocking (bool): Whether block current process
            when transferring data to device.
            New in version 0.3.0.

    Note:
        Data dictionary returned by dataloader must be a dict and at least
        contain the ``inputs`` key.
    """

    def __init__(self, non_blocking: Optional[bool] = False):
        super().__init__()
        self._non_blocking = non_blocking
        self._device = torch.device("cpu")

    def cast_data(self, data: CastData) -> CastData:
        """Copying data to the target device.

        Args:
            data (dict): Data returned by ``DataLoader``.

        Returns:
            CollatedResult: Inputs and data sample at target device.
        """
        if isinstance(data, Mapping):
            return {key: self.cast_data(data[key]) for key in data}
        elif isinstance(data, (str, bytes)) or data is None:
            return data
        elif isinstance(data, tuple) and hasattr(data, "_fields"):
            # namedtuple
            return type(data)(*(self.cast_data(sample) for sample in data))  # type: ignore  # noqa: E501  # yapf:disable
        elif isinstance(data, Sequence):
            return type(data)(self.cast_data(sample) for sample in data)  # type: ignore  # noqa: E501  # yapf:disable
        elif isinstance(data, (torch.Tensor, BaseDataElement)):
            return data.to(self.device, non_blocking=self._non_blocking)
        else:
            return data

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        """Preprocesses the data into the model input format.

        After the data pre-processing of :meth:`cast_data`, ``forward``
        will stack the input tensor list to a batch tensor at the first
        dimension.

        Args:
            data (dict): Data returned by dataloader
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict or list: Data in the same format as the model input.
        """
        return self.cast_data(data)  # type: ignore

    @property
    def device(self):
        return self._device

    def to(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """

        # Since Torch has not officially merged
        # the npu-related fields, using the _parse_to function
        # directly will cause the NPU to not be found.
        # Here, the input parameters are processed to avoid errors.
        if args and isinstance(args[0], str) and "npu" in args[0]:
            args = tuple([list(args)[0].replace("npu", torch.npu.native_device)])
        if kwargs and "npu" in str(kwargs.get("device", "")):
            kwargs["device"] = kwargs["device"].replace("npu", torch.npu.native_device)

        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._device = torch.device(device)
        return super().to(*args, **kwargs)

    def cuda(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(torch.cuda.current_device())
        return super().cuda()

    def npu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(torch.npu.current_device())
        return super().npu()

    def cpu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device("cpu")
        return super().cpu()


MODEL_WRAPPERS = Registry("model_wrapper")
def is_model_wrapper(model: nn.Module, registry: Registry = MODEL_WRAPPERS):
    """Check if a module is a model wrapper.

    The following 4 model in MMEngine (and their subclasses) are regarded as
    model wrappers: DataParallel, DistributedDataParallel,
    MMDataParallel, MMDistributedDataParallel. You may add you own
    model wrapper by registering it to ``mmengine.registry.MODEL_WRAPPERS``.

    Args:
        model (nn.Module): The model to be checked.
        registry (Registry): The parent registry to search for model wrappers.

    Returns:
        bool: True if the input model is a model wrapper.
    """
    module_wrappers = tuple(registry.module_dict.values())
    if isinstance(model, module_wrappers):
        return True

    if not registry.children:
        return False

    return any(is_model_wrapper(model, child) for child in registry.children.values())



class BaseDataElement:
    """A base data interface that supports Tensor-like and dict-like
    operations.

    A typical data elements refer to predicted results or ground truth labels
    on a task, such as predicted bboxes, instance masks, semantic
    segmentation masks, etc. Because groundtruth labels and predicted results
    often have similar properties (for example, the predicted bboxes and the
    groundtruth bboxes), MMEngine uses the same abstract data interface to
    encapsulate predicted results and groundtruth labels, and it is recommended
    to use different name conventions to distinguish them, such as using
    ``gt_instances`` and ``pred_instances`` to distinguish between labels and
    predicted results. Additionally, we distinguish data elements at instance
    level, pixel level, and label level. Each of these types has its own
    characteristics. Therefore, MMEngine defines the base class
    ``BaseDataElement``, and implement ``InstanceData``, ``PixelData``, and
    ``LabelData`` inheriting from ``BaseDataElement`` to represent different
    types of ground truth labels or predictions.

    Another common data element is sample data. A sample data consists of input
    data (such as an image) and its annotations and predictions. In general,
    an image can have multiple types of annotations and/or predictions at the
    same time (for example, both pixel-level semantic segmentation annotations
    and instance-level detection bboxes annotations). All labels and
    predictions of a training sample are often passed between Dataset, Model,
    Visualizer, and Evaluator components. In order to simplify the interface
    between components, we can treat them as a large data element and
    encapsulate them. Such data elements are generally called XXDataSample in
    the OpenMMLab. Therefore, Similar to `nn.Module`, the `BaseDataElement`
    allows `BaseDataElement` as its attribute. Such a class generally
    encapsulates all the data of a sample in the algorithm library, and its
    attributes generally are various types of data elements. For example,
    MMDetection is assigned by the BaseDataElement to encapsulate all the data
    elements of the sample labeling and prediction of a sample in the
    algorithm library.

    The attributes in ``BaseDataElement`` are divided into two parts,
    the ``metainfo`` and the ``data`` respectively.

        - ``metainfo``: Usually contains the
          information about the image such as filename,
          image_shape, pad_shape, etc. The attributes can be accessed or
          modified by dict-like or object-like operations, such as
          ``.`` (for data access and modification), ``in``, ``del``,
          ``pop(str)``, ``get(str)``, ``metainfo_keys()``,
          ``metainfo_values()``, ``metainfo_items()``, ``set_metainfo()`` (for
          set or change key-value pairs in metainfo).

        - ``data``: Annotations or model predictions are
          stored. The attributes can be accessed or modified by
          dict-like or object-like operations, such as
          ``.``, ``in``, ``del``, ``pop(str)``, ``get(str)``, ``keys()``,
          ``values()``, ``items()``. Users can also apply tensor-like
          methods to all :obj:`torch.Tensor` in the ``data_fields``,
          such as ``.cuda()``, ``.cpu()``, ``.numpy()``, ``.to()``,
          ``to_tensor()``, ``.detach()``.

    Args:
        metainfo (dict, optional): A dict contains the meta information
            of single image, such as ``dict(img_shape=(512, 512, 3),
            scale_factor=(1, 1, 1, 1))``. Defaults to None.
        kwargs (dict, optional): A dict contains annotations of single image or
            model predictions. Defaults to None.

    Examples:
        >>> import torch
        >>> from mmengine.structures import BaseDataElement
        >>> gt_instances = BaseDataElement()
        >>> bboxes = torch.rand((5, 4))
        >>> scores = torch.rand((5,))
        >>> img_id = 0
        >>> img_shape = (800, 1333)
        >>> gt_instances = BaseDataElement(
        ...     metainfo=dict(img_id=img_id, img_shape=img_shape),
        ...     bboxes=bboxes, scores=scores)
        >>> gt_instances = BaseDataElement(
        ...     metainfo=dict(img_id=img_id, img_shape=(640, 640)))

        >>> # new
        >>> gt_instances1 = gt_instances.new(
        ...     metainfo=dict(img_id=1, img_shape=(640, 640)),
        ...                   bboxes=torch.rand((5, 4)),
        ...                   scores=torch.rand((5,)))
        >>> gt_instances2 = gt_instances1.new()

        >>> # add and process property
        >>> gt_instances = BaseDataElement()
        >>> gt_instances.set_metainfo(dict(img_id=9, img_shape=(100, 100)))
        >>> assert 'img_shape' in gt_instances.metainfo_keys()
        >>> assert 'img_shape' in gt_instances
        >>> assert 'img_shape' not in gt_instances.keys()
        >>> assert 'img_shape' in gt_instances.all_keys()
        >>> print(gt_instances.img_shape)
        (100, 100)
        >>> gt_instances.scores = torch.rand((5,))
        >>> assert 'scores' in gt_instances.keys()
        >>> assert 'scores' in gt_instances
        >>> assert 'scores' in gt_instances.all_keys()
        >>> assert 'scores' not in gt_instances.metainfo_keys()
        >>> print(gt_instances.scores)
        tensor([0.5230, 0.7885, 0.2426, 0.3911, 0.4876])
        >>> gt_instances.bboxes = torch.rand((5, 4))
        >>> assert 'bboxes' in gt_instances.keys()
        >>> assert 'bboxes' in gt_instances
        >>> assert 'bboxes' in gt_instances.all_keys()
        >>> assert 'bboxes' not in gt_instances.metainfo_keys()
        >>> print(gt_instances.bboxes)
        tensor([[0.0900, 0.0424, 0.1755, 0.4469],
                [0.8648, 0.0592, 0.3484, 0.0913],
                [0.5808, 0.1909, 0.6165, 0.7088],
                [0.5490, 0.4209, 0.9416, 0.2374],
                [0.3652, 0.1218, 0.8805, 0.7523]])

        >>> # delete and change property
        >>> gt_instances = BaseDataElement(
        ...     metainfo=dict(img_id=0, img_shape=(640, 640)),
        ...     bboxes=torch.rand((6, 4)), scores=torch.rand((6,)))
        >>> gt_instances.set_metainfo(dict(img_shape=(1280, 1280)))
        >>> gt_instances.img_shape  # (1280, 1280)
        >>> gt_instances.bboxes = gt_instances.bboxes * 2
        >>> gt_instances.get('img_shape', None)  # (1280, 1280)
        >>> gt_instances.get('bboxes', None)  # 6x4 tensor
        >>> del gt_instances.img_shape
        >>> del gt_instances.bboxes
        >>> assert 'img_shape' not in gt_instances
        >>> assert 'bboxes' not in gt_instances
        >>> gt_instances.pop('img_shape', None)  # None
        >>> gt_instances.pop('bboxes', None)  # None

        >>> # Tensor-like
        >>> cuda_instances = gt_instances.cuda()
        >>> cuda_instances = gt_instances.to('cuda:0')
        >>> cpu_instances = cuda_instances.cpu()
        >>> cpu_instances = cuda_instances.to('cpu')
        >>> fp16_instances = cuda_instances.to(
        ...     device=None, dtype=torch.float16, non_blocking=False,
        ...     copy=False, memory_format=torch.preserve_format)
        >>> cpu_instances = cuda_instances.detach()
        >>> np_instances = cpu_instances.numpy()

        >>> # print
        >>> metainfo = dict(img_shape=(800, 1196, 3))
        >>> gt_instances = BaseDataElement(
        ...     metainfo=metainfo, det_labels=torch.LongTensor([0, 1, 2, 3]))
        >>> sample = BaseDataElement(metainfo=metainfo,
        ...                          gt_instances=gt_instances)
        >>> print(sample)
        <BaseDataElement(
            META INFORMATION
            img_shape: (800, 1196, 3)
            DATA FIELDS
            gt_instances: <BaseDataElement(
                    META INFORMATION
                    img_shape: (800, 1196, 3)
                    DATA FIELDS
                    det_labels: tensor([0, 1, 2, 3])
                ) at 0x7f0ec5eadc70>
        ) at 0x7f0fea49e130>

        >>> # inheritance
        >>> class DetDataSample(BaseDataElement):
        ...     @property
        ...     def proposals(self):
        ...         return self._proposals
        ...     @proposals.setter
        ...     def proposals(self, value):
        ...         self.set_field(value, '_proposals', dtype=BaseDataElement)
        ...     @proposals.deleter
        ...     def proposals(self):
        ...         del self._proposals
        ...     @property
        ...     def gt_instances(self):
        ...         return self._gt_instances
        ...     @gt_instances.setter
        ...     def gt_instances(self, value):
        ...         self.set_field(value, '_gt_instances',
        ...                        dtype=BaseDataElement)
        ...     @gt_instances.deleter
        ...     def gt_instances(self):
        ...         del self._gt_instances
        ...     @property
        ...     def pred_instances(self):
        ...         return self._pred_instances
        ...     @pred_instances.setter
        ...     def pred_instances(self, value):
        ...         self.set_field(value, '_pred_instances',
        ...                        dtype=BaseDataElement)
        ...     @pred_instances.deleter
        ...     def pred_instances(self):
        ...         del self._pred_instances
        >>> det_sample = DetDataSample()
        >>> proposals = BaseDataElement(bboxes=torch.rand((5, 4)))
        >>> det_sample.proposals = proposals
        >>> assert 'proposals' in det_sample
        >>> assert det_sample.proposals == proposals
        >>> del det_sample.proposals
        >>> assert 'proposals' not in det_sample
        >>> with self.assertRaises(AssertionError):
        ...     det_sample.proposals = torch.rand((5, 4))
    """

    def __init__(self, *, metainfo: Optional[dict] = None, **kwargs) -> None:

        self._metainfo_fields: set = set()
        self._data_fields: set = set()

        if metainfo is not None:
            self.set_metainfo(metainfo=metainfo)
        if kwargs:
            self.set_data(kwargs)

    def set_metainfo(self, metainfo: dict) -> None:
        """Set or change key-value pairs in ``metainfo_field`` by parameter
        ``metainfo``.

        Args:
            metainfo (dict): A dict contains the meta information
                of image, such as ``img_shape``, ``scale_factor``, etc.
        """
        assert isinstance(
            metainfo, dict
        ), f"metainfo should be a ``dict`` but got {type(metainfo)}"
        meta = copy.deepcopy(metainfo)
        for k, v in meta.items():
            self.set_field(name=k, value=v, field_type="metainfo", dtype=None)

    def set_data(self, data: dict) -> None:
        """Set or change key-value pairs in ``data_field`` by parameter
        ``data``.

        Args:
            data (dict): A dict contains annotations of image or
                model predictions.
        """
        assert isinstance(data, dict), f"data should be a `dict` but got {data}"
        for k, v in data.items():
            # Use `setattr()` rather than `self.set_field` to allow `set_data`
            # to set property method.
            setattr(self, k, v)

    def update(self, instance: "BaseDataElement") -> None:
        """The update() method updates the BaseDataElement with the elements
        from another BaseDataElement object.

        Args:
            instance (BaseDataElement): Another BaseDataElement object for
                update the current object.
        """
        assert isinstance(
            instance, BaseDataElement
        ), f"instance should be a `BaseDataElement` but got {type(instance)}"
        self.set_metainfo(dict(instance.metainfo_items()))
        self.set_data(dict(instance.items()))

    def new(self, *, metainfo: Optional[dict] = None, **kwargs) -> "BaseDataElement":
        """Return a new data element with same type. If ``metainfo`` and
        ``data`` are None, the new data element will have same metainfo and
        data. If metainfo or data is not None, the new result will overwrite it
        with the input value.

        Args:
            metainfo (dict, optional): A dict contains the meta information
                of image, such as ``img_shape``, ``scale_factor``, etc.
                Defaults to None.
            kwargs (dict): A dict contains annotations of image or
                model predictions.

        Returns:
            BaseDataElement: A new data element with same type.
        """
        new_data = self.__class__()

        if metainfo is not None:
            new_data.set_metainfo(metainfo)
        else:
            new_data.set_metainfo(dict(self.metainfo_items()))
        if kwargs:
            new_data.set_data(kwargs)
        else:
            new_data.set_data(dict(self.items()))
        return new_data

    def clone(self):
        """Deep copy the current data element.

        Returns:
            BaseDataElement: The copy of current data element.
        """
        clone_data = self.__class__()
        clone_data.set_metainfo(dict(self.metainfo_items()))
        clone_data.set_data(dict(self.items()))
        return clone_data

    def keys(self) -> list:
        """
        Returns:
            list: Contains all keys in data_fields.
        """
        # We assume that the name of the attribute related to property is
        # '_' + the name of the property. We use this rule to filter out
        # private keys.
        # TODO: Use a more robust way to solve this problem
        private_keys = {
            "_" + key
            for key in self._data_fields
            if isinstance(getattr(type(self), key, None), property)
        }
        return list(self._data_fields - private_keys)

    def metainfo_keys(self) -> list:
        """
        Returns:
            list: Contains all keys in metainfo_fields.
        """
        return list(self._metainfo_fields)

    def values(self) -> list:
        """
        Returns:
            list: Contains all values in data.
        """
        return [getattr(self, k) for k in self.keys()]

    def metainfo_values(self) -> list:
        """
        Returns:
            list: Contains all values in metainfo.
        """
        return [getattr(self, k) for k in self.metainfo_keys()]

    def all_keys(self) -> list:
        """
        Returns:
            list: Contains all keys in metainfo and data.
        """
        return self.metainfo_keys() + self.keys()

    def all_values(self) -> list:
        """
        Returns:
            list: Contains all values in metainfo and data.
        """
        return self.metainfo_values() + self.values()

    def all_items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``metainfo`` and ``data``.
        """
        for k in self.all_keys():
            yield (k, getattr(self, k))

    def items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``data``.
        """
        for k in self.keys():
            yield (k, getattr(self, k))

    def metainfo_items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``metainfo``.
        """
        for k in self.metainfo_keys():
            yield (k, getattr(self, k))

    @property
    def metainfo(self) -> dict:
        """dict: A dict contains metainfo of current data element."""
        return dict(self.metainfo_items())

    def __setattr__(self, name: str, value: Any):
        """setattr is only used to set data."""
        if name in ("_metainfo_fields", "_data_fields"):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(
                    f"{name} has been used as a "
                    "private attribute, which is immutable."
                )
        else:
            self.set_field(name=name, value=value, field_type="data", dtype=None)

    def __delattr__(self, item: str):
        """Delete the item in dataelement.

        Args:
            item (str): The key to delete.
        """
        if item in ("_metainfo_fields", "_data_fields"):
            raise AttributeError(
                f"{item} has been used as a " "private attribute, which is immutable."
            )
        super().__delattr__(item)
        if item in self._metainfo_fields:
            self._metainfo_fields.remove(item)
        elif item in self._data_fields:
            self._data_fields.remove(item)

    # dict-like methods
    __delitem__ = __delattr__

    def get(self, key, default=None) -> Any:
        """Get property in data and metainfo as the same as python."""
        # Use `getattr()` rather than `self.__dict__.get()` to allow getting
        # properties.
        return getattr(self, key, default)

    def pop(self, *args) -> Any:
        """Pop property in data and metainfo as the same as python."""
        assert len(args) < 3, "``pop`` get more than 2 arguments"
        name = args[0]
        if name in self._metainfo_fields:
            self._metainfo_fields.remove(args[0])
            return self.__dict__.pop(*args)

        elif name in self._data_fields:
            self._data_fields.remove(args[0])
            return self.__dict__.pop(*args)

        # with default value
        elif len(args) == 2:
            return args[1]
        else:
            # don't just use 'self.__dict__.pop(*args)' for only popping key in
            # metainfo or data
            raise KeyError(f"{args[0]} is not contained in metainfo or data")

    def __contains__(self, item: str) -> bool:
        """Whether the item is in dataelement.

        Args:
            item (str): The key to inquire.
        """
        return item in self._data_fields or item in self._metainfo_fields

    def set_field(
        self,
        value: Any,
        name: str,
        dtype: Optional[Union[Type, Tuple[Type, ...]]] = None,
        field_type: str = "data",
    ) -> None:
        """Special method for set union field, used as property.setter
        functions."""
        assert field_type in ["metainfo", "data"]
        if dtype is not None:
            assert isinstance(
                value, dtype
            ), f"{value} should be a {dtype} but got {type(value)}"

        if field_type == "metainfo":
            if name in self._data_fields:
                raise AttributeError(
                    f"Cannot set {name} to be a field of metainfo "
                    f"because {name} is already a data field"
                )
            self._metainfo_fields.add(name)
        else:
            if name in self._metainfo_fields:
                raise AttributeError(
                    f"Cannot set {name} to be a field of data "
                    f"because {name} is already a metainfo field"
                )
            self._data_fields.add(name)
        super().__setattr__(name, value)

    # Tensor-like methods
    def to(self, *args, **kwargs) -> "BaseDataElement":
        """Apply same name function to all tensors in data_fields."""
        new_data = self.new()
        for k, v in self.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def cpu(self) -> "BaseDataElement":
        """Convert all tensors to CPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cpu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def cuda(self) -> "BaseDataElement":
        """Convert all tensors to GPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cuda()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def npu(self) -> "BaseDataElement":
        """Convert all tensors to NPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.npu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def detach(self) -> "BaseDataElement":
        """Detach all tensors in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.detach()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def numpy(self) -> "BaseDataElement":
        """Convert all tensors to np.ndarray in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.detach().cpu().numpy()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def to_tensor(self) -> "BaseDataElement":
        """Convert all np.ndarray to tensor in data."""
        new_data = self.new()
        for k, v in self.items():
            data = {}
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
                data[k] = v
            elif isinstance(v, BaseDataElement):
                v = v.to_tensor()
                data[k] = v
            new_data.set_data(data)
        return new_data

    def to_dict(self) -> dict:
        """Convert BaseDataElement to dict."""
        return {
            k: v.to_dict() if isinstance(v, BaseDataElement) else v
            for k, v in self.all_items()
        }

    def __repr__(self) -> str:
        """Represent the object."""

        def _addindent(s_: str, num_spaces: int) -> str:
            """This func is modified from `pytorch` https://github.com/pytorch/
            pytorch/blob/b17b2b1cc7b017c3daaeff8cc7ec0f514d42ec37/torch/nn/modu
            les/module.py#L29.

            Args:
                s_ (str): The string to add spaces.
                num_spaces (int): The num of space to add.

            Returns:
                str: The string after add indent.
            """
            s = s_.split("\n")
            # don't do anything for single-line stuff
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)  # type: ignore
            s = first + "\n" + s  # type: ignore
            return s  # type: ignore

        def dump(obj: Any) -> str:
            """Represent the object.

            Args:
                obj (Any): The obj to represent.

            Returns:
                str: The represented str.
            """
            _repr = ""
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _repr += f"\n{k}: {_addindent(dump(v), 4)}"
            elif isinstance(obj, BaseDataElement):
                _repr += "\n\n    META INFORMATION"
                metainfo_items = dict(obj.metainfo_items())
                _repr += _addindent(dump(metainfo_items), 4)
                _repr += "\n\n    DATA FIELDS"
                items = dict(obj.items())
                _repr += _addindent(dump(items), 4)
                classname = obj.__class__.__name__
                _repr = f"<{classname}({_repr}\n) at {hex(id(obj))}>"
            else:
                _repr += repr(obj)
            return _repr

        return dump(self)


@MODELS.register_module()
class BaseTTAModel(BaseModel):
    """Base model for inference with test-time augmentation.

    ``BaseTTAModel`` is a wrapper for inference given multi-batch data.
    It implements the :meth:`test_step` for multi-batch data inference.
    ``multi-batch`` data means data processed by different augmentation
    from the same batch.

    During test time augmentation, the data processed by
    :obj:`mmcv.transforms.TestTimeAug`, and then collated by
    ``pseudo_collate`` will have the following format:

    .. code-block::

        result = dict(
            inputs=[
                [image1_aug1, image2_aug1],
                [image1_aug2, image2_aug2]
            ],
            data_samples=[
                [data_sample1_aug1, data_sample2_aug1],
                [data_sample1_aug2, data_sample2_aug2],
            ]
        )

    ``image{i}_aug{j}`` means the i-th image of the batch, which is
    augmented by the j-th augmentation.

    ``BaseTTAModel`` will collate the data to:

     .. code-block::

        data1 = dict(
            inputs=[image1_aug1, image2_aug1],
            data_samples=[data_sample1_aug1, data_sample2_aug1]
        )

        data2 = dict(
            inputs=[image1_aug2, image2_aug2],
            data_samples=[data_sample1_aug2, data_sample2_aug2]
        )

    ``data1`` and ``data2`` will be passed to model, and the results will be
    merged by :meth:`merge_preds`.

    Note:
        :meth:`merge_preds` is an abstract method, all subclasses should
        implement it.

    Warning:
        If ``data_preprocessor`` is not None, it will overwrite the model's
        ``data_preprocessor``.

    Args:
        module (dict or nn.Module): Tested model.
        data_preprocessor (dict or :obj:`BaseDataPreprocessor`, optional):
            If model does not define ``data_preprocessor``, it will be the
            default value for model.
    """

    def __init__(
        self,
        module: Union[dict, nn.Module],
        data_preprocessor: Union[dict, nn.Module, None] = None,
    ):
        super().__init__()
        if isinstance(module, nn.Module):
            self.module = module
        elif isinstance(module, dict):
            if data_preprocessor is not None:
                module["data_preprocessor"] = data_preprocessor
            self.module = MODELS.build(module)
        else:
            raise TypeError(
                "The type of module should be a `nn.Module` "
                f"instance or a dict, but got {module}"
            )
        assert hasattr(
            self.module, "test_step"
        ), "Model wrapped by BaseTTAModel must implement `test_step`!"

    @abstractmethod
    def merge_preds(
        self, data_samples_list: EnhancedBatchDataSamples
    ) -> MergedDataSamples:
        """Merge predictions of enhanced data to one prediction.

        Args:
            data_samples_list (EnhancedBatchDataSamples): List of predictions
                of all enhanced data.

        Returns:
            List[BaseDataElement]: Merged prediction.
        """

    def test_step(self, data):
        """Get predictions of each enhanced data, a multiple predictions.

        Args:
            data (DataBatch): Enhanced data batch sampled from dataloader.

        Returns:
            MergedDataSamples: Merged prediction.
        """
        data_list: Union[List[dict], List[list]]
        if isinstance(data, dict):
            num_augs = len(data[next(iter(data))])
            data_list = [
                {key: value[idx] for key, value in data.items()}
                for idx in range(num_augs)
            ]
        elif isinstance(data, (tuple, list)):
            num_augs = len(data[0])
            data_list = [[_data[idx] for _data in data] for idx in range(num_augs)]
        else:
            raise TypeError(
                "data given by dataLoader should be a dict, "
                f"tuple or a list, but got {type(data)}"
            )

        predictions = []
        for data in data_list:  # type: ignore
            predictions.append(self.module.test_step(data))
        return self.merge_preds(list(zip(*predictions)))  # type: ignore

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[list] = None,
        mode: str = "tensor",
    ) -> Union[Dict[str, torch.Tensor], list]:
        """``BaseTTAModel.forward`` should not be called."""
        raise NotImplementedError(
            "`BaseTTAModel.forward` will not be called during training or"
            "testing. Please call `test_step` instead. If you want to use"
            "`BaseTTAModel.forward`, please implement this method"
        )


def get_dist_info(group: Optional[ProcessGroup] = None) -> Tuple[int, int]:
    """Get distributed information of the given process group.

    Note:
        Calling ``get_dist_info`` in non-distributed environment will return
        (0, 1).

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        tuple[int, int]: Return a tuple containing the ``rank`` and
        ``world_size``.
    """
    world_size = get_world_size(group)
    rank = get_rank(group)
    return rank, world_size


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Defaults to False.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=""):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_model_wrapper(module) or isinstance(module, BaseTTAModel):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            all_missing_keys,
            unexpected_keys,
            err_msg,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [key for key in all_missing_keys if "num_batches_tracked" not in key]

    if unexpected_keys:
        err_msg.append(
            "unexpected key in source " f'state_dict: {", ".join(unexpected_keys)}\n'
        )
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n'
        )

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, "The model and loaded state dict do not match exactly\n")
        err_msg = "\n".join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        else:
            print_log(err_msg, logger=logger, level=logging.WARNING)


def _load_checkpoint_to_model(model,
                              checkpoint,
                              strict=False,
                              logger=None,
                              revise_keys=[(r'^module\.', '')]):

    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
             for k, v in state_dict.items()})
    # Keep metadata in state_dict
    state_dict._metadata = metadata

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def load_checkpoint(
    model,
    filename,
    map_location=None,
    strict=False,
    logger=None,
    revise_keys=[(r"^module\.", "")],
):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Defaults to strip
            the prefix 'module.' by [(r'^module\\.', '')].

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")

    return _load_checkpoint_to_model(model, checkpoint, strict, logger, revise_keys)

DATASETS = Registry("dataset")

def init_detector(
    config: Union[str, Path, Config],
    checkpoint: Optional[str] = None,
    palette: str = "none",
    device: str = "cpu",
    cfg_options: Optional[dict] = None,
) -> nn.Module:
    """Initialize a detector from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        palette (str): Color palette used for visualization. If palette
            is stored in checkpoint, use checkpoint's palette first, otherwise
            use externally passed palette. Currently, supports 'coco', 'voc',
            'citys' and 'random'. Defaults to none.
        device (str): The device where the anchors will be put on.
            Defaults to cpu.
        cfg_options (dict, optional): Options to override some settings in
            the used config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError(
            "config must be a filename or Config object, " f"but got {type(config)}"
        )
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    elif "init_cfg" in config.model.backbone:
        config.model.backbone.init_cfg = None
    init_default_scope(config.get("default_scope", "mmdet"))

    model = MODELS.build(config.model)
    model = revert_sync_batchnorm(model)
    if checkpoint is None:
        warnings.simplefilter("once")
        warnings.warn("checkpoint is None, use COCO classes by default.")
        model.dataset_meta = {"classes": get_classes("coco")}
    else:
        checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")
        # Weights converted from elsewhere may not have meta fields.
        checkpoint_meta = checkpoint.get("meta", {})

        # save the dataset_meta in the model for convenience
        if "dataset_meta" in checkpoint_meta:
            # mmdet 3.x, all keys should be lowercase
            model.dataset_meta = {
                k.lower(): v for k, v in checkpoint_meta["dataset_meta"].items()
            }
        elif "CLASSES" in checkpoint_meta:
            # < mmdet 3.x
            classes = checkpoint_meta["CLASSES"]
            model.dataset_meta = {"classes": classes}
        else:
            warnings.simplefilter("once")
            warnings.warn(
                "dataset_meta or class names are not saved in the "
                "checkpoint's meta data, use COCO classes by default."
            )
            model.dataset_meta = {"classes": get_classes("coco")}

    # Priority:  args.palette -> config -> checkpoint
    if palette != "none":
        model.dataset_meta["palette"] = palette
    else:
        test_dataset_cfg = copy.deepcopy(config.test_dataloader.dataset)
        # lazy init. We only need the metainfo.
        test_dataset_cfg["lazy_init"] = True
        metainfo = DATASETS.build(test_dataset_cfg).metainfo
        cfg_palette = metainfo.get("palette", None)
        if cfg_palette is not None:
            model.dataset_meta["palette"] = cfg_palette
        else:
            if "palette" not in model.dataset_meta:
                warnings.warn(
                    "palette does not exist, random is used by default. "
                    "You can also set the palette to customize."
                )
                model.dataset_meta["palette"] = "random"

    model.cfg = config  # save the config in the model for convenience
    # model.to(device)
    model.eval()
    return model


IndexType = Union[
    str,
    slice,
    int,
    list,
    torch.LongTensor,
    torch.BoolTensor,
    np.ndarray,
]


# Modified from
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/data_structures/instance_data.py # noqa
class InstanceData(BaseDataElement):
    """Data structure for instance-level annotations or predictions.

    Subclass of :class:`BaseDataElement`. All value in `data_fields`
    should have the same length. This design refer to
    https://github.com/facebookresearch/detectron2/blob/master/detectron2/structures/instances.py # noqa E501
    InstanceData also support extra functions: ``index``, ``slice`` and ``cat`` for data field. The type of value
    in data field can be base data structure such as `torch.Tensor`, `numpy.ndarray`, `list`, `str`, `tuple`,
    and can be customized data structure that has ``__len__``, ``__getitem__`` and ``cat`` attributes.

    Examples:
        >>> # custom data structure
        >>> class TmpObject:
        ...     def __init__(self, tmp) -> None:
        ...         assert isinstance(tmp, list)
        ...         self.tmp = tmp
        ...     def __len__(self):
        ...         return len(self.tmp)
        ...     def __getitem__(self, item):
        ...         if isinstance(item, int):
        ...             if item >= len(self) or item < -len(self):  # type:ignore
        ...                 raise IndexError(f'Index {item} out of range!')
        ...             else:
        ...                 # keep the dimension
        ...                 item = slice(item, None, len(self))
        ...         return TmpObject(self.tmp[item])
        ...     @staticmethod
        ...     def cat(tmp_objs):
        ...         assert all(isinstance(results, TmpObject) for results in tmp_objs)
        ...         if len(tmp_objs) == 1:
        ...             return tmp_objs[0]
        ...         tmp_list = [tmp_obj.tmp for tmp_obj in tmp_objs]
        ...         tmp_list = list(itertools.chain(*tmp_list))
        ...         new_data = TmpObject(tmp_list)
        ...         return new_data
        ...     def __repr__(self):
        ...         return str(self.tmp)
        >>> from mmengine.structures import InstanceData
        >>> import numpy as np
        >>> import torch
        >>> img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
        >>> instance_data = InstanceData(metainfo=img_meta)
        >>> 'img_shape' in instance_data
        True
        >>> instance_data.det_labels = torch.LongTensor([2, 3])
        >>> instance_data["det_scores"] = torch.Tensor([0.8, 0.7])
        >>> instance_data.bboxes = torch.rand((2, 4))
        >>> instance_data.polygons = TmpObject([[1, 2, 3, 4], [5, 6, 7, 8]])
        >>> len(instance_data)
        2
        >>> print(instance_data)
        <InstanceData(
            META INFORMATION
            img_shape: (800, 1196, 3)
            pad_shape: (800, 1216, 3)
            DATA FIELDS
            det_labels: tensor([2, 3])
            det_scores: tensor([0.8000, 0.7000])
            bboxes: tensor([[0.4997, 0.7707, 0.0595, 0.4188],
                        [0.8101, 0.3105, 0.5123, 0.6263]])
            polygons: [[1, 2, 3, 4], [5, 6, 7, 8]]
        ) at 0x7fb492de6280>
        >>> sorted_results = instance_data[instance_data.det_scores.sort().indices]
        >>> sorted_results.det_scores
        tensor([0.7000, 0.8000])
        >>> print(instance_data[instance_data.det_scores > 0.75])
        <InstanceData(
            META INFORMATION
            img_shape: (800, 1196, 3)
            pad_shape: (800, 1216, 3)
            DATA FIELDS
            det_labels: tensor([2])
            det_scores: tensor([0.8000])
            bboxes: tensor([[0.4997, 0.7707, 0.0595, 0.4188]])
            polygons: [[1, 2, 3, 4]]
        ) at 0x7f64ecf0ec40>
        >>> print(instance_data[instance_data.det_scores > 1])
        <InstanceData(
            META INFORMATION
            img_shape: (800, 1196, 3)
            pad_shape: (800, 1216, 3)
            DATA FIELDS
            det_labels: tensor([], dtype=torch.int64)
            det_scores: tensor([])
            bboxes: tensor([], size=(0, 4))
            polygons: []
        ) at 0x7f660a6a7f70>
        >>> print(instance_data.cat([instance_data, instance_data]))
        <InstanceData(
            META INFORMATION
            img_shape: (800, 1196, 3)
            pad_shape: (800, 1216, 3)
            DATA FIELDS
            det_labels: tensor([2, 3, 2, 3])
            det_scores: tensor([0.8000, 0.7000, 0.8000, 0.7000])
            bboxes: tensor([[0.4997, 0.7707, 0.0595, 0.4188],
                        [0.8101, 0.3105, 0.5123, 0.6263],
                        [0.4997, 0.7707, 0.0595, 0.4188],
                        [0.8101, 0.3105, 0.5123, 0.6263]])
            polygons: [[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8]]
        ) at 0x7f203542feb0>
    """

    def __setattr__(self, name: str, value: Sized):
        """setattr is only used to set data.

        The value must have the attribute of `__len__` and have the same length
        of `InstanceData`.
        """
        if name in ("_metainfo_fields", "_data_fields"):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(
                    f"{name} has been used as a "
                    "private attribute, which is immutable."
                )

        else:
            assert isinstance(value, Sized), "value must contain `__len__` attribute"

            if len(self) > 0:
                assert len(value) == len(self), (
                    "The length of "
                    f"values {len(value)} is "
                    "not consistent with "
                    "the length of this "
                    ":obj:`InstanceData` "
                    f"{len(self)}"
                )
            super().__setattr__(name, value)

    __setitem__ = __setattr__

    def __getitem__(self, item: IndexType) -> "InstanceData":
        """
        Args:
            item (str, int, list, :obj:`slice`, :obj:`numpy.ndarray`,
                :obj:`torch.LongTensor`, :obj:`torch.BoolTensor`):
                Get the corresponding values according to item.

        Returns:
            :obj:`InstanceData`: Corresponding values.
        """
        if isinstance(item, list):
            item = np.array(item)
        if isinstance(item, np.ndarray):
            # The default int type of numpy is platform dependent, int32 for
            # windows and int64 for linux. `torch.Tensor` requires the index
            # should be int64, therefore we simply convert it to int64 here.
            # More details in https://github.com/numpy/numpy/issues/9464
            item = item.astype(np.int64) if item.dtype == np.int32 else item
            item = torch.from_numpy(item)
        assert isinstance(
            item,
            (
                str,
                slice,
                int,
                torch.LongTensor,
                torch.BoolTensor,
            ),
        )

        if isinstance(item, str):
            return getattr(self, item)

        if isinstance(item, int):
            if item >= len(self) or item < -len(self):  # type:ignore
                raise IndexError(f"Index {item} out of range!")
            else:
                # keep the dimension
                item = slice(item, None, len(self))

        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, torch.Tensor):
            assert item.dim() == 1, (
                "Only support to get the" " values along the first dimension."
            )
            if isinstance(item, (torch.BoolTensor, torch.cuda.BoolTensor)):
                assert len(item) == len(self), (
                    "The shape of the "
                    "input(BoolTensor) "
                    f"{len(item)} "
                    "does not match the shape "
                    "of the indexed tensor "
                    "in results_field "
                    f"{len(self)} at "
                    "first dimension."
                )

            for k, v in self.items():
                if isinstance(v, torch.Tensor):
                    new_data[k] = v[item]
                elif isinstance(v, np.ndarray):
                    new_data[k] = v[item.cpu().numpy()]
                elif isinstance(v, (str, list, tuple)) or (
                    hasattr(v, "__getitem__") and hasattr(v, "cat")
                ):
                    # convert to indexes from BoolTensor
                    if isinstance(item, (torch.BoolTensor, torch.cuda.BoolTensor)):
                        indexes = torch.nonzero(item).view(-1).cpu().numpy().tolist()
                    else:
                        indexes = item.cpu().numpy().tolist()
                    slice_list = []
                    if indexes:
                        for index in indexes:
                            slice_list.append(slice(index, None, len(v)))
                    else:
                        slice_list.append(slice(None, 0, None))
                    r_list = [v[s] for s in slice_list]
                    if isinstance(v, (str, list, tuple)):
                        new_value = r_list[0]
                        for r in r_list[1:]:
                            new_value = new_value + r
                    else:
                        new_value = v.cat(r_list)
                    new_data[k] = new_value
                else:
                    raise ValueError(
                        f"The type of `{k}` is `{type(v)}`, which has no "
                        "attribute of `cat`, so it does not "
                        "support slice with `bool`"
                    )

        else:
            # item is a slice
            for k, v in self.items():
                new_data[k] = v[item]
        return new_data  # type:ignore

    @staticmethod
    def cat(instances_list: List["InstanceData"]) -> "InstanceData":
        """Concat the instances of all :obj:`InstanceData` in the list.

        Note: To ensure that cat returns as expected, make sure that
        all elements in the list must have exactly the same keys.

        Args:
            instances_list (list[:obj:`InstanceData`]): A list
                of :obj:`InstanceData`.

        Returns:
            :obj:`InstanceData`
        """
        assert all(isinstance(results, InstanceData) for results in instances_list)
        assert len(instances_list) > 0
        if len(instances_list) == 1:
            return instances_list[0]

        # metainfo and data_fields must be exactly the
        # same for each element to avoid exceptions.
        field_keys_list = [instances.all_keys() for instances in instances_list]
        assert len({len(field_keys) for field_keys in field_keys_list}) == 1 and len(
            set(itertools.chain(*field_keys_list))
        ) == len(field_keys_list[0]), (
            "There are different keys in "
            "`instances_list`, which may "
            "cause the cat operation "
            "to fail. Please make sure all "
            "elements in `instances_list` "
            "have the exact same key."
        )

        new_data = instances_list[0].__class__(metainfo=instances_list[0].metainfo)
        for k in instances_list[0].keys():
            values = [results[k] for results in instances_list]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                new_values = torch.cat(values, dim=0)
            elif isinstance(v0, np.ndarray):
                new_values = np.concatenate(values, axis=0)
            elif isinstance(v0, (str, list, tuple)):
                new_values = v0[:]
                for v in values[1:]:
                    new_values += v
            elif hasattr(v0, "cat"):
                new_values = v0.cat(values)
            else:
                raise ValueError(
                    f"The type of `{k}` is `{type(v0)}` which has no "
                    "attribute of `cat`"
                )
            new_data[k] = new_values
        return new_data  # type:ignore

    def __len__(self) -> int:
        """int: The length of InstanceData."""
        if len(self._data_fields) > 0:
            return len(self.values()[0])
        else:
            return 0


class PixelData(BaseDataElement):
    """Data structure for pixel-level annotations or predictions.

    All data items in ``data_fields`` of ``PixelData`` meet the following
    requirements:

    - They all have 3 dimensions in orders of channel, height, and width.
    - They should have the same height and width.

    Examples:
        >>> metainfo = dict(
        ...     img_id=random.randint(0, 100),
        ...     img_shape=(random.randint(400, 600), random.randint(400, 600)))
        >>> image = np.random.randint(0, 255, (4, 20, 40))
        >>> featmap = torch.randint(0, 255, (10, 20, 40))
        >>> pixel_data = PixelData(metainfo=metainfo,
        ...                        image=image,
        ...                        featmap=featmap)
        >>> print(pixel_data.shape)
        (20, 40)

        >>> # slice
        >>> slice_data = pixel_data[10:20, 20:40]
        >>> assert slice_data.shape == (10, 20)
        >>> slice_data = pixel_data[10, 20]
        >>> assert slice_data.shape == (1, 1)

        >>> # set
        >>> pixel_data.map3 = torch.randint(0, 255, (20, 40))
        >>> assert tuple(pixel_data.map3.shape) == (1, 20, 40)
        >>> with self.assertRaises(AssertionError):
        ...     # The dimension must be 3 or 2
        ...     pixel_data.map2 = torch.randint(0, 255, (1, 3, 20, 40))
    """

    def __setattr__(self, name: str, value: Union[torch.Tensor, np.ndarray]):
        """Set attributes of ``PixelData``.

        If the dimension of value is 2 and its shape meet the demand, it
        will automatically expand its channel-dimension.

        Args:
            name (str): The key to access the value, stored in `PixelData`.
            value (Union[torch.Tensor, np.ndarray]): The value to store in.
                The type of value must be `torch.Tensor` or `np.ndarray`,
                and its shape must meet the requirements of `PixelData`.
        """
        if name in ("_metainfo_fields", "_data_fields"):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(
                    f"{name} has been used as a "
                    "private attribute, which is immutable."
                )

        else:
            assert isinstance(value, (torch.Tensor, np.ndarray)), (
                f"Can not set {type(value)}, only support"
                f" {(torch.Tensor, np.ndarray)}"
            )

            if self.shape:
                assert tuple(value.shape[-2:]) == self.shape, (
                    "The height and width of "
                    f"values {tuple(value.shape[-2:])} is "
                    "not consistent with "
                    "the shape of this "
                    ":obj:`PixelData` "
                    f"{self.shape}"
                )
            assert value.ndim in [
                2,
                3,
            ], f"The dim of value must be 2 or 3, but got {value.ndim}"
            if value.ndim == 2:
                value = value[None]
                warnings.warn(
                    "The shape of value will convert from "
                    f"{value.shape[-2:]} to {value.shape}"
                )
            super().__setattr__(name, value)

    # TODO torch.Long/bool
    def __getitem__(self, item: Sequence[Union[int, slice]]) -> "PixelData":
        """
        Args:
            item (Sequence[Union[int, slice]]): Get the corresponding values
                according to item.

        Returns:
            :obj:`PixelData`: Corresponding values.
        """

        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, tuple):

            assert len(item) == 2, "Only support to slice height and width"
            tmp_item: List[slice] = list()
            for index, single_item in enumerate(item[::-1]):
                if isinstance(single_item, int):
                    tmp_item.insert(0, slice(single_item, None, self.shape[-index - 1]))
                elif isinstance(single_item, slice):
                    tmp_item.insert(0, single_item)
                else:
                    raise TypeError(
                        "The type of element in input must be int or slice, "
                        f"but got {type(single_item)}"
                    )
            tmp_item.insert(0, slice(None, None, None))
            item = tuple(tmp_item)
            for k, v in self.items():
                setattr(new_data, k, v[item])
        else:
            raise TypeError(f"Unsupported type {type(item)} for slicing PixelData")
        return new_data

    @property
    def shape(self):
        """The shape of pixel data."""
        if len(self._data_fields) > 0:
            return tuple(self.values()[0].shape[-2:])
        else:
            return None

    # TODO padding, resize


class DetDataSample(BaseDataElement):
    """A data structure interface of MMDetection. They are used as interfaces
    between different components.

    The attributes in ``DetDataSample`` are divided into several parts:

        - ``proposals``(InstanceData): Region proposals used in two-stage
            detectors.
        - ``gt_instances``(InstanceData): Ground truth of instance annotations.
        - ``pred_instances``(InstanceData): Instances of model predictions.
        - ``ignored_instances``(InstanceData): Instances to be ignored during
            training/testing.
        - ``gt_panoptic_seg``(PixelData): Ground truth of panoptic
            segmentation.
        - ``pred_panoptic_seg``(PixelData): Prediction of panoptic
           segmentation.
        - ``gt_sem_seg``(PixelData): Ground truth of semantic segmentation.
        - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.

    Examples:
         >>> import torch
         >>> import numpy as np
         >>> from mmengine.structures import InstanceData
         >>> from mmdet.structures import DetDataSample

         >>> data_sample = DetDataSample()
         >>> img_meta = dict(img_shape=(800, 1196),
         ...                 pad_shape=(800, 1216))
         >>> gt_instances = InstanceData(metainfo=img_meta)
         >>> gt_instances.bboxes = torch.rand((5, 4))
         >>> gt_instances.labels = torch.rand((5,))
         >>> data_sample.gt_instances = gt_instances
         >>> assert 'img_shape' in data_sample.gt_instances.metainfo_keys()
         >>> len(data_sample.gt_instances)
         5
         >>> print(data_sample)
        <DetDataSample(

            META INFORMATION

            DATA FIELDS
            gt_instances: <InstanceData(

                    META INFORMATION
                    pad_shape: (800, 1216)
                    img_shape: (800, 1196)

                    DATA FIELDS
                    labels: tensor([0.8533, 0.1550, 0.5433, 0.7294, 0.5098])
                    bboxes:
                    tensor([[9.7725e-01, 5.8417e-01, 1.7269e-01, 6.5694e-01],
                            [1.7894e-01, 5.1780e-01, 7.0590e-01, 4.8589e-01],
                            [7.0392e-01, 6.6770e-01, 1.7520e-01, 1.4267e-01],
                            [2.2411e-01, 5.1962e-01, 9.6953e-01, 6.6994e-01],
                            [4.1338e-01, 2.1165e-01, 2.7239e-04, 6.8477e-01]])
                ) at 0x7f21fb1b9190>
        ) at 0x7f21fb1b9880>
         >>> pred_instances = InstanceData(metainfo=img_meta)
         >>> pred_instances.bboxes = torch.rand((5, 4))
         >>> pred_instances.scores = torch.rand((5,))
         >>> data_sample = DetDataSample(pred_instances=pred_instances)
         >>> assert 'pred_instances' in data_sample

         >>> data_sample = DetDataSample()
         >>> gt_instances_data = dict(
         ...                        bboxes=torch.rand(2, 4),
         ...                        labels=torch.rand(2),
         ...                        masks=np.random.rand(2, 2, 2))
         >>> gt_instances = InstanceData(**gt_instances_data)
         >>> data_sample.gt_instances = gt_instances
         >>> assert 'gt_instances' in data_sample
         >>> assert 'masks' in data_sample.gt_instances

         >>> data_sample = DetDataSample()
         >>> gt_panoptic_seg_data = dict(panoptic_seg=torch.rand(2, 4))
         >>> gt_panoptic_seg = PixelData(**gt_panoptic_seg_data)
         >>> data_sample.gt_panoptic_seg = gt_panoptic_seg
         >>> print(data_sample)
        <DetDataSample(

            META INFORMATION

            DATA FIELDS
            _gt_panoptic_seg: <BaseDataElement(

                    META INFORMATION

                    DATA FIELDS
                    panoptic_seg: tensor([[0.7586, 0.1262, 0.2892, 0.9341],
                                [0.3200, 0.7448, 0.1052, 0.5371]])
                ) at 0x7f66c2bb7730>
            gt_panoptic_seg: <BaseDataElement(

                    META INFORMATION

                    DATA FIELDS
                    panoptic_seg: tensor([[0.7586, 0.1262, 0.2892, 0.9341],
                                [0.3200, 0.7448, 0.1052, 0.5371]])
                ) at 0x7f66c2bb7730>
        ) at 0x7f66c2bb7280>
        >>> data_sample = DetDataSample()
        >>> gt_segm_seg_data = dict(segm_seg=torch.rand(2, 2, 2))
        >>> gt_segm_seg = PixelData(**gt_segm_seg_data)
        >>> data_sample.gt_segm_seg = gt_segm_seg
        >>> assert 'gt_segm_seg' in data_sample
        >>> assert 'segm_seg' in data_sample.gt_segm_seg
    """

    @property
    def proposals(self) -> InstanceData:
        return self._proposals

    @proposals.setter
    def proposals(self, value: InstanceData):
        self.set_field(value, "_proposals", dtype=InstanceData)

    @proposals.deleter
    def proposals(self):
        del self._proposals

    @property
    def gt_instances(self) -> InstanceData:
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData):
        self.set_field(value, "_gt_instances", dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    @property
    def pred_instances(self) -> InstanceData:
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData):
        self.set_field(value, "_pred_instances", dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances

    @property
    def ignored_instances(self) -> InstanceData:
        return self._ignored_instances

    @ignored_instances.setter
    def ignored_instances(self, value: InstanceData):
        self.set_field(value, "_ignored_instances", dtype=InstanceData)

    @ignored_instances.deleter
    def ignored_instances(self):
        del self._ignored_instances

    @property
    def gt_panoptic_seg(self) -> PixelData:
        return self._gt_panoptic_seg

    @gt_panoptic_seg.setter
    def gt_panoptic_seg(self, value: PixelData):
        self.set_field(value, "_gt_panoptic_seg", dtype=PixelData)

    @gt_panoptic_seg.deleter
    def gt_panoptic_seg(self):
        del self._gt_panoptic_seg

    @property
    def pred_panoptic_seg(self) -> PixelData:
        return self._pred_panoptic_seg

    @pred_panoptic_seg.setter
    def pred_panoptic_seg(self, value: PixelData):
        self.set_field(value, "_pred_panoptic_seg", dtype=PixelData)

    @pred_panoptic_seg.deleter
    def pred_panoptic_seg(self):
        del self._pred_panoptic_seg

    @property
    def gt_sem_seg(self) -> PixelData:
        return self._gt_sem_seg

    @gt_sem_seg.setter
    def gt_sem_seg(self, value: PixelData):
        self.set_field(value, "_gt_sem_seg", dtype=PixelData)

    @gt_sem_seg.deleter
    def gt_sem_seg(self):
        del self._gt_sem_seg

    @property
    def pred_sem_seg(self) -> PixelData:
        return self._pred_sem_seg

    @pred_sem_seg.setter
    def pred_sem_seg(self, value: PixelData):
        self.set_field(value, "_pred_sem_seg", dtype=PixelData)

    @pred_sem_seg.deleter
    def pred_sem_seg(self):
        del self._pred_sem_seg


SampleList = List[DetDataSample]
OptSampleList = Optional[SampleList]


class BaseInstanceMasks(metaclass=ABCMeta):
    """Base class for instance masks."""

    @abstractmethod
    def rescale(self, scale, interpolation="nearest"):
        """Rescale masks as large as possible while keeping the aspect ratio.
        For details can refer to `mmcv.imrescale`.

        Args:
            scale (tuple[int]): The maximum size (h, w) of rescaled mask.
            interpolation (str): Same as :func:`mmcv.imrescale`.

        Returns:
            BaseInstanceMasks: The rescaled masks.
        """

    @abstractmethod
    def resize(self, out_shape, interpolation="nearest"):
        """Resize masks to the given out_shape.

        Args:
            out_shape: Target (h, w) of resized mask.
            interpolation (str): See :func:`mmcv.imresize`.

        Returns:
            BaseInstanceMasks: The resized masks.
        """

    @abstractmethod
    def flip(self, flip_direction="horizontal"):
        """Flip masks alone the given direction.

        Args:
            flip_direction (str): Either 'horizontal' or 'vertical'.

        Returns:
            BaseInstanceMasks: The flipped masks.
        """

    @abstractmethod
    def pad(self, out_shape, pad_val):
        """Pad masks to the given size of (h, w).

        Args:
            out_shape (tuple[int]): Target (h, w) of padded mask.
            pad_val (int): The padded value.

        Returns:
            BaseInstanceMasks: The padded masks.
        """

    @abstractmethod
    def crop(self, bbox):
        """Crop each mask by the given bbox.

        Args:
            bbox (ndarray): Bbox in format [x1, y1, x2, y2], shape (4, ).

        Return:
            BaseInstanceMasks: The cropped masks.
        """

    @abstractmethod
    def crop_and_resize(
        self, bboxes, out_shape, inds, device, interpolation="bilinear", binarize=True
    ):
        """Crop and resize masks by the given bboxes.

        This function is mainly used in mask targets computation.
        It firstly align mask to bboxes by assigned_inds, then crop mask by the
        assigned bbox and resize to the size of (mask_h, mask_w)

        Args:
            bboxes (Tensor): Bboxes in format [x1, y1, x2, y2], shape (N, 4)
            out_shape (tuple[int]): Target (h, w) of resized mask
            inds (ndarray): Indexes to assign masks to each bbox,
                shape (N,) and values should be between [0, num_masks - 1].
            device (str): Device of bboxes
            interpolation (str): See `mmcv.imresize`
            binarize (bool): if True fractional values are rounded to 0 or 1
                after the resize operation. if False and unsupported an error
                will be raised. Defaults to True.

        Return:
            BaseInstanceMasks: the cropped and resized masks.
        """

    @abstractmethod
    def expand(self, expanded_h, expanded_w, top, left):
        """see :class:`Expand`."""

    @property
    @abstractmethod
    def areas(self):
        """ndarray: areas of each instance."""

    @abstractmethod
    def to_ndarray(self):
        """Convert masks to the format of ndarray.

        Return:
            ndarray: Converted masks in the format of ndarray.
        """

    @abstractmethod
    def to_tensor(self, dtype, device):
        """Convert masks to the format of Tensor.

        Args:
            dtype (str): Dtype of converted mask.
            device (torch.device): Device of converted masks.

        Returns:
            Tensor: Converted masks in the format of Tensor.
        """

    @abstractmethod
    def translate(
        self,
        out_shape,
        offset,
        direction="horizontal",
        border_value=0,
        interpolation="bilinear",
    ):
        """Translate the masks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            offset (int | float): The offset for translate.
            direction (str): The translate direction, either "horizontal"
                or "vertical".
            border_value (int | float): Border value. Default 0.
            interpolation (str): Same as :func:`mmcv.imtranslate`.

        Returns:
            Translated masks.
        """

    def shear(
        self,
        out_shape,
        magnitude,
        direction="horizontal",
        border_value=0,
        interpolation="bilinear",
    ):
        """Shear the masks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            magnitude (int | float): The magnitude used for shear.
            direction (str): The shear direction, either "horizontal"
                or "vertical".
            border_value (int | tuple[int]): Value used in case of a
                constant border. Default 0.
            interpolation (str): Same as in :func:`mmcv.imshear`.

        Returns:
            ndarray: Sheared masks.
        """

    @abstractmethod
    def rotate(self, out_shape, angle, center=None, scale=1.0, border_value=0):
        """Rotate the masks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            angle (int | float): Rotation angle in degrees. Positive values
                mean counter-clockwise rotation.
            center (tuple[float], optional): Center point (w, h) of the
                rotation in source image. If not specified, the center of
                the image will be used.
            scale (int | float): Isotropic scale factor.
            border_value (int | float): Border value. Default 0 for masks.

        Returns:
            Rotated masks.
        """

    def get_bboxes(self, dst_type="hbb"):
        """Get the certain type boxes from masks.

        Please refer to ``mmdet.structures.bbox.box_type`` for more details of
        the box type.

        Args:
            dst_type: Destination box type.

        Returns:
            :obj:`BaseBoxes`: Certain type boxes.
        """

        _, box_type_cls = get_box_type(dst_type)
        return box_type_cls.from_instance_masks(self)

    @classmethod
    @abstractmethod
    def cat(cls: Type[T], masks: Sequence[T]) -> T:
        """Concatenate a sequence of masks into one single mask instance.

        Args:
            masks (Sequence[T]): A sequence of mask instances.

        Returns:
            T: Concatenated mask instance.
        """


def _scale_size(
    size: Tuple[int, int],
    scale: Union[float, int, Tuple[float, float], Tuple[int, int]],
) -> Tuple[int, int]:
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | int | tuple(float) | tuple(int)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)


def rescale_size(old_size: tuple,
                 scale: Union[float, int, Tuple[int, int]],
                 return_scale: bool = False) -> tuple:
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | int | tuple[int]): The scaling factor or maximum size.
            If it is a float number or an integer, then the image will be
            rescaled by this factor, else if it is a tuple of 2 integers, then
            the image will be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size

if Image is not None:
    if hasattr(Image, 'Resampling'):
        pillow_interp_codes = {
            'nearest': Image.Resampling.NEAREST,
            'bilinear': Image.Resampling.BILINEAR,
            'bicubic': Image.Resampling.BICUBIC,
            'box': Image.Resampling.BOX,
            'lanczos': Image.Resampling.LANCZOS,
            'hamming': Image.Resampling.HAMMING
        }
    else:
        pillow_interp_codes = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'box': Image.BOX,
            'lanczos': Image.LANCZOS,
            'hamming': Image.HAMMING
        }

imread_backend = 'cv2'
cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}
def imresize(
    img: np.ndarray,
    size: Tuple[int, int],
    return_scale: bool = False,
    interpolation: str = "bilinear",
    out: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
) -> Union[Tuple[np.ndarray, float, float], np.ndarray]:
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
        `resized_img`.
    """
    h, w = img.shape[:2]
    if backend is None:
        backend = imread_backend
    if backend not in ["cv2", "pillow"]:
        raise ValueError(
            f"backend: {backend} is not supported for resize."
            f"Supported backends are 'cv2', 'pillow'"
        )

    if backend == "pillow":
        assert img.dtype == np.uint8, "Pillow backend only support uint8 type"
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, pillow_interp_codes[interpolation])
        resized_img = np.array(pil_image)
    else:
        resized_img = cv2.resize(
            img, size, dst=out, interpolation=cv2_interp_codes[interpolation]
        )
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


def imrescale(
    img: np.ndarray,
    scale: Union[float, int, Tuple[int, int]],
    return_scale: bool = False,
    interpolation: str = "bilinear",
    backend: Optional[str] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """Resize image while keeping the aspect ratio.

    Args:
        img (ndarray): The input image.
        scale (float | int | tuple[int]): The scaling factor or maximum size.
            If it is a float number or an integer, then the image will be
            rescaled by this factor, else if it is a tuple of 2 integers, then
            the image will be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.
        backend (str | None): Same as :func:`resize`.

    Returns:
        ndarray: The rescaled image.
    """
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(img, new_size, interpolation=interpolation, backend=backend)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img

def imflip(img: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
    """Flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".

    Returns:
        ndarray: The flipped image.
    """
    assert direction in ['horizontal', 'vertical', 'diagonal']
    if direction == 'horizontal':
        return np.flip(img, axis=1)
    elif direction == 'vertical':
        return np.flip(img, axis=0)
    else:
        return np.flip(img, axis=(0, 1))


def impad(img: np.ndarray,
          *,
          shape: Optional[Tuple[int, int]] = None,
          padding: Union[int, tuple, None] = None,
          pad_val: Union[float, List] = 0,
          padding_mode: str = 'constant') -> np.ndarray:
    """Pad the given image to a certain shape or pad on all sides with
    specified padding mode and padding value.

    Args:
        img (ndarray): Image to be padded.
        shape (tuple[int]): Expected padding shape (h, w). Default: None.
        padding (int or tuple[int]): Padding on each border. If a single int is
            provided this is used to pad all borders. If tuple of length 2 is
            provided this is the padding on left/right and top/bottom
            respectively. If a tuple of length 4 is provided this is the
            padding for the left, top, right and bottom borders respectively.
            Default: None. Note that `shape` and `padding` can not be both
            set.
        pad_val (Number | Sequence[Number]): Values to be filled in padding
            areas when padding_mode is 'constant'. Default: 0.
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default: constant.

            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        ndarray: The padded image.
    """

    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        width = max(shape[1] - img.shape[1], 0)
        height = max(shape[0] - img.shape[0], 0)
        padding = (0, 0, width, height)

    # check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError('pad_val must be a int or a tuple. '
                        f'But received {type(pad_val)}')

    # check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                         f'But received {padding}')

    # check padding mode
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }
    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        border_type[padding_mode],
        value=pad_val)

    return img


def _get_translate_matrix(offset: Union[int, float],
                          direction: str = 'horizontal') -> np.ndarray:
    """Generate the translate matrix.

    Args:
        offset (int | float): The offset used for translate.
        direction (str): The translate direction, either
            "horizontal" or "vertical".

    Returns:
        ndarray: The translate matrix with dtype float32.
    """
    if direction == 'horizontal':
        translate_matrix = np.float32([[1, 0, offset], [0, 1, 0]])
    elif direction == 'vertical':
        translate_matrix = np.float32([[1, 0, 0], [0, 1, offset]])
    return translate_matrix


def imtranslate(
    img: np.ndarray,
    offset: Union[int, float],
    direction: str = "horizontal",
    border_value: Union[int, tuple] = 0,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Translate an image.

    Args:
        img (ndarray): Image to be translated with format
            (h, w) or (h, w, c).
        offset (int | float): The offset used for translate.
        direction (str): The translate direction, either "horizontal"
            or "vertical".
        border_value (int | tuple[int]): Value used in case of a
            constant border.
        interpolation (str): Same as :func:`resize`.

    Returns:
        ndarray: The translated image.
    """
    assert direction in ["horizontal", "vertical"], f"Invalid direction: {direction}"
    height, width = img.shape[:2]
    if img.ndim == 2:
        channels = 1
    elif img.ndim == 3:
        channels = img.shape[-1]
    if isinstance(border_value, int):
        border_value = tuple([border_value] * channels)
    elif isinstance(border_value, tuple):
        assert len(border_value) == channels, (
            "Expected the num of elements in tuple equals the channels"
            "of input image. Found {} vs {}".format(len(border_value), channels)
        )
    else:
        raise ValueError(f"Invalid type {type(border_value)} for `border_value`.")
    translate_matrix = _get_translate_matrix(offset, direction)
    translated = cv2.warpAffine(
        img,
        translate_matrix,
        (width, height),
        # Note case when the number elements in `border_value`
        # greater than 3 (e.g. translating masks whose channels
        # large than 3) will raise TypeError in `cv2.warpAffine`.
        # Here simply slice the first 3 values in `border_value`.
        borderValue=border_value[:3],
        flags=cv2_interp_codes[interpolation],
    )
    return translated

def _get_shear_matrix(magnitude: Union[int, float],
                      direction: str = 'horizontal') -> np.ndarray:
    """Generate the shear matrix for transformation.

    Args:
        magnitude (int | float): The magnitude used for shear.
        direction (str): The flip direction, either "horizontal"
            or "vertical".

    Returns:
        ndarray: The shear matrix with dtype float32.
    """
    if direction == 'horizontal':
        shear_matrix = np.float32([[1, magnitude, 0], [0, 1, 0]])
    elif direction == 'vertical':
        shear_matrix = np.float32([[1, 0, 0], [magnitude, 1, 0]])
    return shear_matrix


def imshear(img: np.ndarray,
            magnitude: Union[int, float],
            direction: str = 'horizontal',
            border_value: Union[int, Tuple[int, int]] = 0,
            interpolation: str = 'bilinear') -> np.ndarray:
    """Shear an image.

    Args:
        img (ndarray): Image to be sheared with format (h, w)
            or (h, w, c).
        magnitude (int | float): The magnitude used for shear.
        direction (str): The flip direction, either "horizontal"
            or "vertical".
        border_value (int | tuple[int]): Value used in case of a
            constant border.
        interpolation (str): Same as :func:`resize`.

    Returns:
        ndarray: The sheared image.
    """
    assert direction in ['horizontal',
                         'vertical'], f'Invalid direction: {direction}'
    height, width = img.shape[:2]
    if img.ndim == 2:
        channels = 1
    elif img.ndim == 3:
        channels = img.shape[-1]
    if isinstance(border_value, int):
        border_value = tuple([border_value] * channels)  # type: ignore
    elif isinstance(border_value, tuple):
        assert len(border_value) == channels, \
            'Expected the num of elements in tuple equals the channels' \
            'of input image. Found {} vs {}'.format(
                len(border_value), channels)
    else:
        raise ValueError(
            f'Invalid type {type(border_value)} for `border_value`')
    shear_matrix = _get_shear_matrix(magnitude, direction)
    sheared = cv2.warpAffine(
        img,
        shear_matrix,
        (width, height),
        # Note case when the number elements in `border_value`
        # greater than 3 (e.g. shearing masks whose channels large
        # than 3) will raise TypeError in `cv2.warpAffine`.
        # Here simply slice the first 3 values in `border_value`.
        borderValue=border_value[:3],  # type: ignore
        flags=cv2_interp_codes[interpolation])
    return sheared

cv2_border_modes = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,
    "wrap": cv2.BORDER_WRAP,
    "reflect_101": cv2.BORDER_REFLECT_101,
    "transparent": cv2.BORDER_TRANSPARENT,
    "isolated": cv2.BORDER_ISOLATED,
}


def imrotate(
    img: np.ndarray,
    angle: float,
    center: Optional[Tuple[float, float]] = None,
    scale: float = 1.0,
    border_value: int = 0,
    interpolation: str = "bilinear",
    auto_bound: bool = False,
    border_mode: str = "constant",
) -> np.ndarray:
    """Rotate an image.

    Args:
        img (np.ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used.
        scale (float): Isotropic scale factor.
        border_value (int): Border value used in case of a constant border.
            Defaults to 0.
        interpolation (str): Same as :func:`resize`.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image.
        border_mode (str): Pixel extrapolation method. Defaults to 'constant'.

    Returns:
        np.ndarray: The rotated image.
    """
    if center is not None and auto_bound:
        raise ValueError("`auto_bound` conflicts with `center`")
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2_interp_codes[interpolation],
        borderMode=cv2_border_modes[border_mode],
        borderValue=border_value,
    )
    return rotated

def ensure_rng(rng=None):
    """Coerces input into a random number generator.

    If the input is None, then a global random state is returned.

    If the input is a numeric value, then that is used as a seed to construct a
    random state. Otherwise the input is returned as-is.

    Adapted from [1]_.

    Args:
        rng (int | numpy.random.RandomState | None):
            if None, then defaults to the global rng. Otherwise this can be an
            integer or a RandomState class
    Returns:
        (numpy.random.RandomState) : rng -
            a numpy random number generator

    References:
        .. [1] https://gitlab.kitware.com/computer-vision/kwarray/blob/master/kwarray/util_random.py#L270  # noqa: E501
    """

    if rng is None:
        rng = np.random.mtrand._rand
    elif isinstance(rng, int):
        rng = np.random.RandomState(rng)
    else:
        rng = rng
    return rng

class BitmapMasks(BaseInstanceMasks):
    """This class represents masks in the form of bitmaps.

    Args:
        masks (ndarray): ndarray of masks in shape (N, H, W), where N is
            the number of objects.
        height (int): height of masks
        width (int): width of masks

    Example:
        >>> from mmdet.data_elements.mask.structures import *  # NOQA
        >>> num_masks, H, W = 3, 32, 32
        >>> rng = np.random.RandomState(0)
        >>> masks = (rng.rand(num_masks, H, W) > 0.1).astype(np.int64)
        >>> self = BitmapMasks(masks, height=H, width=W)

        >>> # demo crop_and_resize
        >>> num_boxes = 5
        >>> bboxes = np.array([[0, 0, 30, 10.0]] * num_boxes)
        >>> out_shape = (14, 14)
        >>> inds = torch.randint(0, len(self), size=(num_boxes,))
        >>> device = 'cpu'
        >>> interpolation = 'bilinear'
        >>> new = self.crop_and_resize(
        ...     bboxes, out_shape, inds, device, interpolation)
        >>> assert len(new) == num_boxes
        >>> assert new.height, new.width == out_shape
    """

    def __init__(self, masks, height, width):
        self.height = height
        self.width = width
        if len(masks) == 0:
            self.masks = np.empty((0, self.height, self.width), dtype=np.uint8)
        else:
            assert isinstance(masks, (list, np.ndarray))
            if isinstance(masks, list):
                assert isinstance(masks[0], np.ndarray)
                assert masks[0].ndim == 2  # (H, W)
            else:
                assert masks.ndim == 3  # (N, H, W)

            self.masks = np.stack(masks).reshape(-1, height, width)
            assert self.masks.shape[1] == self.height
            assert self.masks.shape[2] == self.width

    def __getitem__(self, index):
        """Index the BitmapMask.

        Args:
            index (int | ndarray): Indices in the format of integer or ndarray.

        Returns:
            :obj:`BitmapMasks`: Indexed bitmap masks.
        """
        masks = self.masks[index].reshape(-1, self.height, self.width)
        return BitmapMasks(masks, self.height, self.width)

    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += f"num_masks={len(self.masks)}, "
        s += f"height={self.height}, "
        s += f"width={self.width})"
        return s

    def __len__(self):
        """Number of masks."""
        return len(self.masks)

    def rescale(self, scale, interpolation="nearest"):
        """See :func:`BaseInstanceMasks.rescale`."""
        if len(self.masks) == 0:
            new_w, new_h = rescale_size((self.width, self.height), scale)
            rescaled_masks = np.empty((0, new_h, new_w), dtype=np.uint8)
        else:
            rescaled_masks = np.stack(
                [
                    imrescale(mask, scale, interpolation=interpolation)
                    for mask in self.masks
                ]
            )
        height, width = rescaled_masks.shape[1:]
        return BitmapMasks(rescaled_masks, height, width)

    def resize(self, out_shape, interpolation="nearest"):
        """See :func:`BaseInstanceMasks.resize`."""
        if len(self.masks) == 0:
            resized_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            resized_masks = np.stack(
                [
                    imresize(mask, out_shape[::-1], interpolation=interpolation)
                    for mask in self.masks
                ]
            )
        return BitmapMasks(resized_masks, *out_shape)

    def flip(self, flip_direction="horizontal"):
        """See :func:`BaseInstanceMasks.flip`."""
        assert flip_direction in ("horizontal", "vertical", "diagonal")

        if len(self.masks) == 0:
            flipped_masks = self.masks
        else:
            flipped_masks = np.stack(
                [imflip(mask, direction=flip_direction) for mask in self.masks]
            )
        return BitmapMasks(flipped_masks, self.height, self.width)

    def pad(self, out_shape, pad_val=0):
        """See :func:`BaseInstanceMasks.pad`."""
        if len(self.masks) == 0:
            padded_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            padded_masks = np.stack(
                [
                    impad(mask, shape=out_shape, pad_val=pad_val)
                    for mask in self.masks
                ]
            )
        return BitmapMasks(padded_masks, *out_shape)

    def crop(self, bbox):
        """See :func:`BaseInstanceMasks.crop`."""
        assert isinstance(bbox, np.ndarray)
        assert bbox.ndim == 1

        # clip the boundary
        bbox = bbox.copy()
        bbox[0::2] = np.clip(bbox[0::2], 0, self.width)
        bbox[1::2] = np.clip(bbox[1::2], 0, self.height)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1, 1)
        h = np.maximum(y2 - y1, 1)

        if len(self.masks) == 0:
            cropped_masks = np.empty((0, h, w), dtype=np.uint8)
        else:
            cropped_masks = self.masks[:, y1 : y1 + h, x1 : x1 + w]
        return BitmapMasks(cropped_masks, h, w)

    def crop_and_resize(
        self,
        bboxes,
        out_shape,
        inds,
        device="cpu",
        interpolation="bilinear",
        binarize=True,
    ):
        """See :func:`BaseInstanceMasks.crop_and_resize`."""
        if len(self.masks) == 0:
            empty_masks = np.empty((0, *out_shape), dtype=np.uint8)
            return BitmapMasks(empty_masks, *out_shape)

        # convert bboxes to tensor
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes).to(device=device)
        if isinstance(inds, np.ndarray):
            inds = torch.from_numpy(inds).to(device=device)

        num_bbox = bboxes.shape[0]
        fake_inds = torch.arange(num_bbox, device=device).to(dtype=bboxes.dtype)[
            :, None
        ]
        rois = torch.cat([fake_inds, bboxes], dim=1)  # Nx5
        rois = rois.to(device=device)
        if num_bbox > 0:
            gt_masks_th = (
                torch.from_numpy(self.masks)
                .to(device)
                .index_select(0, inds)
                .to(dtype=rois.dtype)
            )
            targets = roi_align(
                gt_masks_th[:, None, :, :], rois, out_shape, 1.0, 0, "avg", True
            ).squeeze(1)
            if binarize:
                resized_masks = (targets >= 0.5).cpu().numpy()
            else:
                resized_masks = targets.cpu().numpy()
        else:
            resized_masks = []
        return BitmapMasks(resized_masks, *out_shape)

    def expand(self, expanded_h, expanded_w, top, left):
        """See :func:`BaseInstanceMasks.expand`."""
        if len(self.masks) == 0:
            expanded_mask = np.empty((0, expanded_h, expanded_w), dtype=np.uint8)
        else:
            expanded_mask = np.zeros(
                (len(self), expanded_h, expanded_w), dtype=np.uint8
            )
            expanded_mask[:, top : top + self.height, left : left + self.width] = (
                self.masks
            )
        return BitmapMasks(expanded_mask, expanded_h, expanded_w)

    def translate(
        self,
        out_shape,
        offset,
        direction="horizontal",
        border_value=0,
        interpolation="bilinear",
    ):
        """Translate the BitmapMasks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            offset (int | float): The offset for translate.
            direction (str): The translate direction, either "horizontal"
                or "vertical".
            border_value (int | float): Border value. Default 0 for masks.
            interpolation (str): Same as :func:`mmcv.imtranslate`.

        Returns:
            BitmapMasks: Translated BitmapMasks.

        Example:
            >>> from mmdet.data_elements.mask.structures import BitmapMasks
            >>> self = BitmapMasks.random(dtype=np.uint8)
            >>> out_shape = (32, 32)
            >>> offset = 4
            >>> direction = 'horizontal'
            >>> border_value = 0
            >>> interpolation = 'bilinear'
            >>> # Note, There seem to be issues when:
            >>> # * the mask dtype is not supported by cv2.AffineWarp
            >>> new = self.translate(out_shape, offset, direction,
            >>>                      border_value, interpolation)
            >>> assert len(new) == len(self)
            >>> assert new.height, new.width == out_shape
        """
        if len(self.masks) == 0:
            translated_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            masks = self.masks
            if masks.shape[-2:] != out_shape:
                empty_masks = np.zeros((masks.shape[0], *out_shape), dtype=masks.dtype)
                min_h = min(out_shape[0], masks.shape[1])
                min_w = min(out_shape[1], masks.shape[2])
                empty_masks[:, :min_h, :min_w] = masks[:, :min_h, :min_w]
                masks = empty_masks
            translated_masks = imtranslate(
                masks.transpose((1, 2, 0)),
                offset,
                direction,
                border_value=border_value,
                interpolation=interpolation,
            )
            if translated_masks.ndim == 2:
                translated_masks = translated_masks[:, :, None]
            translated_masks = translated_masks.transpose((2, 0, 1)).astype(
                self.masks.dtype
            )
        return BitmapMasks(translated_masks, *out_shape)

    def shear(
        self,
        out_shape,
        magnitude,
        direction="horizontal",
        border_value=0,
        interpolation="bilinear",
    ):
        """Shear the BitmapMasks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            magnitude (int | float): The magnitude used for shear.
            direction (str): The shear direction, either "horizontal"
                or "vertical".
            border_value (int | tuple[int]): Value used in case of a
                constant border.
            interpolation (str): Same as in :func:`mmcv.imshear`.

        Returns:
            BitmapMasks: The sheared masks.
        """
        if len(self.masks) == 0:
            sheared_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            sheared_masks = imshear(
                self.masks.transpose((1, 2, 0)),
                magnitude,
                direction,
                border_value=border_value,
                interpolation=interpolation,
            )
            if sheared_masks.ndim == 2:
                sheared_masks = sheared_masks[:, :, None]
            sheared_masks = sheared_masks.transpose((2, 0, 1)).astype(self.masks.dtype)
        return BitmapMasks(sheared_masks, *out_shape)

    def rotate(
        self,
        out_shape,
        angle,
        center=None,
        scale=1.0,
        border_value=0,
        interpolation="bilinear",
    ):
        """Rotate the BitmapMasks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            angle (int | float): Rotation angle in degrees. Positive values
                mean counter-clockwise rotation.
            center (tuple[float], optional): Center point (w, h) of the
                rotation in source image. If not specified, the center of
                the image will be used.
            scale (int | float): Isotropic scale factor.
            border_value (int | float): Border value. Default 0 for masks.
            interpolation (str): Same as in :func:`mmcv.imrotate`.

        Returns:
            BitmapMasks: Rotated BitmapMasks.
        """
        if len(self.masks) == 0:
            rotated_masks = np.empty((0, *out_shape), dtype=self.masks.dtype)
        else:
            rotated_masks = imrotate(
                self.masks.transpose((1, 2, 0)),
                angle,
                center=center,
                scale=scale,
                border_value=border_value,
                interpolation=interpolation,
            )
            if rotated_masks.ndim == 2:
                # case when only one mask, (h, w)
                rotated_masks = rotated_masks[:, :, None]  # (h, w, 1)
            rotated_masks = rotated_masks.transpose((2, 0, 1)).astype(self.masks.dtype)
        return BitmapMasks(rotated_masks, *out_shape)

    @property
    def areas(self):
        """See :py:attr:`BaseInstanceMasks.areas`."""
        return self.masks.sum((1, 2))

    def to_ndarray(self):
        """See :func:`BaseInstanceMasks.to_ndarray`."""
        return self.masks

    def to_tensor(self, dtype, device):
        """See :func:`BaseInstanceMasks.to_tensor`."""
        return torch.tensor(self.masks, dtype=dtype, device=device)

    @classmethod
    def random(cls, num_masks=3, height=32, width=32, dtype=np.uint8, rng=None):
        """Generate random bitmap masks for demo / testing purposes.

        Example:
            >>> from mmdet.data_elements.mask.structures import BitmapMasks
            >>> self = BitmapMasks.random()
            >>> print('self = {}'.format(self))
            self = BitmapMasks(num_masks=3, height=32, width=32)
        """

        rng = ensure_rng(rng)
        masks = (rng.rand(num_masks, height, width) > 0.1).astype(dtype)
        self = cls(masks, height=height, width=width)
        return self

    @classmethod
    def cat(cls: Type[T], masks: Sequence[T]) -> T:
        """Concatenate a sequence of masks into one single mask instance.

        Args:
            masks (Sequence[BitmapMasks]): A sequence of mask instances.

        Returns:
            BitmapMasks: Concatenated mask instance.
        """
        assert isinstance(masks, Sequence)
        if len(masks) == 0:
            raise ValueError("masks should not be an empty list.")
        assert all(isinstance(m, cls) for m in masks)

        mask_array = np.concatenate([m.masks for m in masks], axis=0)
        return cls(mask_array, *mask_array.shape[1:])


class PolygonMasks(BaseInstanceMasks):
    """This class represents masks in the form of polygons.

    Polygons is a list of three levels. The first level of the list
    corresponds to objects, the second level to the polys that compose the
    object, the third level to the poly coordinates

    Args:
        masks (list[list[ndarray]]): The first level of the list
            corresponds to objects, the second level to the polys that
            compose the object, the third level to the poly coordinates
        height (int): height of masks
        width (int): width of masks

    Example:
        >>> from mmdet.data_elements.mask.structures import *  # NOQA
        >>> masks = [
        >>>     [ np.array([0, 0, 10, 0, 10, 10., 0, 10, 0, 0]) ]
        >>> ]
        >>> height, width = 16, 16
        >>> self = PolygonMasks(masks, height, width)

        >>> # demo translate
        >>> new = self.translate((16, 16), 4., direction='horizontal')
        >>> assert np.all(new.masks[0][0][1::2] == masks[0][0][1::2])
        >>> assert np.all(new.masks[0][0][0::2] == masks[0][0][0::2] + 4)

        >>> # demo crop_and_resize
        >>> num_boxes = 3
        >>> bboxes = np.array([[0, 0, 30, 10.0]] * num_boxes)
        >>> out_shape = (16, 16)
        >>> inds = torch.randint(0, len(self), size=(num_boxes,))
        >>> device = 'cpu'
        >>> interpolation = 'bilinear'
        >>> new = self.crop_and_resize(
        ...     bboxes, out_shape, inds, device, interpolation)
        >>> assert len(new) == num_boxes
        >>> assert new.height, new.width == out_shape
    """

    def __init__(self, masks, height, width):
        assert isinstance(masks, list)
        if len(masks) > 0:
            assert isinstance(masks[0], list)
            assert isinstance(masks[0][0], np.ndarray)

        self.height = height
        self.width = width
        self.masks = masks

    def __getitem__(self, index):
        """Index the polygon masks.

        Args:
            index (ndarray | List): The indices.

        Returns:
            :obj:`PolygonMasks`: The indexed polygon masks.
        """
        if isinstance(index, np.ndarray):
            if index.dtype == bool:
                index = np.where(index)[0].tolist()
            else:
                index = index.tolist()
        if isinstance(index, list):
            masks = [self.masks[i] for i in index]
        else:
            try:
                masks = self.masks[index]
            except Exception:
                raise ValueError(
                    f"Unsupported input of type {type(index)} for indexing!"
                )
        if len(masks) and isinstance(masks[0], np.ndarray):
            masks = [masks]  # ensure a list of three levels
        return PolygonMasks(masks, self.height, self.width)

    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += f"num_masks={len(self.masks)}, "
        s += f"height={self.height}, "
        s += f"width={self.width})"
        return s

    def __len__(self):
        """Number of masks."""
        return len(self.masks)

    def rescale(self, scale, interpolation=None):
        """see :func:`BaseInstanceMasks.rescale`"""
        new_w, new_h = rescale_size((self.width, self.height), scale)
        if len(self.masks) == 0:
            rescaled_masks = PolygonMasks([], new_h, new_w)
        else:
            rescaled_masks = self.resize((new_h, new_w))
        return rescaled_masks

    def resize(self, out_shape, interpolation=None):
        """see :func:`BaseInstanceMasks.resize`"""
        if len(self.masks) == 0:
            resized_masks = PolygonMasks([], *out_shape)
        else:
            h_scale = out_shape[0] / self.height
            w_scale = out_shape[1] / self.width
            resized_masks = []
            for poly_per_obj in self.masks:
                resized_poly = []
                for p in poly_per_obj:
                    p = p.copy()
                    p[0::2] = p[0::2] * w_scale
                    p[1::2] = p[1::2] * h_scale
                    resized_poly.append(p)
                resized_masks.append(resized_poly)
            resized_masks = PolygonMasks(resized_masks, *out_shape)
        return resized_masks

    def flip(self, flip_direction="horizontal"):
        """see :func:`BaseInstanceMasks.flip`"""
        assert flip_direction in ("horizontal", "vertical", "diagonal")
        if len(self.masks) == 0:
            flipped_masks = PolygonMasks([], self.height, self.width)
        else:
            flipped_masks = []
            for poly_per_obj in self.masks:
                flipped_poly_per_obj = []
                for p in poly_per_obj:
                    p = p.copy()
                    if flip_direction == "horizontal":
                        p[0::2] = self.width - p[0::2]
                    elif flip_direction == "vertical":
                        p[1::2] = self.height - p[1::2]
                    else:
                        p[0::2] = self.width - p[0::2]
                        p[1::2] = self.height - p[1::2]
                    flipped_poly_per_obj.append(p)
                flipped_masks.append(flipped_poly_per_obj)
            flipped_masks = PolygonMasks(flipped_masks, self.height, self.width)
        return flipped_masks

    def crop(self, bbox):
        """see :func:`BaseInstanceMasks.crop`"""
        assert isinstance(bbox, np.ndarray)
        assert bbox.ndim == 1

        # clip the boundary
        bbox = bbox.copy()
        bbox[0::2] = np.clip(bbox[0::2], 0, self.width)
        bbox[1::2] = np.clip(bbox[1::2], 0, self.height)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1, 1)
        h = np.maximum(y2 - y1, 1)

        if len(self.masks) == 0:
            cropped_masks = PolygonMasks([], h, w)
        else:
            # reference: https://github.com/facebookresearch/fvcore/blob/main/fvcore/transforms/transform.py  # noqa
            crop_box = geometry.box(x1, y1, x2, y2).buffer(0.0)
            cropped_masks = []
            # suppress shapely warnings util it incorporates GEOS>=3.11.2
            # reference: https://github.com/shapely/shapely/issues/1345
            initial_settings = np.seterr()
            np.seterr(invalid="ignore")
            for poly_per_obj in self.masks:
                cropped_poly_per_obj = []
                for p in poly_per_obj:
                    p = p.copy()
                    p = geometry.Polygon(p.reshape(-1, 2)).buffer(0.0)
                    # polygon must be valid to perform intersection.
                    if not p.is_valid:
                        continue
                    cropped = p.intersection(crop_box)
                    if cropped.is_empty:
                        continue
                    if isinstance(cropped, geometry.collection.BaseMultipartGeometry):
                        cropped = cropped.geoms
                    else:
                        cropped = [cropped]
                    # one polygon may be cropped to multiple ones
                    for poly in cropped:
                        # ignore lines or points
                        if not isinstance(poly, geometry.Polygon) or not poly.is_valid:
                            continue
                        coords = np.asarray(poly.exterior.coords)
                        # remove an extra identical vertex at the end
                        coords = coords[:-1]
                        coords[:, 0] -= x1
                        coords[:, 1] -= y1
                        cropped_poly_per_obj.append(coords.reshape(-1))
                # a dummy polygon to avoid misalignment between masks and boxes
                if len(cropped_poly_per_obj) == 0:
                    cropped_poly_per_obj = [np.array([0, 0, 0, 0, 0, 0])]
                cropped_masks.append(cropped_poly_per_obj)
            np.seterr(**initial_settings)
            cropped_masks = PolygonMasks(cropped_masks, h, w)
        return cropped_masks

    def pad(self, out_shape, pad_val=0):
        """padding has no effect on polygons`"""
        return PolygonMasks(self.masks, *out_shape)

    def expand(self, *args, **kwargs):
        """TODO: Add expand for polygon"""
        raise NotImplementedError

    def crop_and_resize(
        self,
        bboxes,
        out_shape,
        inds,
        device="cpu",
        interpolation="bilinear",
        binarize=True,
    ):
        """see :func:`BaseInstanceMasks.crop_and_resize`"""
        out_h, out_w = out_shape
        if len(self.masks) == 0:
            return PolygonMasks([], out_h, out_w)

        if not binarize:
            raise ValueError(
                "Polygons are always binary, " "setting binarize=False is unsupported"
            )

        resized_masks = []
        for i in range(len(bboxes)):
            mask = self.masks[inds[i]]
            bbox = bboxes[i, :]
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1, 1)
            h = np.maximum(y2 - y1, 1)
            h_scale = out_h / max(h, 0.1)  # avoid too large scale
            w_scale = out_w / max(w, 0.1)

            resized_mask = []
            for p in mask:
                p = p.copy()
                # crop
                # pycocotools will clip the boundary
                p[0::2] = p[0::2] - bbox[0]
                p[1::2] = p[1::2] - bbox[1]

                # resize
                p[0::2] = p[0::2] * w_scale
                p[1::2] = p[1::2] * h_scale
                resized_mask.append(p)
            resized_masks.append(resized_mask)
        return PolygonMasks(resized_masks, *out_shape)

    def translate(
        self,
        out_shape,
        offset,
        direction="horizontal",
        border_value=None,
        interpolation=None,
    ):
        """Translate the PolygonMasks.

        Example:
            >>> self = PolygonMasks.random(dtype=np.int64)
            >>> out_shape = (self.height, self.width)
            >>> new = self.translate(out_shape, 4., direction='horizontal')
            >>> assert np.all(new.masks[0][0][1::2] == self.masks[0][0][1::2])
            >>> assert np.all(new.masks[0][0][0::2] == self.masks[0][0][0::2] + 4)  # noqa: E501
        """
        assert border_value is None or border_value == 0, (
            "Here border_value is not "
            f"used, and defaultly should be None or 0. got {border_value}."
        )
        if len(self.masks) == 0:
            translated_masks = PolygonMasks([], *out_shape)
        else:
            translated_masks = []
            for poly_per_obj in self.masks:
                translated_poly_per_obj = []
                for p in poly_per_obj:
                    p = p.copy()
                    if direction == "horizontal":
                        p[0::2] = np.clip(p[0::2] + offset, 0, out_shape[1])
                    elif direction == "vertical":
                        p[1::2] = np.clip(p[1::2] + offset, 0, out_shape[0])
                    translated_poly_per_obj.append(p)
                translated_masks.append(translated_poly_per_obj)
            translated_masks = PolygonMasks(translated_masks, *out_shape)
        return translated_masks

    def shear(
        self,
        out_shape,
        magnitude,
        direction="horizontal",
        border_value=0,
        interpolation="bilinear",
    ):
        """See :func:`BaseInstanceMasks.shear`."""
        if len(self.masks) == 0:
            sheared_masks = PolygonMasks([], *out_shape)
        else:
            sheared_masks = []
            if direction == "horizontal":
                shear_matrix = np.stack([[1, magnitude], [0, 1]]).astype(np.float32)
            elif direction == "vertical":
                shear_matrix = np.stack([[1, 0], [magnitude, 1]]).astype(np.float32)
            for poly_per_obj in self.masks:
                sheared_poly = []
                for p in poly_per_obj:
                    p = np.stack([p[0::2], p[1::2]], axis=0)  # [2, n]
                    new_coords = np.matmul(shear_matrix, p)  # [2, n]
                    new_coords[0, :] = np.clip(new_coords[0, :], 0, out_shape[1])
                    new_coords[1, :] = np.clip(new_coords[1, :], 0, out_shape[0])
                    sheared_poly.append(new_coords.transpose((1, 0)).reshape(-1))
                sheared_masks.append(sheared_poly)
            sheared_masks = PolygonMasks(sheared_masks, *out_shape)
        return sheared_masks

    def rotate(
        self,
        out_shape,
        angle,
        center=None,
        scale=1.0,
        border_value=0,
        interpolation="bilinear",
    ):
        """See :func:`BaseInstanceMasks.rotate`."""
        if len(self.masks) == 0:
            rotated_masks = PolygonMasks([], *out_shape)
        else:
            rotated_masks = []
            rotate_matrix = cv2.getRotationMatrix2D(center, -angle, scale)
            for poly_per_obj in self.masks:
                rotated_poly = []
                for p in poly_per_obj:
                    p = p.copy()
                    coords = np.stack([p[0::2], p[1::2]], axis=1)  # [n, 2]
                    # pad 1 to convert from format [x, y] to homogeneous
                    # coordinates format [x, y, 1]
                    coords = np.concatenate(
                        (coords, np.ones((coords.shape[0], 1), coords.dtype)), axis=1
                    )  # [n, 3]
                    rotated_coords = np.matmul(
                        rotate_matrix[None, :, :], coords[:, :, None]
                    )[
                        ..., 0
                    ]  # [n, 2, 1] -> [n, 2]
                    rotated_coords[:, 0] = np.clip(
                        rotated_coords[:, 0], 0, out_shape[1]
                    )
                    rotated_coords[:, 1] = np.clip(
                        rotated_coords[:, 1], 0, out_shape[0]
                    )
                    rotated_poly.append(rotated_coords.reshape(-1))
                rotated_masks.append(rotated_poly)
            rotated_masks = PolygonMasks(rotated_masks, *out_shape)
        return rotated_masks

    def to_bitmap(self):
        """convert polygon masks to bitmap masks."""
        bitmap_masks = self.to_ndarray()
        return BitmapMasks(bitmap_masks, self.height, self.width)

    @property
    def areas(self):
        """Compute areas of masks.

        This func is modified from `detectron2
        <https://github.com/facebookresearch/detectron2/blob/ffff8acc35ea88ad1cb1806ab0f00b4c1c5dbfd9/detectron2/structures/masks.py#L387>`_.
        The function only works with Polygons using the shoelace formula.

        Return:
            ndarray: areas of each instance
        """  # noqa: W501
        area = []
        for polygons_per_obj in self.masks:
            area_per_obj = 0
            for p in polygons_per_obj:
                area_per_obj += self._polygon_area(p[0::2], p[1::2])
            area.append(area_per_obj)
        return np.asarray(area)

    def _polygon_area(self, x, y):
        """Compute the area of a component of a polygon.

        Using the shoelace formula:
        https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

        Args:
            x (ndarray): x coordinates of the component
            y (ndarray): y coordinates of the component

        Return:
            float: the are of the component
        """  # noqa: 501
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def to_ndarray(self):
        """Convert masks to the format of ndarray."""
        if len(self.masks) == 0:
            return np.empty((0, self.height, self.width), dtype=np.uint8)
        bitmap_masks = []
        for poly_per_obj in self.masks:
            bitmap_masks.append(
                polygon_to_bitmap(poly_per_obj, self.height, self.width)
            )
        return np.stack(bitmap_masks)

    def to_tensor(self, dtype, device):
        """See :func:`BaseInstanceMasks.to_tensor`."""
        if len(self.masks) == 0:
            return torch.empty((0, self.height, self.width), dtype=dtype, device=device)
        ndarray_masks = self.to_ndarray()
        return torch.tensor(ndarray_masks, dtype=dtype, device=device)

    @classmethod
    def random(
        cls, num_masks=3, height=32, width=32, n_verts=5, dtype=np.float32, rng=None
    ):
        """Generate random polygon masks for demo / testing purposes.

        Adapted from [1]_

        References:
            .. [1] https://gitlab.kitware.com/computer-vision/kwimage/-/blob/928cae35ca8/kwimage/structs/polygon.py#L379  # noqa: E501

        Example:
            >>> from mmdet.data_elements.mask.structures import PolygonMasks
            >>> self = PolygonMasks.random()
            >>> print('self = {}'.format(self))
        """

        rng = ensure_rng(rng)

        def _gen_polygon(n, irregularity, spikeyness):
            """Creates the polygon by sampling points on a circle around the
            centre.  Random noise is added by varying the angular spacing
            between sequential points, and by varying the radial distance of
            each point from the centre.

            Based on original code by Mike Ounsworth

            Args:
                n (int): number of vertices
                irregularity (float): [0,1] indicating how much variance there
                    is in the angular spacing of vertices. [0,1] will map to
                    [0, 2pi/numberOfVerts]
                spikeyness (float): [0,1] indicating how much variance there is
                    in each vertex from the circle of radius aveRadius. [0,1]
                    will map to [0, aveRadius]

            Returns:
                a list of vertices, in CCW order.
            """

            # Generate around the unit circle
            cx, cy = (0.0, 0.0)
            radius = 1

            tau = np.pi * 2

            irregularity = np.clip(irregularity, 0, 1) * 2 * np.pi / n
            spikeyness = np.clip(spikeyness, 1e-9, 1)

            # generate n angle steps
            lower = (tau / n) - irregularity
            upper = (tau / n) + irregularity
            angle_steps = rng.uniform(lower, upper, n)

            # normalize the steps so that point 0 and point n+1 are the same
            k = angle_steps.sum() / (2 * np.pi)
            angles = (angle_steps / k).cumsum() + rng.uniform(0, tau)

            # Convert high and low values to be wrt the standard normal range
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
            low = 0
            high = 2 * radius
            mean = radius
            std = spikeyness
            a = (low - mean) / std
            b = (high - mean) / std
            tnorm = truncnorm(a=a, b=b, loc=mean, scale=std)

            # now generate the points
            radii = tnorm.rvs(n, random_state=rng)
            x_pts = cx + radii * np.cos(angles)
            y_pts = cy + radii * np.sin(angles)

            points = np.hstack([x_pts[:, None], y_pts[:, None]])

            # Scale to 0-1 space
            points = points - points.min(axis=0)
            points = points / points.max(axis=0)

            # Randomly place within 0-1 space
            points = points * (rng.rand() * 0.8 + 0.2)
            min_pt = points.min(axis=0)
            max_pt = points.max(axis=0)

            high = 1 - max_pt
            low = 0 - min_pt
            offset = (rng.rand(2) * (high - low)) + low
            points = points + offset
            return points

        def _order_vertices(verts):
            """
            References:
                https://stackoverflow.com/questions/1709283/how-can-i-sort-a-coordinate-list-for-a-rectangle-counterclockwise
            """
            mlat = verts.T[0].sum() / len(verts)
            mlng = verts.T[1].sum() / len(verts)

            tau = np.pi * 2
            angle = (np.arctan2(mlat - verts.T[0], verts.T[1] - mlng) + tau) % tau
            sortx = angle.argsort()
            verts = verts.take(sortx, axis=0)
            return verts

        # Generate a random exterior for each requested mask
        masks = []
        for _ in range(num_masks):
            exterior = _order_vertices(_gen_polygon(n_verts, 0.9, 0.9))
            exterior = (exterior * [(width, height)]).astype(dtype)
            masks.append([exterior.ravel()])

        self = cls(masks, height, width)
        return self

    @classmethod
    def cat(cls: Type[T], masks: Sequence[T]) -> T:
        """Concatenate a sequence of masks into one single mask instance.

        Args:
            masks (Sequence[PolygonMasks]): A sequence of mask instances.

        Returns:
            PolygonMasks: Concatenated mask instance.
        """
        assert isinstance(masks, Sequence)
        if len(masks) == 0:
            raise ValueError("masks should not be an empty list.")
        assert all(isinstance(m, cls) for m in masks)

        mask_list = list(itertools.chain(*[m.masks for m in masks]))
        return cls(mask_list, masks[0].height, masks[0].width)


def polygon_to_bitmap(polygons, height, width):
    """Convert masks from the form of polygons to bitmaps.

    Args:
        polygons (list[ndarray]): masks in polygon representation
        height (int): mask height
        width (int): mask width

    Return:
        ndarray: the converted masks in bitmap representation
    """
    rles = maskUtils.frPyObjects(polygons, height, width)
    rle = maskUtils.merge(rles)
    bitmap_mask = maskUtils.decode(rle).astype(bool)
    return bitmap_mask


DeviceType = Union[str, torch.device]
MaskType = Union[BitmapMasks, PolygonMasks]
box_converters: dict = {}

class BaseBoxes(metaclass=ABCMeta):
    """The base class for 2D box types.

    The functions of ``BaseBoxes`` lie in three fields:

    - Verify the boxes shape.
    - Support tensor-like operations.
    - Define abstract functions for 2D boxes.

    In ``__init__`` , ``BaseBoxes`` verifies the validity of the data shape
    w.r.t ``box_dim``. The tensor with the dimension >= 2 and the length
    of the last dimension being ``box_dim`` will be regarded as valid.
    ``BaseBoxes`` will restore them at the field ``tensor``. It's necessary
    to override ``box_dim`` in subclass to guarantee the data shape is
    correct.

    There are many basic tensor-like functions implemented in ``BaseBoxes``.
    In most cases, users can operate ``BaseBoxes`` instance like a normal
    tensor. To protect the validity of data shape, All tensor-like functions
    cannot modify the last dimension of ``self.tensor``.

    When creating a new box type, users need to inherit from ``BaseBoxes``
    and override abstract methods and specify the ``box_dim``. Then, register
    the new box type by using the decorator ``register_box_type``.

    Args:
        data (Tensor or np.ndarray or Sequence): The box data with shape
            (..., box_dim).
        dtype (torch.dtype, Optional): data type of boxes. Defaults to None.
        device (str or torch.device, Optional): device of boxes.
            Default to None.
        clone (bool): Whether clone ``boxes`` or not. Defaults to True.
    """

    # Used to verify the last dimension length
    # Should override it in subclass.
    box_dim: int = 0

    def __init__(
        self,
        data: Union[Tensor, np.ndarray, Sequence],
        dtype: Optional[torch.dtype] = None,
        device: Optional[DeviceType] = None,
        clone: bool = True,
    ) -> None:
        if isinstance(data, (np.ndarray, Tensor, Sequence)):
            data = torch.as_tensor(data)
        else:
            raise TypeError(
                "boxes should be Tensor, ndarray, or Sequence, ",
                f"but got {type(data)}",
            )

        if device is not None or dtype is not None:
            data = data.to(dtype=dtype, device=device)
        # Clone the data to avoid potential bugs
        if clone:
            data = data.clone()
        # handle the empty input like []
        if data.numel() == 0:
            data = data.reshape((-1, self.box_dim))

        assert data.dim() >= 2 and data.size(-1) == self.box_dim, (
            "The boxes dimension must >= 2 and the length of the last "
            f"dimension must be {self.box_dim}, but got boxes with "
            f"shape {data.shape}."
        )
        self.tensor = data

    def convert_to(self, dst_type: Union[str, type]) -> "BaseBoxes":
        """Convert self to another box type.

        Args:
            dst_type (str or type): destination box type.

        Returns:
            :obj:`BaseBoxes`: destination box type object .
        """

        return convert_box_type(self, dst_type=dst_type)

    def empty_boxes(
        self: T,
        dtype: Optional[torch.dtype] = None,
        device: Optional[DeviceType] = None,
    ) -> T:
        """Create empty box.

        Args:
            dtype (torch.dtype, Optional): data type of boxes.
            device (str or torch.device, Optional): device of boxes.

        Returns:
            T: empty boxes with shape of (0, box_dim).
        """
        empty_box = self.tensor.new_zeros(0, self.box_dim, dtype=dtype, device=device)
        return type(self)(empty_box, clone=False)

    def fake_boxes(
        self: T,
        sizes: Tuple[int],
        fill: float = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[DeviceType] = None,
    ) -> T:
        """Create fake boxes with specific sizes and fill values.

        Args:
            sizes (Tuple[int]): The size of fake boxes. The last value must
                be equal with ``self.box_dim``.
            fill (float): filling value. Defaults to 0.
            dtype (torch.dtype, Optional): data type of boxes.
            device (str or torch.device, Optional): device of boxes.

        Returns:
            T: Fake boxes with shape of ``sizes``.
        """
        fake_boxes = self.tensor.new_full(sizes, fill, dtype=dtype, device=device)
        return type(self)(fake_boxes, clone=False)

    def __getitem__(self: T, index: IndexType) -> T:
        """Rewrite getitem to protect the last dimension shape."""
        boxes = self.tensor
        if isinstance(index, np.ndarray):
            index = torch.as_tensor(index, device=self.device)
        if isinstance(index, Tensor) and index.dtype == torch.bool:
            assert index.dim() < boxes.dim()
        elif isinstance(index, tuple):
            assert len(index) < boxes.dim()
            # `Ellipsis`(...) is commonly used in index like [None, ...].
            # When `Ellipsis` is in index, it must be the last item.
            if Ellipsis in index:
                assert index[-1] is Ellipsis

        boxes = boxes[index]
        if boxes.dim() == 1:
            boxes = boxes.reshape(1, -1)
        return type(self)(boxes, clone=False)

    def __setitem__(self: T, index: IndexType, values: Union[Tensor, T]) -> T:
        """Rewrite setitem to protect the last dimension shape."""
        assert type(values) is type(
            self
        ), "The value to be set must be the same box type as self"
        values = values.tensor

        if isinstance(index, np.ndarray):
            index = torch.as_tensor(index, device=self.device)
        if isinstance(index, Tensor) and index.dtype == torch.bool:
            assert index.dim() < self.tensor.dim()
        elif isinstance(index, tuple):
            assert len(index) < self.tensor.dim()
            # `Ellipsis`(...) is commonly used in index like [None, ...].
            # When `Ellipsis` is in index, it must be the last item.
            if Ellipsis in index:
                assert index[-1] is Ellipsis

        self.tensor[index] = values

    def __len__(self) -> int:
        """Return the length of self.tensor first dimension."""
        return self.tensor.size(0)

    def __deepcopy__(self, memo):
        """Only clone the ``self.tensor`` when applying deepcopy."""
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other
        other.tensor = self.tensor.clone()
        return other

    def __repr__(self) -> str:
        """Return a strings that describes the object."""
        return self.__class__.__name__ + "(\n" + str(self.tensor) + ")"

    def new_tensor(self, *args, **kwargs) -> Tensor:
        """Reload ``new_tensor`` from self.tensor."""
        return self.tensor.new_tensor(*args, **kwargs)

    def new_full(self, *args, **kwargs) -> Tensor:
        """Reload ``new_full`` from self.tensor."""
        return self.tensor.new_full(*args, **kwargs)

    def new_empty(self, *args, **kwargs) -> Tensor:
        """Reload ``new_empty`` from self.tensor."""
        return self.tensor.new_empty(*args, **kwargs)

    def new_ones(self, *args, **kwargs) -> Tensor:
        """Reload ``new_ones`` from self.tensor."""
        return self.tensor.new_ones(*args, **kwargs)

    def new_zeros(self, *args, **kwargs) -> Tensor:
        """Reload ``new_zeros`` from self.tensor."""
        return self.tensor.new_zeros(*args, **kwargs)

    def size(self, dim: Optional[int] = None) -> Union[int, torch.Size]:
        """Reload new_zeros from self.tensor."""
        # self.tensor.size(dim) cannot work when dim=None.
        return self.tensor.size() if dim is None else self.tensor.size(dim)

    def dim(self) -> int:
        """Reload ``dim`` from self.tensor."""
        return self.tensor.dim()

    @property
    def device(self) -> torch.device:
        """Reload ``device`` from self.tensor."""
        return self.tensor.device

    @property
    def dtype(self) -> torch.dtype:
        """Reload ``dtype`` from self.tensor."""
        return self.tensor.dtype

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    def numel(self) -> int:
        """Reload ``numel`` from self.tensor."""
        return self.tensor.numel()

    def numpy(self) -> np.ndarray:
        """Reload ``numpy`` from self.tensor."""
        return self.tensor.numpy()

    def to(self: T, *args, **kwargs) -> T:
        """Reload ``to`` from self.tensor."""
        return type(self)(self.tensor.to(*args, **kwargs), clone=False)

    def cpu(self: T) -> T:
        """Reload ``cpu`` from self.tensor."""
        return type(self)(self.tensor.cpu(), clone=False)

    def cuda(self: T, *args, **kwargs) -> T:
        """Reload ``cuda`` from self.tensor."""
        return type(self)(self.tensor.cuda(*args, **kwargs), clone=False)

    def clone(self: T) -> T:
        """Reload ``clone`` from self.tensor."""
        return type(self)(self.tensor)

    def detach(self: T) -> T:
        """Reload ``detach`` from self.tensor."""
        return type(self)(self.tensor.detach(), clone=False)

    def view(self: T, *shape: Tuple[int]) -> T:
        """Reload ``view`` from self.tensor."""
        return type(self)(self.tensor.view(shape), clone=False)

    def reshape(self: T, *shape: Tuple[int]) -> T:
        """Reload ``reshape`` from self.tensor."""
        return type(self)(self.tensor.reshape(shape), clone=False)

    def expand(self: T, *sizes: Tuple[int]) -> T:
        """Reload ``expand`` from self.tensor."""
        return type(self)(self.tensor.expand(sizes), clone=False)

    def repeat(self: T, *sizes: Tuple[int]) -> T:
        """Reload ``repeat`` from self.tensor."""
        return type(self)(self.tensor.repeat(sizes), clone=False)

    def transpose(self: T, dim0: int, dim1: int) -> T:
        """Reload ``transpose`` from self.tensor."""
        ndim = self.tensor.dim()
        assert dim0 != -1 and dim0 != ndim - 1
        assert dim1 != -1 and dim1 != ndim - 1
        return type(self)(self.tensor.transpose(dim0, dim1), clone=False)

    def permute(self: T, *dims: Tuple[int]) -> T:
        """Reload ``permute`` from self.tensor."""
        assert dims[-1] == -1 or dims[-1] == self.tensor.dim() - 1
        return type(self)(self.tensor.permute(dims), clone=False)

    def split(
        self: T, split_size_or_sections: Union[int, Sequence[int]], dim: int = 0
    ) -> List[T]:
        """Reload ``split`` from self.tensor."""
        assert dim != -1 and dim != self.tensor.dim() - 1
        boxes_list = self.tensor.split(split_size_or_sections, dim=dim)
        return [type(self)(boxes, clone=False) for boxes in boxes_list]

    def chunk(self: T, chunks: int, dim: int = 0) -> List[T]:
        """Reload ``chunk`` from self.tensor."""
        assert dim != -1 and dim != self.tensor.dim() - 1
        boxes_list = self.tensor.chunk(chunks, dim=dim)
        return [type(self)(boxes, clone=False) for boxes in boxes_list]

    def unbind(self: T, dim: int = 0) -> T:
        """Reload ``unbind`` from self.tensor."""
        assert dim != -1 and dim != self.tensor.dim() - 1
        boxes_list = self.tensor.unbind(dim=dim)
        return [type(self)(boxes, clone=False) for boxes in boxes_list]

    def flatten(self: T, start_dim: int = 0, end_dim: int = -2) -> T:
        """Reload ``flatten`` from self.tensor."""
        assert end_dim != -1 and end_dim != self.tensor.dim() - 1
        return type(self)(self.tensor.flatten(start_dim, end_dim), clone=False)

    def squeeze(self: T, dim: Optional[int] = None) -> T:
        """Reload ``squeeze`` from self.tensor."""
        boxes = self.tensor.squeeze() if dim is None else self.tensor.squeeze(dim)
        return type(self)(boxes, clone=False)

    def unsqueeze(self: T, dim: int) -> T:
        """Reload ``unsqueeze`` from self.tensor."""
        assert dim != -1 and dim != self.tensor.dim()
        return type(self)(self.tensor.unsqueeze(dim), clone=False)

    @classmethod
    def cat(cls: Type[T], box_list: Sequence[T], dim: int = 0) -> T:
        """Cancatenates a box instance list into one single box instance.
        Similar to ``torch.cat``.

        Args:
            box_list (Sequence[T]): A sequence of box instances.
            dim (int): The dimension over which the box are concatenated.
                Defaults to 0.

        Returns:
            T: Concatenated box instance.
        """
        assert isinstance(box_list, Sequence)
        if len(box_list) == 0:
            raise ValueError("box_list should not be a empty list.")

        assert dim != -1 and dim != box_list[0].dim() - 1
        assert all(isinstance(boxes, cls) for boxes in box_list)

        th_box_list = [boxes.tensor for boxes in box_list]
        return cls(torch.cat(th_box_list, dim=dim), clone=False)

    @classmethod
    def stack(cls: Type[T], box_list: Sequence[T], dim: int = 0) -> T:
        """Concatenates a sequence of tensors along a new dimension. Similar to
        ``torch.stack``.

        Args:
            box_list (Sequence[T]): A sequence of box instances.
            dim (int): Dimension to insert. Defaults to 0.

        Returns:
            T: Concatenated box instance.
        """
        assert isinstance(box_list, Sequence)
        if len(box_list) == 0:
            raise ValueError("box_list should not be a empty list.")

        assert dim != -1 and dim != box_list[0].dim()
        assert all(isinstance(boxes, cls) for boxes in box_list)

        th_box_list = [boxes.tensor for boxes in box_list]
        return cls(torch.stack(th_box_list, dim=dim), clone=False)

    @abstractproperty
    def centers(self) -> Tensor:
        """Return a tensor representing the centers of boxes."""
        pass

    @abstractproperty
    def areas(self) -> Tensor:
        """Return a tensor representing the areas of boxes."""
        pass

    @abstractproperty
    def widths(self) -> Tensor:
        """Return a tensor representing the widths of boxes."""
        pass

    @abstractproperty
    def heights(self) -> Tensor:
        """Return a tensor representing the heights of boxes."""
        pass

    @abstractmethod
    def flip_(self, img_shape: Tuple[int, int], direction: str = "horizontal") -> None:
        """Flip boxes horizontally or vertically in-place.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            direction (str): Flip direction, options are "horizontal",
                "vertical" and "diagonal". Defaults to "horizontal"
        """
        pass

    @abstractmethod
    def translate_(self, distances: Tuple[float, float]) -> None:
        """Translate boxes in-place.

        Args:
            distances (Tuple[float, float]): translate distances. The first
                is horizontal distance and the second is vertical distance.
        """
        pass

    @abstractmethod
    def clip_(self, img_shape: Tuple[int, int]) -> None:
        """Clip boxes according to the image shape in-place.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
        """
        pass

    @abstractmethod
    def rotate_(self, center: Tuple[float, float], angle: float) -> None:
        """Rotate all boxes in-place.

        Args:
            center (Tuple[float, float]): Rotation origin.
            angle (float): Rotation angle represented in degrees. Positive
                values mean clockwise rotation.
        """
        pass

    @abstractmethod
    def project_(self, homography_matrix: Union[Tensor, np.ndarray]) -> None:
        """Geometric transformat boxes in-place.

        Args:
            homography_matrix (Tensor or np.ndarray]):
                Shape (3, 3) for geometric transformation.
        """
        pass

    @abstractmethod
    def rescale_(self, scale_factor: Tuple[float, float]) -> None:
        """Rescale boxes w.r.t. rescale_factor in-place.

        Note:
            Both ``rescale_`` and ``resize_`` will enlarge or shrink boxes
            w.r.t ``scale_facotr``. The difference is that ``resize_`` only
            changes the width and the height of boxes, but ``rescale_`` also
            rescales the box centers simultaneously.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling boxes.
                The length should be 2.
        """
        pass

    @abstractmethod
    def resize_(self, scale_factor: Tuple[float, float]) -> None:
        """Resize the box width and height w.r.t scale_factor in-place.

        Note:
            Both ``rescale_`` and ``resize_`` will enlarge or shrink boxes
            w.r.t ``scale_facotr``. The difference is that ``resize_`` only
            changes the width and the height of boxes, but ``rescale_`` also
            rescales the box centers simultaneously.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling box
                shapes. The length should be 2.
        """
        pass

    @abstractmethod
    def is_inside(
        self,
        img_shape: Tuple[int, int],
        all_inside: bool = False,
        allowed_border: int = 0,
    ) -> BoolTensor:
        """Find boxes inside the image.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            all_inside (bool): Whether the boxes are all inside the image or
                part inside the image. Defaults to False.
            allowed_border (int): Boxes that extend beyond the image shape
                boundary by more than ``allowed_border`` are considered
                "outside" Defaults to 0.
        Returns:
            BoolTensor: A BoolTensor indicating whether the box is inside
            the image. Assuming the original boxes have shape (m, n, box_dim),
            the output has shape (m, n).
        """
        pass

    @abstractmethod
    def find_inside_points(
        self, points: Tensor, is_aligned: bool = False
    ) -> BoolTensor:
        """Find inside box points. Boxes dimension must be 2.

        Args:
            points (Tensor): Points coordinates. Has shape of (m, 2).
            is_aligned (bool): Whether ``points`` has been aligned with boxes
                or not. If True, the length of boxes and ``points`` should be
                the same. Defaults to False.

        Returns:
            BoolTensor: A BoolTensor indicating whether a point is inside
            boxes. Assuming the boxes has shape of (n, box_dim), if
            ``is_aligned`` is False. The index has shape of (m, n). If
            ``is_aligned`` is True, m should be equal to n and the index has
            shape of (m, ).
        """
        pass

    @abstractstaticmethod
    def overlaps(
        boxes1: "BaseBoxes",
        boxes2: "BaseBoxes",
        mode: str = "iou",
        is_aligned: bool = False,
        eps: float = 1e-6,
    ) -> Tensor:
        """Calculate overlap between two set of boxes with their types
        converted to the present box type.

        Args:
            boxes1 (:obj:`BaseBoxes`): BaseBoxes with shape of (m, box_dim)
                or empty.
            boxes2 (:obj:`BaseBoxes`): BaseBoxes with shape of (n, box_dim)
                or empty.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground). Defaults to "iou".
            is_aligned (bool): If True, then m and n must be equal. Defaults
                to False.
            eps (float): A value added to the denominator for numerical
                stability. Defaults to 1e-6.

        Returns:
            Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
        """
        pass

    @abstractstaticmethod
    def from_instance_masks(masks: MaskType) -> "BaseBoxes":
        """Create boxes from instance masks.

        Args:
            masks (:obj:`BitmapMasks` or :obj:`PolygonMasks`): BitmapMasks or
                PolygonMasks instance with length of n.

        Returns:
            :obj:`BaseBoxes`: Converted boxes with shape of (n, box_dim).
        """
        pass


def samplelist_boxtype2tensor(batch_data_samples: SampleList) -> SampleList:
    for data_samples in batch_data_samples:
        if "gt_instances" in data_samples:
            bboxes = data_samples.gt_instances.get("bboxes", None)
            if isinstance(bboxes, BaseBoxes):
                data_samples.gt_instances.bboxes = bboxes.tensor
        if "pred_instances" in data_samples:
            bboxes = data_samples.pred_instances.get("bboxes", None)
            if isinstance(bboxes, BaseBoxes):
                data_samples.pred_instances.bboxes = bboxes.tensor
        if "ignored_instances" in data_samples:
            bboxes = data_samples.ignored_instances.get("bboxes", None)
            if isinstance(bboxes, BaseBoxes):
                data_samples.ignored_instances.bboxes = bboxes.tensor


box_types: dict = {}
_box_type_to_name: dict = {}
def get_box_type(box_type: Union[str, type]) -> Tuple[str, type]:
    """get both box type name and class.

    Args:
        box_type (str or type): Single box type name or class.

    Returns:
        Tuple[str, type]: A tuple of box type name and class.
    """
    if isinstance(box_type, str):
        type_name = box_type.lower()
        assert type_name in box_types, \
            f"Box type {type_name} hasn't been registered in box_types."
        type_cls = box_types[type_name]
    elif issubclass(box_type, BaseBoxes):
        assert box_type in _box_type_to_name, \
            f"Box type {box_type} hasn't been registered in box_types."
        type_name = _box_type_to_name[box_type]
        type_cls = box_type
    else:
        raise KeyError('box_type must be a str or class inheriting from '
                       f'BaseBoxes, but got {type(box_type)}.')
    return type_name, type_cls


BoxType = Union[np.ndarray, Tensor, BaseBoxes]


def convert_box_type(
    boxes: BoxType,
    *,
    src_type: Union[str, type] = None,
    dst_type: Union[str, type] = None,
) -> BoxType:
    """Convert boxes from source type to destination type.

    If ``boxes`` is a instance of BaseBoxes, the ``src_type`` will be set
    as the type of ``boxes``.

    Args:
        boxes (np.ndarray or Tensor or :obj:`BaseBoxes`): boxes need to
            convert.
        src_type (str or type, Optional): source box type. Defaults to None.
        dst_type (str or type, Optional): destination box type. Defaults to
            None.

    Returns:
        Union[np.ndarray, Tensor, :obj:`BaseBoxes`]: Converted boxes. It's type
        is consistent with the input's type.
    """
    assert dst_type is not None
    dst_type_name, dst_type_cls = get_box_type(dst_type)

    is_box_cls = False
    is_numpy = False
    if isinstance(boxes, BaseBoxes):
        src_type_name, _ = get_box_type(type(boxes))
        is_box_cls = True
    elif isinstance(boxes, (Tensor, np.ndarray)):
        assert src_type is not None
        src_type_name, _ = get_box_type(src_type)
        if isinstance(boxes, np.ndarray):
            is_numpy = True
    else:
        raise TypeError(
            "boxes must be a instance of BaseBoxes, Tensor or "
            f"ndarray, but get {type(boxes)}."
        )

    if src_type_name == dst_type_name:
        return boxes

    converter_name = src_type_name + "2" + dst_type_name
    assert (
        converter_name in box_converters
    ), "Convert function hasn't been registered in box_converters."
    converter = box_converters[converter_name]

    if is_box_cls:
        boxes = converter(boxes.tensor)
        return dst_type_cls(boxes)
    elif is_numpy:
        boxes = converter(torch.from_numpy(boxes))
        return boxes.numpy()
    else:
        return converter(boxes)


def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Defaults to 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    assert "parrots" not in version_str
    version = parse(version_str)
    assert version.release, f"failed to parse version {version_str}"
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {"a": -3, "b": -2, "rc": -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(
                    f"unknown prerelease version {version.pre[0]}, "
                    "version checking may go wrong"
                )
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])  # type: ignore
    else:
        release.extend([0, 0])
    return tuple(release)


def stack_batch(
    tensor_list: List[torch.Tensor],
    pad_size_divisor: int = 1,
    pad_value: Union[int, float] = 0,
) -> torch.Tensor:
    """Stack multiple tensors to form a batch and pad the tensor to the max
    shape use the right bottom padding mode in these images. If
    ``pad_size_divisor > 0``, add padding to ensure the shape of each dim is
    divisible by ``pad_size_divisor``.

    Args:
        tensor_list (List[Tensor]): A list of tensors with the same dim.
        pad_size_divisor (int): If ``pad_size_divisor > 0``, add padding
            to ensure the shape of each dim is divisible by
            ``pad_size_divisor``. This depends on the model, and many
            models need to be divisible by 32. Defaults to 1
        pad_value (int, float): The padding value. Defaults to 0.

    Returns:
       Tensor: The n dim tensor.
    """
    assert isinstance(
        tensor_list, list
    ), f"Expected input type to be list, but got {type(tensor_list)}"
    assert tensor_list, "`tensor_list` could not be an empty list"
    assert len({tensor.ndim for tensor in tensor_list}) == 1, (
        f"Expected the dimensions of all tensors must be the same, "
        f"but got {[tensor.ndim for tensor in tensor_list]}"
    )

    dim = tensor_list[0].dim()
    num_img = len(tensor_list)
    all_sizes: torch.Tensor = torch.Tensor([tensor.shape for tensor in tensor_list])
    max_sizes = (
        torch.ceil(torch.max(all_sizes, dim=0)[0] / pad_size_divisor) * pad_size_divisor
    )
    padded_sizes = max_sizes - all_sizes
    # The first dim normally means channel,  which should not be padded.
    padded_sizes[:, 0] = 0
    if padded_sizes.sum() == 0:
        return torch.stack(tensor_list)
    # `pad` is the second arguments of `F.pad`. If pad is (1, 2, 3, 4),
    # it means that padding the last dim with 1(left) 2(right), padding the
    # penultimate dim to 3(top) 4(bottom). The order of `pad` is opposite of
    # the `padded_sizes`. Therefore, the `padded_sizes` needs to be reversed,
    # and only odd index of pad should be assigned to keep padding "right" and
    # "bottom".
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate(tensor_list):
        batch_tensor.append(F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))
    return torch.stack(batch_tensor)


@MODELS.register_module()
class ImgDataPreprocessor(BaseDataPreprocessor):
    """Image pre-processor for normalization and bgr to rgb conversion.

    Accepts the data sampled by the dataloader, and preprocesses it into the
    format of the model input. ``ImgDataPreprocessor`` provides the
    basic data pre-processing as follows

    - Collates and moves data to the target device.
    - Converts inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalizes image with defined std and mean.
    - Pads inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.

    For ``ImgDataPreprocessor``, the dimension of the single inputs must be
    (3, H, W).

    Note:
        ``ImgDataPreprocessor`` and its subclass is built in the
        constructor of :class:`BaseDataset`.

    Args:
        mean (Sequence[float or int], optional): The pixel mean of image
            channels. If ``bgr_to_rgb=True`` it means the mean value of R,
            G, B channels. If the length of `mean` is 1, it means all
            channels have the same mean value, or the input is a gray image.
            If it is not specified, images will not be normalized. Defaults
            None.
        std (Sequence[float or int], optional): The pixel standard deviation of
            image channels. If ``bgr_to_rgb=True`` it means the standard
            deviation of R, G, B channels. If the length of `std` is 1,
            it means all channels have the same standard deviation, or the
            input is a gray image.  If it is not specified, images will
            not be normalized. Defaults None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        non_blocking (bool): Whether block current process
            when transferring data to device.
            New in version v0.3.0.

    Note:
        if images do not need to be normalized, `std` and `mean` should be
        both set to None, otherwise both of them should be set to a tuple of
        corresponding values.
    """

    def __init__(
        self,
        mean: Optional[Sequence[Union[float, int]]] = None,
        std: Optional[Sequence[Union[float, int]]] = None,
        pad_size_divisor: int = 1,
        pad_value: Union[float, int] = 0,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        non_blocking: Optional[bool] = False,
    ):
        super().__init__(non_blocking)
        assert not (
            bgr_to_rgb and rgb_to_bgr
        ), "`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time"
        assert (mean is None) == (
            std is None
        ), "mean and std should be both None or tuple"
        if mean is not None:
            assert len(mean) == 3 or len(mean) == 1, (
                "`mean` should have 1 or 3 values, to be compatible with "
                f"RGB or gray image, but got {len(mean)} values"
            )
            assert len(std) == 3 or len(std) == 1, (  # type: ignore
                "`std` should have 1 or 3 values, to be compatible with RGB "  # type: ignore # noqa: E501
                f"or gray image, but got {len(std)} values"
            )  # type: ignore
            self._enable_normalize = True
            self.register_buffer("mean", torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer("std", torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False
        self._channel_conversion = rgb_to_bgr or bgr_to_rgb
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        """Performs normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataset. If the collate
                function of DataLoader is :obj:`pseudo_collate`, data will be a
                list of dict. If collate function is :obj:`default_collate`,
                data will be a tuple with batch input tensor and list of data
                samples.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.

        Returns:
            dict or list: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        _batch_inputs = data["inputs"]
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_inputs = []
            for _batch_input in _batch_inputs:
                # channel transform
                if self._channel_conversion:
                    _batch_input = _batch_input[[2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_input = _batch_input.float()
                # Normalization.
                if self._enable_normalize:
                    if self.mean.shape[0] == 3:
                        assert _batch_input.dim() == 3 and _batch_input.shape[0] == 3, (
                            "If the mean has 3 values, the input tensor "
                            "should in shape of (3, H, W), but got the tensor "
                            f"with shape {_batch_input.shape}"
                        )
                    _batch_input = (_batch_input - self.mean) / self.std
                batch_inputs.append(_batch_input)
            # Pad and stack Tensor.
            batch_inputs = stack_batch(
                batch_inputs, self.pad_size_divisor, self.pad_value
            )
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                "The input of `ImgDataPreprocessor` should be a NCHW tensor "
                "or a list of tensor, but got a tensor with shape: "
                f"{_batch_inputs.shape}"
            )
            if self._channel_conversion:
                _batch_inputs = _batch_inputs[:, [2, 1, 0], ...]
            # Convert to float after channel conversion to ensure
            # efficiency
            _batch_inputs = _batch_inputs.float()
            if self._enable_normalize:
                _batch_inputs = (_batch_inputs - self.mean) / self.std
            h, w = _batch_inputs.shape[2:]
            target_h = math.ceil(h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            batch_inputs = F.pad(
                _batch_inputs, (0, pad_w, 0, pad_h), "constant", self.pad_value
            )
        else:
            raise TypeError(
                "Output of `cast_data` should be a dict of "
                "list/tuple with inputs and data_samples, "
                f"but got {type(data)}： {data}"
            )
        data["inputs"] = batch_inputs
        data.setdefault("data_samples", None)
        return data


@MODELS.register_module()
class DetDataPreprocessor(ImgDataPreprocessor):
    """Image pre-processor for detection tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It supports batch augmentations.
    2. It will additionally append batch_input_shape and pad_shape
    to data_samples considering the object detection task.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded pixel value for instance masks.
            Defaults to 0.
        pad_seg (bool): Whether to pad semantic segmentation maps.
            Defaults to False.
        seg_pad_value (int): The padded pixel value for semantic
            segmentation maps. Defaults to 255.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        boxtype2tensor (bool): Whether to keep the ``BaseBoxes`` type of
            bboxes data or not. Defaults to True.
        non_blocking (bool): Whether block current process
            when transferring data to device. Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        pad_size_divisor: int = 1,
        pad_value: Union[float, int] = 0,
        pad_mask: bool = False,
        mask_pad_value: int = 0,
        pad_seg: bool = False,
        seg_pad_value: int = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        boxtype2tensor: bool = True,
        non_blocking: Optional[bool] = False,
        batch_augments: Optional[List[dict]] = None,
    ):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking,
        )
        if batch_augments is not None:
            self.batch_augments = nn.ModuleList(
                [MODELS.build(aug) for aug in batch_augments]
            )
        else:
            self.batch_augments = None
        self.pad_mask = pad_mask
        self.mask_pad_value = mask_pad_value
        self.pad_seg = pad_seg
        self.seg_pad_value = seg_pad_value
        self.boxtype2tensor = boxtype2tensor

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        batch_pad_shape = self._get_pad_shape(data)
        data = super().forward(data=data, training=training)
        inputs, data_samples = data["inputs"], data["data_samples"]

        if data_samples is not None:
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            batch_input_shape = tuple(inputs[0].size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo(
                    {"batch_input_shape": batch_input_shape, "pad_shape": pad_shape}
                )

            if self.boxtype2tensor:
                samplelist_boxtype2tensor(data_samples)

            if self.pad_mask and training:
                self.pad_gt_masks(data_samples)

            if self.pad_seg and training:
                self.pad_gt_sem_seg(data_samples)

        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        return {"inputs": inputs, "data_samples": data_samples}

    def _get_pad_shape(self, data: dict) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        _batch_inputs = data["inputs"]
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                pad_h = (
                    int(np.ceil(ori_input.shape[1] / self.pad_size_divisor))
                    * self.pad_size_divisor
                )
                pad_w = (
                    int(np.ceil(ori_input.shape[2] / self.pad_size_divisor))
                    * self.pad_size_divisor
                )
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                "The input of `ImgDataPreprocessor` should be a NCHW tensor "
                "or a list of tensor, but got a tensor with shape: "
                f"{_batch_inputs.shape}"
            )
            pad_h = (
                int(np.ceil(_batch_inputs.shape[1] / self.pad_size_divisor))
                * self.pad_size_divisor
            )
            pad_w = (
                int(np.ceil(_batch_inputs.shape[2] / self.pad_size_divisor))
                * self.pad_size_divisor
            )
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            raise TypeError(
                "Output of `cast_data` should be a dict "
                "or a tuple with inputs and data_samples, but got"
                f"{type(data)}： {data}"
            )
        return batch_pad_shape

    def pad_gt_masks(self, batch_data_samples: Sequence[DetDataSample]) -> None:
        """Pad gt_masks to shape of batch_input_shape."""
        if "masks" in batch_data_samples[0].gt_instances:
            for data_samples in batch_data_samples:
                masks = data_samples.gt_instances.masks
                data_samples.gt_instances.masks = masks.pad(
                    data_samples.batch_input_shape, pad_val=self.mask_pad_value
                )

    def pad_gt_sem_seg(self, batch_data_samples: Sequence[DetDataSample]) -> None:
        """Pad gt_sem_seg to shape of batch_input_shape."""
        if "gt_sem_seg" in batch_data_samples[0]:
            for data_samples in batch_data_samples:
                gt_sem_seg = data_samples.gt_sem_seg.sem_seg
                h, w = gt_sem_seg.shape[-2:]
                pad_h, pad_w = data_samples.batch_input_shape
                gt_sem_seg = F.pad(
                    gt_sem_seg,
                    pad=(0, max(pad_w - w, 0), 0, max(pad_h - h, 0)),
                    mode="constant",
                    value=self.seg_pad_value,
                )
                data_samples.gt_sem_seg = PixelData(sem_seg=gt_sem_seg)


@MODELS.register_module()
class YOLOWDetDataPreprocessor(DetDataPreprocessor):
    """Rewrite collate_fn to get faster training speed.

    Note: It must be used together with `mmyolo.datasets.utils.yolow_collate`
    """

    def __init__(self, *args, non_blocking: Optional[bool] = True, **kwargs):
        super().__init__(*args, non_blocking=non_blocking, **kwargs)

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``DetDataPreprocessorr``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        if not training:
            return super().forward(data, training)

        data = self.cast_data(data)
        inputs, data_samples = data["inputs"], data["data_samples"]
        assert isinstance(data["data_samples"], dict)

        # TODO: Supports multi-scale training
        if self._channel_conversion and inputs.shape[1] == 3:
            inputs = inputs[:, [2, 1, 0], ...]
        if self._enable_normalize:
            inputs = (inputs - self.mean) / self.std

        if self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        img_metas = [{"batch_input_shape": inputs.shape[2:]}] * len(inputs)
        data_samples_output = {
            "bboxes_labels": data_samples["bboxes_labels"],
            "texts": data_samples["texts"],
            "img_metas": img_metas,
        }
        if "masks" in data_samples:
            data_samples_output["masks"] = data_samples["masks"]
        if "is_detection" in data_samples:
            data_samples_output["is_detection"] = data_samples["is_detection"]

        return {"inputs": inputs, "data_samples": data_samples_output}


def is_cuda_available() -> bool:
    """Returns True if cuda devices exist."""
    return torch.cuda.is_available()

try:
    import torch_npu  # noqa: F401

    # Enable operator support for dynamic shape and
    # binary operator support on the NPU.
    npu_jit_compile = bool(os.getenv("NPUJITCompile", False))
    torch.npu.set_compile_mode(jit_compile=npu_jit_compile)
    IS_NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()
except Exception:
    IS_NPU_AVAILABLE = False

def is_npu_available() -> bool:
    """Returns True if Ascend PyTorch and npu devices exist."""
    return IS_NPU_AVAILABLE


DEVICE = 'cpu'
if is_npu_available():
    DEVICE = 'npu'
elif is_cuda_available():
    DEVICE = 'cuda'

def get_device() -> str:
    """Returns the currently existing device type.

    Returns:
        str: cuda | npu | mlu | mps | cpu.
    """
    return DEVICE
@contextmanager
def autocast(
    device_type: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    enabled: bool = True,
    cache_enabled: Optional[bool] = None,
):
    """A wrapper of ``torch.autocast`` and ``toch.cuda.amp.autocast``.

    Pytorch 1.5.0 provide ``torch.cuda.amp.autocast`` for running in
    mixed precision , and update it to ``torch.autocast`` in 1.10.0.
    Both interfaces have different arguments, and ``torch.autocast``
    support running with cpu additionally.

    This function provides a unified interface by wrapping
    ``torch.autocast`` and ``torch.cuda.amp.autocast``, which resolves the
    compatibility issues that ``torch.cuda.amp.autocast`` does not support
    running mixed precision with cpu, and both contexts have different
    arguments. We suggest users using this function in the code
    to achieve maximized compatibility of different PyTorch versions.

    Note:
        ``autocast`` requires pytorch version >= 1.5.0. If pytorch version
        <= 1.10.0 and cuda is not available, it will raise an error with
        ``enabled=True``, since ``torch.cuda.amp.autocast`` only support cuda
        mode.

    Examples:
         >>> # case1: 1.10 > Pytorch version >= 1.5.0
         >>> with autocast():
         >>>    # run in mixed precision context
         >>>    pass
         >>> with autocast(device_type='cpu')::
         >>>    # raise error, torch.cuda.amp.autocast only support cuda mode.
         >>>    pass
         >>> # case2: Pytorch version >= 1.10.0
         >>> with autocast():
         >>>    # default cuda mixed precision context
         >>>    pass
         >>> with autocast(device_type='cpu'):
         >>>    # cpu mixed precision context
         >>>    pass
         >>> with autocast(
         >>>     device_type='cuda', enabled=True, cache_enabled=True):
         >>>    # enable precision context with more specific arguments.
         >>>    pass

    Args:
        device_type (str, required):  Whether to use 'cuda' or 'cpu' device.
        enabled(bool):  Whether autocasting should be enabled in the region.
            Defaults to True
        dtype (torch_dtype, optional):  Whether to use ``torch.float16`` or
            ``torch.bfloat16``.
        cache_enabled(bool, optional):  Whether the weight cache inside
            autocast should be enabled.
    """
    # If `enabled` is True, enable an empty context and all calculations
    # are performed under fp32.
    assert digit_version(TORCH_VERSION) >= digit_version("1.5.0"), (
        "The minimum pytorch version requirements of mmengine is 1.5.0, but "
        f"got {TORCH_VERSION}"
    )

    if digit_version("1.5.0") <= digit_version(TORCH_VERSION) < digit_version("1.10.0"):
        # If pytorch version is between 1.5.0 and 1.10.0, the default value of
        # dtype for `torch.cuda.amp.autocast` is torch.float16.
        assert device_type == "cuda" or device_type is None, (
            "Pytorch version under 1.10.0 only supports running automatic "
            "mixed training with cuda"
        )
        if dtype is not None or cache_enabled is not None:
            print_log(
                f"{dtype} and {device_type} will not work for "
                "`autocast` since your Pytorch version: "
                f"{TORCH_VERSION} <= 1.10.0",
                logger="current",
                level=logging.WARNING,
            )

        if is_npu_available():
            with torch.npu.amp.autocast(enabled=enabled):
                yield
        elif is_cuda_available():
            with torch.cuda.amp.autocast(enabled=enabled):
                yield
        else:
            if not enabled:
                yield
            else:
                raise RuntimeError(
                    "If pytorch versions is between 1.5.0 and 1.10, "
                    "`autocast` is only available in gpu mode"
                )

    else:
        # Modified from https://github.com/pytorch/pytorch/blob/master/torch/amp/autocast_mode.py # noqa: E501
        # This code should update with the `torch.autocast`.
        if cache_enabled is None:
            cache_enabled = torch.is_autocast_cache_enabled()
        device = get_device()
        device_type = device if device_type is None else device_type

        if device_type == "cuda":
            if dtype is None:
                dtype = torch.get_autocast_gpu_dtype()

            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "Current CUDA Device does not support bfloat16. Please "
                    "switch dtype to float16."
                )

        elif device_type == "cpu":
            if dtype is None:
                dtype = torch.bfloat16
            assert (
                dtype == torch.bfloat16
            ), "In CPU autocast, only support `torch.bfloat16` dtype"

        elif device_type == "mlu":
            pass

        elif device_type == "npu":
            pass

        else:
            # Device like MPS does not support fp16 training or testing.
            # If an inappropriate device is set and fp16 is enabled, an error
            # will be thrown.
            if enabled is False:
                yield
                return
            else:
                raise ValueError(
                    "User specified autocast device_type must be "
                    f"cuda or cpu, but got {device_type}"
                )

        with torch.autocast(
            device_type=device_type,
            enabled=enabled,
            dtype=dtype,
            cache_enabled=cache_enabled,
        ):
            yield

class TimerError(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(message)

class Timer:
    """A flexible Timer class.

    Examples:
        >>> import time
        >>> import mmcv
        >>> with mmcv.Timer():
        >>>     # simulate a code block that will run for 1s
        >>>     time.sleep(1)
        1.000
        >>> with mmcv.Timer(print_tmpl='it takes {:.1f} seconds'):
        >>>     # simulate a code block that will run for 1s
        >>>     time.sleep(1)
        it takes 1.0 seconds
        >>> timer = mmcv.Timer()
        >>> time.sleep(0.5)
        >>> print(timer.since_start())
        0.500
        >>> time.sleep(0.5)
        >>> print(timer.since_last_check())
        0.500
        >>> print(timer.since_start())
        1.000
    """

    def __init__(self, start=True, print_tmpl=None):
        self._is_running = False
        self.print_tmpl = print_tmpl if print_tmpl else '{:.3f}'
        if start:
            self.start()

    @property
    def is_running(self):
        """bool: indicate whether the timer is running"""
        return self._is_running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        print(self.print_tmpl.format(self.since_last_check()))
        self._is_running = False

    def start(self):
        """Start the timer."""
        if not self._is_running:
            self._t_start = ti()
            self._is_running = True
        self._t_last = ti()

    def since_start(self):
        """Total time since the timer is started.

        Returns:
            float: Time in seconds.
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        self._t_last = ti()
        return self._t_last - self._t_start

    def since_last_check(self):
        """Time since the last checking.

        Either :func:`since_start` or :func:`since_last_check` is a checking
        operation.

        Returns:
            float: Time in seconds.
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        dur = ti() - self._t_last
        self._t_last = ti()
        return dur


class ProgressBar:
    """A progress bar which can print the progress."""

    def __init__(self, task_num=0, bar_width=50, start=True, file=sys.stdout):
        self.task_num = task_num
        self.bar_width = bar_width
        self.completed = 0
        self.file = file
        if start:
            self.start()

    @property
    def terminal_width(self):
        width, _ = get_terminal_size()
        return width

    def start(self):
        if self.task_num > 0:
            self.file.write(
                f'[{" " * self.bar_width}] 0/{self.task_num}, ' "elapsed: 0s, ETA:"
            )
        else:
            self.file.write("completed: 0, elapsed: 0s")
        self.file.flush()
        self.timer = Timer()

    def update(self, num_tasks=1):
        assert num_tasks > 0
        self.completed += num_tasks
        elapsed = self.timer.since_start()
        if elapsed > 0:
            fps = self.completed / elapsed
        else:
            fps = float("inf")
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            msg = (
                f"\r[{{}}] {self.completed}/{self.task_num}, "
                f"{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, "
                f"ETA: {eta:5}s"
            )

            bar_width = min(
                self.bar_width,
                int(self.terminal_width - len(msg)) + 2,
                int(self.terminal_width * 0.6),
            )
            bar_width = max(2, bar_width)
            mark_width = int(bar_width * percentage)
            bar_chars = ">" * mark_width + " " * (bar_width - mark_width)
            self.file.write(msg.format(bar_chars))
        else:
            self.file.write(
                f"completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,"
                f" {fps:.1f} tasks/s"
            )
        self.file.flush()


def get_test_pipeline_cfg(cfg: Union[str, ConfigDict]) -> ConfigDict:
    """Get the test dataset pipeline from entire config.

    Args:
        cfg (str or :obj:`ConfigDict`): the entire config. Can be a config
            file or a ``ConfigDict``.

    Returns:
        :obj:`ConfigDict`: the config of test dataset.
    """
    if isinstance(cfg, str):
        cfg = Config.fromfile(cfg)

    def _get_test_pipeline_cfg(dataset_cfg):
        if "pipeline" in dataset_cfg:
            return dataset_cfg.pipeline
        # handle dataset wrapper
        elif "dataset" in dataset_cfg:
            return _get_test_pipeline_cfg(dataset_cfg.dataset)
        # handle dataset wrappers like ConcatDataset
        elif "datasets" in dataset_cfg:
            return _get_test_pipeline_cfg(dataset_cfg.datasets[0])

        raise RuntimeError("Cannot find `pipeline` in `test_dataloader`")

    return _get_test_pipeline_cfg(cfg.test_dataloader.dataset)


def deprecated_api_warning(name_dict: dict,
                           cls_name: Optional[str] = None) -> Callable:
    """A decorator to check if some arguments are deprecate and try to replace
    deprecate src_arg_name to dst_arg_name.

    Args:
        name_dict(dict):
            key (str): Deprecate argument names.
            val (str): Expected argument names.

    Returns:
        func: New function.
    """

    def api_warning_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get name of the function
            func_name = old_func.__name__
            if cls_name is not None:
                func_name = f'{cls_name}.{func_name}'
            if args:
                arg_names = args_info.args[:len(args)]
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in arg_names:
                        warnings.warn(
                            f'"{src_arg_name}" is deprecated in '
                            f'`{func_name}`, please use "{dst_arg_name}" '
                            'instead', DeprecationWarning)
                        arg_names[arg_names.index(src_arg_name)] = dst_arg_name
            if kwargs:
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in kwargs:
                        assert dst_arg_name not in kwargs, (
                            f'The expected behavior is to replace '
                            f'the deprecated key `{src_arg_name}` to '
                            f'new key `{dst_arg_name}`, but got them '
                            f'in the arguments at the same time, which '
                            f'is confusing. `{src_arg_name} will be '
                            f'deprecated in the future, please '
                            f'use `{dst_arg_name}` instead.')

                        warnings.warn(
                            f'"{src_arg_name}" is deprecated in '
                            f'`{func_name}`, please use "{dst_arg_name}" '
                            'instead', DeprecationWarning)
                        kwargs[dst_arg_name] = kwargs.pop(src_arg_name)

            # apply converted arguments to the decorated method
            output = old_func(*args, **kwargs)
            return output

        return new_func

    return api_warning_wrapper


ArrayOrTensor = Union[np.ndarray, torch.Tensor]
def _to_torch(x: ArrayOrTensor, device=None, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    raise TypeError("Expected np.ndarray or torch.Tensor")


def nms(
    boxes: ArrayOrTensor,
    scores: ArrayOrTensor,
    iou_threshold: float,
    offset: int = 0,
    score_threshold: float = 0.0,
    max_num: int = -1,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Pure-PyTorch NMS. Returns (dets, inds).
    - boxes: (N,4) [x1,y1,x2,y2]
    - scores: (N,)
    - iou_threshold: float
    - offset: 0 or 1 (area calculation adds offset)
    - score_threshold: discard boxes with score <= this before NMS
    - max_num: keep at most this many boxes (if >0)

    Returns:
      dets: Tensor (M,5) = [x1,y1,x2,y2,score] (same dtype as boxes)
      inds: LongTensor (M,) indices into original input order
    """
    # convert to torch on same device
    boxes_t = _to_torch(boxes)
    scores_t = _to_torch(scores)

    assert boxes_t.ndim == 2 and boxes_t.size(1) == 4
    assert scores_t.ndim == 1 and boxes_t.size(0) == scores_t.size(0)
    assert offset in (0, 1)

    device = boxes_t.device
    dtype = boxes_t.dtype

    # filter by score_threshold
    if score_threshold > 0:
        keep_mask = scores_t > float(score_threshold)
        if keep_mask.sum().item() == 0:
            # nothing kept
            empty_dets = boxes_t.new_zeros((0, 5))
            empty_inds = torch.empty((0,), dtype=torch.long, device=device)
            return (
                empty_dets
                if isinstance(boxes, torch.Tensor)
                else empty_dets.cpu().numpy()
            ), (
                empty_inds
                if isinstance(scores, torch.Tensor)
                else empty_inds.cpu().numpy()
            )
        boxes_t = boxes_t[keep_mask]
        scores_t = scores_t[keep_mask]
        # map kept indices back to original
        orig_inds = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
    else:
        orig_inds = torch.arange(boxes_t.size(0), device=device, dtype=torch.long)

    x1 = boxes_t[:, 0]
    y1 = boxes_t[:, 1]
    x2 = boxes_t[:, 2]
    y2 = boxes_t[:, 3]

    areas = (x2 - x1 + offset).clamp(min=0) * (y2 - y1 + offset).clamp(min=0)

    # sort by score descending
    _, order = scores_t.sort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        # compute IoU of the highest score box with the rest
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        inter_w = (xx2 - xx1 + offset).clamp(min=0)
        inter_h = (yy2 - yy1 + offset).clamp(min=0)
        inter = inter_w * inter_h

        union = areas[i] + areas[order[1:]] - inter
        # avoid division by zero
        iou = torch.zeros_like(inter)
        mask = union > 0
        iou[mask] = inter[mask] / union[mask]

        # keep boxes with IoU <= threshold
        rem_mask = iou <= iou_threshold
        order = order[1:][rem_mask]

    # convert kept indices (relative to filtered boxes) to original indices
    keep = torch.as_tensor(keep, dtype=torch.long, device=device)
    final_inds = orig_inds[keep]

    # optionally limit to max_num
    if max_num > 0 and final_inds.numel() > max_num:
        final_inds = final_inds[:max_num]
        keep = keep[:max_num]

    final_boxes = boxes_t[keep]
    final_scores = scores_t[keep].unsqueeze(1)
    dets = torch.cat([final_boxes, final_scores], dim=1)

    # return same type as inputs (torch or numpy)
    if isinstance(boxes, np.ndarray) or isinstance(scores, np.ndarray):
        return dets.cpu().numpy(), final_inds.cpu().numpy()
    return dets, final_inds
