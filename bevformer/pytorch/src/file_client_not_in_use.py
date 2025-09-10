# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Optional, Union


class HardDiskBackend:
    def get(self, filepath: Union[str, Path]) -> bytes:
        with open(filepath, "rb") as f:
            return f.read()

    def get_text(self, filepath: Union[str, Path], encoding: str = "utf-8") -> str:
        with open(filepath, "r", encoding=encoding) as f:
            return f.read()


class FileClient:

    _backends = {
        "disk": HardDiskBackend,
    }

    _instances = {}

    def __new__(cls, backend: Optional[str] = None, **kwargs):
        backend = backend or "disk"
        if backend not in cls._backends:
            raise ValueError(
                f"Backend {backend} is not supported. Currently supported ones are {list(cls._backends.keys())}"
            )

        # use a simple instance cache keyed by backend and kwargs
        arg_key = backend
        for key, value in kwargs.items():
            arg_key += f":{key}:{value}"

        if arg_key in cls._instances:
            return cls._instances[arg_key]

        instance = super().__new__(cls)
        instance.client = cls._backends[backend](**kwargs)
        cls._instances[arg_key] = instance
        return instance

    @classmethod
    def infer_client(
        cls,
        file_client_args: Optional[dict] = None,
        uri: Optional[Union[str, Path]] = None,
    ) -> "FileClient":
        # Only disk backend is supported; ignore uri and default to disk when args are not provided
        return (
            cls(**file_client_args)
            if file_client_args is not None
            else cls(backend="disk")
        )

    def get(self, filepath: Union[str, Path]) -> bytes:
        return self.client.get(filepath)

    def get_text(self, filepath: Union[str, Path], encoding: str = "utf-8") -> str:
        return self.client.get_text(filepath, encoding)
