"""
hackerargs - Simple argument container

A minimal implementation to allow the codebase to import `args` as a dict.
The original `hackerargs` appears to be a custom internal library.
"""

from typing import Any, Dict, Optional
import sys
import os

_args: Dict[str, Any] = {}


def set_arg(key: str, value: Any) -> None:
    _args[key] = value


def get_arg(key: str, default: Any = None) -> Any:
    return _args.get(key, default)


def load_args_from_env(prefix: str = "PIU_") -> None:
    for key, value in os.environ.items():
        if key.startswith(prefix):
            arg_key = key[len(prefix):].lower()
            _args[arg_key] = value


def load_args_from_file(filepath: str) -> None:
    import json
    if os.path.exists(filepath):
        with open(filepath) as f:
            data = json.load(f)
            _args.update(data)


args = _args

__all__ = ['args', 'set_arg', 'get_arg', 'load_args_from_env', 'load_args_from_file']