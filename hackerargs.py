from __future__ import annotations
"""
hackerargs - Simple argument container
"""
from typing import Any, Dict, Optional
import sys
import os

class ArgsDict(dict):
    def parse_args(self, parser, config_file=None):
        parsed_args = parser.parse_args()
        self.update(vars(parsed_args))
        return parsed_args
    
    def setdefault(self, key, default):
        if key not in self:
            self[key] = default
        return self[key]

args = ArgsDict()

def set_arg(key: str, value: Any) -> None:
    args[key] = value

def get_arg(key: str, default: Any = None) -> Any:
    return args.get(key, default)

__all__ = ['args', 'set_arg', 'get_arg']