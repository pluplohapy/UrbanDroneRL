from __future__ import annotations

import importlib
import inspect
from types import ModuleType

def _is_syncable_name(name: str, value) -> bool:
    if not name.isupper():
        return False
    if inspect.ismodule(value) or inspect.isfunction(value) or inspect.ismethod(value):
        return False
    return True

def sync_runtime_config(stage_config: ModuleType) -> ModuleType:
    runtime_config = importlib.import_module("config")

    for name in dir(stage_config):
        if name.startswith("_"):
            continue
        value = getattr(stage_config, name)
        if _is_syncable_name(name, value):
            setattr(runtime_config, name, value)

    return runtime_config
