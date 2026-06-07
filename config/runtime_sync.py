"""
Utilities for syncing stage-specific config modules into runtime ``config``.

Why this exists:
- Environments import ``config`` (the module), not the stage module directly.
- Training/eval/visualization load a stage config via ``load_config(...)``.
- Without explicit syncing, runtime may keep base defaults in some fields.
"""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType


def _is_syncable_name(name: str, value) -> bool:
    """Sync only constant-like values (UPPER_CASE, non-callable, non-module)."""
    if not name.isupper():
        return False
    if inspect.ismodule(value) or inspect.isfunction(value) or inspect.ismethod(value):
        return False
    return True


def sync_runtime_config(stage_config: ModuleType) -> ModuleType:
    """
    Copy all uppercase fields from ``stage_config`` into runtime ``config`` module.

    Returns:
        The runtime ``config`` module after synchronization.
    """
    runtime_config = importlib.import_module("config")

    for name in dir(stage_config):
        if name.startswith("_"):
            continue
        value = getattr(stage_config, name)
        if _is_syncable_name(name, value):
            setattr(runtime_config, name, value)

    return runtime_config

