"""
Config loader - imports appropriate config based on stage.

Usage:
    from config import load_config

    config = load_config('pretrain')
    print(config.ARENA_SIZE_X)
"""

from types import SimpleNamespace


def load_config(stage='0'):
    """
    Load configuration for specified stage.

    Args:
        stage: '0', '1', or 'pretrain'

    Returns:
        config module with all parameters
    """
    if stage == '0':
        from . import stage0 as cfg
    elif stage == '1':
        from . import stage1 as cfg
    elif stage == 'pretrain':
        from . import pretrain as cfg
    else:
        raise ValueError(f"Unknown stage: {stage}. Must be '0', '1', or 'pretrain'")

    # Always add debug config
    from . import debug
    for key in dir(debug):
        if not key.startswith('_'):
            setattr(cfg, key, getattr(debug, key))

    return cfg


def apply_pretrain_obstacle_overrides(cfg, obstacle_type):
    """Return a config view with explicit pretrain-map overrides applied."""
    params = getattr(cfg, "OBSTACLE_TYPES", {}).get(obstacle_type, {})
    overrides = dict(params.get("config_overrides", {}))
    if not overrides:
        return cfg

    values = {
        name: getattr(cfg, name)
        for name in dir(cfg)
        if not name.startswith("_")
    }
    values.update(overrides)
    return SimpleNamespace(**values)


# For backward compatibility - default import (stage 0)
from .base import *
from .debug import *

# Also import stage-specific for direct access
try:
    from .stage0 import ARENA_SIZE_X, ARENA_SIZE_Y, ARENA_HEIGHT, MIN_START_GOAL_DIST
    from .stage1 import STAGE1_N_OBSTACLES, STAGE1_RADIUS, STAGE1_HEIGHT
except ImportError:
    pass
