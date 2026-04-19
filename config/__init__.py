"""
Config loader - imports appropriate config based on stage.

Usage:
    from config import load_config

    config = load_config('pretrain')
    print(config.ARENA_SIZE_X)
"""

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


# For backward compatibility - default import (stage 0)
from .base import *
from .debug import *

# Also import stage-specific for direct access
try:
    from .stage0 import ARENA_SIZE_X, ARENA_SIZE_Y, ARENA_HEIGHT, MIN_START_GOAL_DIST
    from .stage1 import STAGE1_N_OBSTACLES, STAGE1_RADIUS, STAGE1_HEIGHT
except ImportError:
    pass
