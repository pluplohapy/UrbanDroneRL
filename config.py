"""
Legacy config.py - for backward compatibility.
Imports from new config/ structure.

New code should use:
    from config import load_config
    config = load_config('pretrain')
"""

# Import everything from base config for backward compatibility
from config.base import *
from config.debug import *

# Import stage-specific parameters
from config.stage0 import (
    ARENA_SIZE_X, ARENA_SIZE_Y, ARENA_HEIGHT, MIN_START_GOAL_DIST
)

from config.stage1 import (
    STAGE1_N_OBSTACLES, STAGE1_RADIUS, STAGE1_HEIGHT
)

# For new code, use load_config
from config import load_config

__all__ = [
    'load_config',
    # Base
    'PYB_FREQ', 'CTRL_FREQ', 'MAX_STEPS',
    'VX_MAX', 'VY_MAX', 'VZ_MAX', 'YAW_RATE_MAX',
    'N_RAYS', 'RAY_LENGTH',
    'SEED', 'N_ENVS',
    'PPO_PARAMS',
    'REWARD_SUCCESS', 'REWARD_CRASH', 'REWARD_WAYPOINT',
    'REWARD_PROGRESS_SCALE', 'REWARD_VELOCITY_SCALE',
    'REWARD_HEADING_SCALE', 'REWARD_ACTION_SMOOTHNESS_SCALE',
    'REWARD_YAW_PENALTY_SCALE', 'REWARD_PROXIMITY_THRESHOLD',
    'REWARD_PROXIMITY_SCALE', 'REWARD_STEP_PENALTY',
    'REWARD_EXPLORATION_BONUS', 'REWARD_EFFICIENCY_BONUS',
    'REWARD_EFFICIENCY_SCALE', 'EXPLORATION_GRID_SIZE',
    'MIN_CLEARANCE', 'SUCCESS_DIST', 'WAYPOINT_THRESHOLD',
    # Stage specific
    'ARENA_SIZE_X', 'ARENA_SIZE_Y', 'ARENA_HEIGHT', 'MIN_START_GOAL_DIST',
    'STAGE1_N_OBSTACLES', 'STAGE1_RADIUS', 'STAGE1_HEIGHT',
    # Debug
    'DEBUG_MODE', 'LOG_REWARD_COMPONENTS', 'LOG_NAVIGATION_METRICS',
    'LOG_PATH_FOLLOWING_METRICS', 'LOG_EPISODE_METRICS',
    'LOG_EXTENDED_EPISODE_METRICS', 'LOG_TIMEOUT_ANALYSIS',
    'LOG_ACTION_STATS', 'LOG_PPO_INTERNALS',
    'LOG_INTERVAL_EPISODES', 'LOG_INTERVAL_STEPS',
    'LOG_TENSORBOARD', 'LOG_CSV', 'LOG_CSV_PATH',
    'CROSS_TRACK_ERROR_THRESHOLD', 'HOVERING_SPEED_THRESHOLD',
    'SPINNING_YAW_THRESHOLD', 'TIMEOUT_NEAR_GOAL_THRESHOLD',
    'TIMEOUT_STUCK_MOVEMENT_THRESHOLD'
]
