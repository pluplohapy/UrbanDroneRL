"""
Stage Pretrain configuration - diverse obstacles for curriculum learning.
No RRT* planner - pure RL training.
"""

from .base import *

# ===== ARENA PARAMETERS =====
# Narrow corridor arena for intensive obstacle avoidance training
ARENA_SIZE_X = 4.0   # Narrow width (corridor-like)
ARENA_SIZE_Y = 12.0  # Long length
ARENA_HEIGHT = 3.0   # Low ceiling
MIN_START_GOAL_DIST = 6.0  # Long distance along corridor

# ===== START/GOAL ZONES =====
# Both start and goal INSIDE the corridor (within 12m length)
# Drone flies through the narrow corridor avoiding obstacles
START_ZONE_X = (-1.5, 1.5)   # Narrow zone across width
GOAL_ZONE_X = (-1.5, 1.5)    # Narrow zone across width
START_ZONE_Y = (-5.5, -4.5)  # Start near one end (inside corridor)
GOAL_ZONE_Y = (4.5, 5.5)     # Goal near opposite end (inside corridor)
START_ZONE_Z = (0.8, 1.5)    # Lower height
GOAL_ZONE_Z = (0.8, 1.5)     # Lower height

# ===== NO PLANNER FOR PRETRAIN =====
USE_PLANNER = False

# ===== PPO STABILITY TUNING (PRETRAIN ONLY) =====
# Copy base params and override only pretrain-specific settings.
PPO_PARAMS = {
    **PPO_PARAMS,
    "learning_rate": 5e-5,
    "target_kl": 0.02,
}

# ===== TERMINAL APPROACH + BOUNDARY STABILITY =====
# Reduce overshoot near the goal and reduce OOB exits near arena borders.
GOAL_VEL_SOFT_RADIUS = 1.4
GOAL_VEL_MIN_SCALE = 0.2
BOUNDARY_VEL_SOFT_MARGIN = 1.2
BOUNDARY_VEL_MIN_SCALE = 0.0

# Mildly stronger regularization against spinning/jerk in dense maps.
REWARD_YAW_PENALTY_SCALE = 10.0
REWARD_ACTION_SMOOTHNESS_SCALE = 0.7

# ===== CYLINDER MAP QUALITY (ANTI-DEAD-END) =====
# Keep cylinders away from borders and from each other, and require a
# feasible XY corridor from start to goal during sampling.
CYLINDER_WALL_MARGIN = 0.35
CYLINDER_PAIR_CLEARANCE = 0.45
CYLINDER_PATH_CLEARANCE = 0.45
CYLINDER_PATH_GRID_RESOLUTION = 0.20

# ===== DYNAMIC MAP QUALITY =====
# Dynamic obstacles should teach timing and vertical avoidance, not create
# impossible full-corridor gates. These constraints keep spawn/goal areas clear
# and preserve a practical low passage under moving bars.
DYNAMIC_START_GOAL_CLEARANCE = 1.1
BEAM_PAIR_Y_CLEARANCE = 0.9
BEAM_BOTTOM_GAP = 0.45
BEAM_CEILING_GAP = 0.25
STICK_PAIR_Y_CLEARANCE = 0.65
STICK_BOTTOM_GAP_FRACTION = 1.0 / 3.0
STICK_CEILING_GAP = 0.25
STICK_SIDE_MARGIN = 0.15

# ===== OBSTACLE TYPES CONFIGURATION =====
OBSTACLE_TYPES = {
    'empty': {
        'name': 'Пустая карта',
        'count': (0, 0),
        'dynamic': False
    },

    'cylinders': {
        'name': 'Цилиндры (столбы)',
        'count': (4, 5),  # x2 количество
        'radius': (0.4, 0.8),  # x2 ширина
        'height': (2.0, 2.8),
        'dynamic': False
    },

    'spheres': {
        'name': 'Сферы (птицы)',
        'count': (3, 5),  # x1.5 количество (было 2-3)
        'radius': (0.225, 0.375),  # x1.5 размер (было 0.15-0.25)
        'speed': (0.3, 0.8),
        'dynamic': True,
        'movement': 'sinusoidal'
    },

    'walls': {
        'name': 'Вертикальные стены',
        'count': (1, 2),
        'width': (2.0, 3.5),
        'height': (2.0, 2.8),
        'thickness': 0.15,
        'dynamic': False
    },

    'beams': {
        'name': 'Вертикально качающиеся балки',
        'count': (4, 6),
        'length': (1.2, 2.2),
        'height': (1.15, 2.15),
        'thickness': 0.18,
        'dynamic': True,
        'swing_angle': 45,
        'swing_period': (3.0, 5.0),
        'swing_axis': 'pitch'
    },

    'boxes': {
        'name': 'Кубы/блоки (прямоугольные столбы)',
        'count': (4, 8),  # x2 количество
        'size': (0.4, 1.0),  # x2 размер (было 0.2-0.5)
        'height': (1.0, 2.5),
        'dynamic': False
    },

    'swinging_sticks': {
        'name': 'Качающиеся палки (ветки)',
        'count': (5, 8),
        'length': (2.6, 3.7),
        'thickness': 0.12,
        'swing_angle': 25,
        'swing_period': (5.0, 9.0),
        'dynamic': True,
        'vertical_swing': True,
        'vertical_amplitude': (0.25, 0.65)
    }
}

# ===== DYNAMIC OBSTACLES PARAMETERS =====
SPHERE_MOVEMENT_AMPLITUDE = (0.5, 1.2)  # Smaller amplitude for narrow corridor
SPHERE_MOVEMENT_FREQUENCY = (0.3, 0.8)
BEAM_SWING_SPEED = 0.5
STICK_SWING_SPEED = 0.8
