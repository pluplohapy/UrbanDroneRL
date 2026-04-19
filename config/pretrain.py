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

# ===== OBSTACLE TYPES CONFIGURATION =====
OBSTACLE_TYPES = {
    'cylinders': {
        'name': 'Цилиндры (столбы)',
        'count': (2, 4),  # x2 количество
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
        'name': 'Горизонтальные балки',
        'count': (2, 3),
        'length': (1.5, 3.0),
        'height': (1.2, 2.5),
        'thickness': 0.2,
        'dynamic': True,
        'swing_angle': 15,
        'swing_period': (3.0, 5.0)
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
        'count': (9, 18),  # x3 от предыдущего (было 3-6)
        'length': (4.0, 4.0),  # Полная ширина коридора (4м)
        'thickness': 0.12,  # Оставили как было
        'swing_angle': 25,  # Движение вверх-вниз
        'swing_period': (5.0, 9.0),  # x2 медленнее (было 2.5-4.5)
        'dynamic': True,
        'vertical_swing': True,  # Новый параметр для вертикального движения
        'vertical_amplitude': (1.4, 1.4)  # Амплитуда движения от низа до верха (±1.4м от центра)
    }
}

# ===== DYNAMIC OBSTACLES PARAMETERS =====
SPHERE_MOVEMENT_AMPLITUDE = (0.5, 1.2)  # Smaller amplitude for narrow corridor
SPHERE_MOVEMENT_FREQUENCY = (0.3, 0.8)
BEAM_SWING_SPEED = 0.5
STICK_SWING_SPEED = 0.8
