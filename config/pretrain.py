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

# Make corridor borders real PyBullet obstacles so raycasts can see them.
PHYSICAL_ARENA_WALLS = True
PHYSICAL_ARENA_WALLS_INCLUDE_ENDS = True
PHYSICAL_ARENA_WALL_THICKNESS = 0.08
PHYSICAL_ARENA_WALL_HEIGHT = None
PHYSICAL_ARENA_WALL_ALPHA = 0.10
PHYSICAL_ARENA_CEILING = False
USE_ENHANCED_OBS = True

# ===== START/GOAL ZONES =====
# Both start and goal INSIDE the corridor (within 12m length)
# Drone flies through the narrow corridor avoiding obstacles
START_ZONE_X = (-1.2, 1.2)   # Start may be wider; terminal goals stay easier to capture.
GOAL_ZONE_X = (-0.85, 0.85)  # Keep default goals away from side-wall penalty zones.
START_ZONE_Y = (-5.2, -4.2)  # Start near one end with room before the wall
GOAL_ZONE_Y = (3.8, 4.8)     # Goal near opposite end with real braking room before wall
START_ZONE_Z = (0.8, 1.5)    # Lower height
GOAL_ZONE_Z = (0.95, 1.4)    # Narrower terminal altitude band for easier 3D capture.
GOAL_WALL_CLEARANCE = 1.05   # Match hard obstacle threshold: goal is never inside wall-danger zone.
GOAL_CORNER_BIAS = 0.0       # Enable later in curriculum, after center goals are reliable.
GOAL_CORNER_X_BANDS = ((-0.95, -0.75), (0.75, 0.95))
GOAL_END_Y_BIAS = 0.35
GOAL_END_Y_EDGE_FRACTION = 0.45
PRETRAIN_BIDIRECTIONAL_GOALS = False  # Fast-repro mode: learn the original -Y -> +Y corridor task.

# ===== NO PLANNER FOR PRETRAIN =====
USE_PLANNER = False

# ===== SIM DRONE CONTROL =====
# Moderate speed-up from the original training caps. Full CF2X unlock
# (8.33 m/s per axis) made scratch learning collapse into early crashes, so
# this run uses 2x the old stable caps: 1.2/1.2/0.8 -> 2.4/2.4/1.6 m/s.
VX_MAX = 2.4
VY_MAX = 2.4
VZ_MAX = 1.6
YAW_RATE_MAX = 1.2
MAX_COMMAND_SPEED = 0.0
VELOCITY_TARGET_LOOKAHEAD_STEPS = 3.0

# ===== PPO STABILITY TUNING (PRETRAIN ONLY) =====
# Copy base params and override only pretrain-specific settings.
PPO_PARAMS = {
    **PPO_PARAMS,
    "learning_rate": 5e-5,
    "target_kl": 0.02,
}

# ===== TERMINAL APPROACH =====
# Precise hit succeeds immediately; a slow stable capture in a slightly larger
# radius also succeeds so near-goal behavior is learnable instead of brittle.
SUCCESS_DIST = 0.55
SUCCESS_HOLD_RADIUS = 0.85
SUCCESS_HOLD_STEPS = 2
SUCCESS_HOLD_MAX_SPEED = 0.35
GOAL_VEL_SOFT_RADIUS = 1.8
GOAL_VEL_MIN_SCALE = 0.18
GOAL_AWAY_VEL_SOFT_RADIUS = 1.5
GOAL_AWAY_VEL_MIN_SCALE = 0.05
BOUNDARY_VEL_SOFT_MARGIN = 0.0
BOUNDARY_VEL_MIN_SCALE = 1.0

# Make reaching the goal clearly better than early contact/OOB failures.
# With dense static maps, otherwise a long but successful episode can still
# have worse return than a short crash.
REWARD_SUCCESS = 3000.0
REWARD_EFFICIENCY_SCALE = 450.0
REWARD_OUT_OF_BOUNDS_EXTRA = -1100.0
REWARD_COLLISION_EXTRA = -650.0
REWARD_TIMEOUT = -500.0

# Match the fast enhanced-observation run: keep explicit boundary pressure in
# addition to physical walls/raycasts so the policy learns the corridor limits.
REWARD_BOUNDARY_THRESHOLD = 1.4
REWARD_BOUNDARY_SCALE = 11.0
REWARD_BOUNDARY_OUTWARD_SCALE = 12.0
REWARD_NEAR_GOAL_BOUNDARY_THRESHOLD = 0.55
REWARD_NEAR_GOAL_BOUNDARY_SCALE = 7.0
REWARD_NEAR_GOAL_RADIUS = 2.0
REWARD_NEAR_GOAL_PROGRESS_EPS = 0.01
REWARD_NEAR_GOAL_PROGRESS_SCALE = 40.0
REWARD_NEAR_GOAL_STALL_SCALE = 220.0
REWARD_NEAR_GOAL_AWAY_SCALE = 28.0
REWARD_NEAR_GOAL_SPEED_SCALE = 7.0
REWARD_NEAR_GOAL_SPEED_TARGET = 0.25
REWARD_NEAR_GOAL_CAPTURE_RADIUS = 0.90
REWARD_NEAR_GOAL_CAPTURE_SPEED_TARGET = 0.35
REWARD_NEAR_GOAL_CAPTURE_SCALE = 2.8
REWARD_TIMEOUT_NEAR_GOAL_RADIUS = 1.50
REWARD_TIMEOUT_NEAR_GOAL_SCALE = 1000.0
REWARD_TIMEOUT_REGRESSION_RADIUS = 1.50
REWARD_TIMEOUT_REGRESSION_SCALE = 700.0

# Contacts were mostly early and costly in the logs, especially on beams and
# spheres. Make dangerous closing trajectories more visible before impact.
REWARD_OBSTACLE_APPROACH_SCALE = 10.0
REWARD_OBSTACLE_HARD_THRESHOLD = 1.05
REWARD_OBSTACLE_HARD_SCALE = 14.0

# Mildly stronger regularization against spinning/jerk in dense maps.
REWARD_YAW_PENALTY_SCALE = 10.0
REWARD_ACTION_SMOOTHNESS_SCALE = 0.7

# ===== CYLINDER MAP QUALITY (ANTI-DEAD-END) =====
# Keep cylinders away from borders and from each other, and require a
# feasible XY corridor from start to goal during sampling.
CYLINDER_WALL_MARGIN = 0.40
CYLINDER_PAIR_CLEARANCE = 0.50
CYLINDER_PATH_CLEARANCE = 0.50
CYLINDER_PATH_GRID_RESOLUTION = 0.15

# ===== DYNAMIC MAP QUALITY =====
# Dynamic obstacles should teach timing and vertical avoidance, not create
# impossible full-corridor gates. These constraints keep spawn/goal areas clear
# and preserve a practical low passage under moving bars.
DYNAMIC_START_GOAL_CLEARANCE = 1.1
BEAM_PAIR_Y_CLEARANCE = 0.9
BEAM_BOTTOM_GAP = 0.85
BEAM_CEILING_GAP = 0.25
STICK_PAIR_Y_CLEARANCE = 0.65
STICK_BOTTOM_GAP_FRACTION = 0.22
STICK_CEILING_GAP = 0.85
STICK_SIDE_MARGIN = 0.06

# Dynamic mix curriculum weights. Spheres are frequent enough to train moving
# obstacle timing, while beams/sticks still cover vertical avoidance.
DYNAMIC_MIX_WEIGHTS = {
    'spheres': 0.25,
    'crossing_spheres': 0.25,
    'beams': 0.25,
    'swinging_sticks': 0.25,
}

# ===== OBSTACLE TYPES CONFIGURATION =====
OBSTACLE_TYPES = {
    'empty': {
        'name': 'Пустая карта',
        'count': (0, 0),
        'dynamic': False
    },

    'cylinders': {
        'name': 'Цилиндры (столбы)',
        'count': (3, 5),
        'radius': (0.28, 0.58),
        'height': (2.0, 2.8),
        'dynamic': False
    },

    'spheres': {
        'name': 'Сферы (птицы)',
        'count': (4, 6),
        'radius': (0.22, 0.34),
        'speed': (0.35, 0.65),
        'dynamic': True,
        'movement': 'sinusoidal'
    },

    'crossing_spheres': {
        'name': 'Пересекающие сферы',
        'count': (4, 6),
        'radius': (0.22, 0.32),
        'speed': (0.40, 0.70),
        'amplitude': (0.75, 1.20),
        'frequency': (0.12, 0.25),
        'dynamic': True,
        'movement': 'crossing'
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
        'length': (1.4, 2.9),
        'height': (1.35, 2.15),
        'thickness': 0.20,
        'dynamic': True,
        'swing_angle': 35,
        # Slower by 1.5x than the previous 3.0-5.0s range.
        'swing_period': (4.5, 7.5),
        'swing_axis': 'pitch'
    },

    'gates': {
        'name': 'Ворота с проходами',
        'count': (3, 5),
        'gap_width': (1.15, 1.55),
        'gap_center': (-0.45, 0.45),
        'height': (1.8, 2.6),
        'thickness': 0.16,
        'dynamic': False
    },

    'slalom': {
        'name': 'Слалом между блоками',
        'count': (6, 8),
        'size': (0.45, 0.70),
        'height': (1.6, 2.5),
        'lateral_offset': (0.65, 1.25),
        'dynamic': False
    },

    'city_blocks': {
        'name': 'Мини-город с улицей',
        'count': (6, 8),
        'avenue_width': (1.05, 1.45),
        'block_size': (0.45, 0.75),
        'height': (1.4, 2.7),
        'gate_gap_width': (1.10, 1.45),
        'center_shift': 0.45,
        'dynamic': False
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
        'length': (3.55, 3.85),
        'thickness': 0.12,
        'swing_angle': 25,
        'swing_period': (5.0, 9.0),
        'dynamic': True,
        'vertical_swing': True,
        'vertical_amplitude': (0.22, 0.50)
    }
}

# ===== DYNAMIC OBSTACLES PARAMETERS =====
SPHERE_MOVEMENT_AMPLITUDE = (0.35, 0.85)
SPHERE_MOVEMENT_FREQUENCY = (0.12, 0.28)
BEAM_SWING_SPEED = 0.5
STICK_SWING_SPEED = 0.8
