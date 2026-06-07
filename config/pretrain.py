from .base import *

ARENA_SIZE_X = 4.0
ARENA_SIZE_Y = 12.0
ARENA_HEIGHT = 3.0
MIN_START_GOAL_DIST = 6.0

PHYSICAL_ARENA_WALLS = True
PHYSICAL_ARENA_WALLS_INCLUDE_ENDS = True
PHYSICAL_ARENA_WALL_THICKNESS = 0.08
PHYSICAL_ARENA_WALL_HEIGHT = None
PHYSICAL_ARENA_WALL_ALPHA = 0.10
PHYSICAL_ARENA_CEILING = False
USE_ENHANCED_OBS = True

START_ZONE_X = (-1.2, 1.2)
GOAL_ZONE_X = (-0.85, 0.85)
START_ZONE_Y = (-5.2, -4.2)
GOAL_ZONE_Y = (3.8, 4.8)
START_ZONE_Z = (0.8, 1.5)
GOAL_ZONE_Z = (0.95, 1.4)
GOAL_WALL_CLEARANCE = 1.05
GOAL_CORNER_BIAS = 0.0
GOAL_CORNER_X_BANDS = ((-0.95, -0.75), (0.75, 0.95))
GOAL_END_Y_BIAS = 0.35
GOAL_END_Y_EDGE_FRACTION = 0.45
PRETRAIN_BIDIRECTIONAL_GOALS = False

USE_PLANNER = False

VX_MAX = 2.4
VY_MAX = 2.4
VZ_MAX = 1.6
YAW_RATE_MAX = 1.2
MAX_COMMAND_SPEED = 0.0
VELOCITY_TARGET_LOOKAHEAD_STEPS = 3.0

PPO_PARAMS = {
    **PPO_PARAMS,
    "learning_rate": 5e-5,
    "target_kl": 0.02,
}

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

REWARD_SUCCESS = 3000.0
REWARD_EFFICIENCY_SCALE = 450.0
REWARD_OUT_OF_BOUNDS_EXTRA = -1100.0
REWARD_COLLISION_EXTRA = -650.0
REWARD_TIMEOUT = -500.0

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

REWARD_OBSTACLE_APPROACH_SCALE = 10.0
REWARD_OBSTACLE_HARD_THRESHOLD = 1.05
REWARD_OBSTACLE_HARD_SCALE = 14.0

REWARD_YAW_PENALTY_SCALE = 10.0
REWARD_ACTION_SMOOTHNESS_SCALE = 0.7

CYLINDER_WALL_MARGIN = 0.40
CYLINDER_PAIR_CLEARANCE = 0.50
CYLINDER_PATH_CLEARANCE = 0.50
CYLINDER_PATH_GRID_RESOLUTION = 0.15

DYNAMIC_START_GOAL_CLEARANCE = 1.1
BEAM_PAIR_Y_CLEARANCE = 0.9
BEAM_BOTTOM_GAP = 0.85
BEAM_CEILING_GAP = 0.25
STICK_PAIR_Y_CLEARANCE = 0.65
STICK_BOTTOM_GAP_FRACTION = 0.22
STICK_CEILING_GAP = 0.85
STICK_SIDE_MARGIN = 0.06

DYNAMIC_MIX_WEIGHTS = {
    'spheres': 0.25,
    'crossing_spheres': 0.25,
    'beams': 0.25,
    'swinging_sticks': 0.25,
}

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
        'count': (5, 6),
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

    'city_dynamic': {
        'name': 'Динамический мини-город',
        'count': (7, 9),
        'avenue_width': (1.8, 2.6),
        'building_depth_y': (1.15, 2.05),
        'height': (2.0, 4.8),
        'center_shift': 0.65,
        'cross_streets': (1, 2),
        'cross_street_width': 2.2,
        'lamp_radius': (0.035, 0.065),
        'lamp_height': (2.0, 3.1),
        'wire_count': (4, 7),
        'wire_length': (3.0, 5.8),
        'wire_height': (3.0, 4.4),
        'wire_thickness': (0.025, 0.055),
        'vehicle_count': (3, 5),
        'vehicle_width': 0.48,
        'vehicle_length': 0.95,
        'vehicle_height': 0.38,
        'vehicle_speed': (0.35, 0.85),
        'vehicle_amplitude': (1.8, 3.6),
        'vehicle_frequency': (0.04, 0.11),
        'bird_count': (4, 7),
        'bird_radius': (0.14, 0.22),
        'bird_speed': (0.35, 0.75),
        'bird_amplitude': (0.7, 1.45),
        'bird_frequency': (0.05, 0.14),
        'bird_height': (2.7, 4.15),
        'bird_roof_clearance': 0.28,
        'spawn_clearance': 0.45,
        'spawn_validation_attempts': 40,
        'dynamic': True,

        'requires_explicit_obstacle_type': True,
        'config_overrides': {
            'ARENA_SIZE_X': 8.0,
            'ARENA_SIZE_Y': 18.0,
            'ARENA_HEIGHT': 5.0,
            'MIN_START_GOAL_DIST': 10.0,
            'START_ZONE_X': (-1.2, 1.2),
            'GOAL_ZONE_X': (-1.4, 1.4),
            'START_ZONE_Y': (-8.1, -7.2),
            'GOAL_ZONE_Y': (7.0, 8.1),
            'START_ZONE_Z': (1.0, 2.2),
            'GOAL_ZONE_Z': (1.2, 2.8),
            'GOAL_WALL_CLEARANCE': 1.35,
            'RAY_LENGTH': 7.0,
            'MAX_STEPS': 1800,
            'PHYSICAL_ARENA_WALLS': True,
            'PHYSICAL_ARENA_WALLS_INCLUDE_ENDS': True,
            'PHYSICAL_ARENA_WALL_HEIGHT': None,
        }
    },

    'construction_site_dynamic': {
        'name': 'Динамическая стройплощадка',
        'count': (6, 8),
        'lane_width': (2.2, 3.0),
        'frame_depth_y': (1.25, 2.05),
        'frame_height': (2.8, 5.3),
        'floor_levels': (1, 2),
        'floor_slab_thickness': (0.12, 0.18),
        'column_radius': (0.055, 0.085),
        'scaffold_radius': (0.035, 0.055),
        'scaffold_height': (2.3, 4.8),
        'scaffold_beam_thickness': (0.045, 0.075),
        'barrier_height': (0.45, 0.75),
        'barrier_depth': (0.10, 0.16),
        'crane_count': (1, 2),
        'crane_mast_radius': (0.08, 0.12),
        'crane_height': (4.8, 5.6),
        'crane_arm_length': (3.4, 5.2),
        'crane_arm_thickness': (0.08, 0.12),
        'crane_swing_angle': 55,
        'crane_swing_period': (13.0, 20.0),
        'load_count': (1, 2),
        'load_size': (0.42, 0.62),
        'load_height': (0.42, 0.70),
        'load_swing_amplitude': (0.65, 1.35),
        'load_frequency': (0.045, 0.085),
        'vehicle_count': (3, 5),
        'vehicle_width': 0.55,
        'vehicle_length': 1.20,
        'vehicle_height': 0.42,
        'vehicle_speed': (0.35, 0.85),
        'vehicle_amplitude': (2.0, 4.4),
        'vehicle_frequency': (0.035, 0.09),
        'lift_count': (2, 3),
        'lift_size': (0.65, 0.95),
        'lift_height': 0.16,
        'lift_amplitude': (1.0, 2.1),
        'lift_frequency': (0.035, 0.075),
        'swinging_pipe_count': (2, 4),
        'swinging_pipe_length': (1.8, 2.8),
        'swinging_pipe_thickness': (0.07, 0.11),
        'swinging_pipe_angle': 22,
        'swinging_pipe_period': (6.0, 10.0),
        'debris_count': (4, 7),
        'debris_radius': (0.12, 0.20),
        'debris_amplitude': (0.55, 1.20),
        'debris_frequency': (0.05, 0.12),
        'debris_speed': (0.25, 0.65),
        'debris_height': (2.6, 4.8),
        'spawn_clearance': 0.55,
        'spawn_validation_attempts': 60,
        'dynamic': True,
        'requires_explicit_obstacle_type': True,
        'config_overrides': {
            'ARENA_SIZE_X': 10.0,
            'ARENA_SIZE_Y': 22.0,
            'ARENA_HEIGHT': 6.0,
            'MIN_START_GOAL_DIST': 13.0,
            'START_ZONE_X': (-1.3, 1.3),
            'GOAL_ZONE_X': (-1.5, 1.5),
            'START_ZONE_Y': (-10.0, -8.8),
            'GOAL_ZONE_Y': (8.8, 10.0),
            'START_ZONE_Z': (1.1, 2.2),
            'GOAL_ZONE_Z': (1.2, 2.8),
            'GOAL_WALL_CLEARANCE': 1.45,
            'RAY_LENGTH': 8.0,
            'MAX_STEPS': 2200,
            'PHYSICAL_ARENA_WALLS': True,
            'PHYSICAL_ARENA_WALLS_INCLUDE_ENDS': True,
            'PHYSICAL_ARENA_WALL_HEIGHT': None,
        }
    },

    'boxes': {
        'name': 'Кубы/блоки (прямоугольные столбы)',
        'count': (4, 8),
        'size': (0.4, 1.0),
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

SPHERE_MOVEMENT_AMPLITUDE = (0.35, 0.85)
SPHERE_MOVEMENT_FREQUENCY = (0.12, 0.28)
BEAM_SWING_SPEED = 0.5
STICK_SWING_SPEED = 0.8
