"""
Configuration file for RL drone navigation project.
All constants and hyperparameters in one place.
"""

# Environment parameters
ARENA_SIZE_X = 15.0
ARENA_SIZE_Y = 15.0
ARENA_HEIGHT = 5.0
MIN_START_GOAL_DIST = 8.0
MIN_CLEARANCE = 1.0
SUCCESS_DIST = 1.0  # Увеличено с 0.5 для более легкого попадания
MAX_STEPS = 750

# Control parameters (velocity control)
VX_MAX = 1.5
VY_MAX = 1.5
VZ_MAX = 0.8
YAW_RATE_MAX = 0.8

# Raycasts
N_RAYS = 16
RAY_LENGTH = 5.0  # Откат к 5.0

# Reward parameters
REWARD_SUCCESS = 500.0
REWARD_CRASH = -100.0
REWARD_PROGRESS_SCALE = 6.0
REWARD_PROXIMITY_THRESHOLD = 1.0  # Компромисс: не 0.5, не 2.0
REWARD_PROXIMITY_SCALE = 12.0  # Компромисс: не 8.0, не 20.0
REWARD_STEP_PENALTY = 0.002
REWARD_EXPLORATION_BONUS = 0.5  # Награда за посещение новой клетки
EXPLORATION_GRID_SIZE = 1.0  # Размер клетки сетки в метрах

# Curriculum parameters
CURRICULUM_WINDOW = 200
THRESHOLDS = {0: 0.90, 1: 0.75}

# Stage 1 parameters (static obstacles)
STAGE1_N_OBSTACLES = (8, 12)  # Умеренное количество
STAGE1_RADIUS = (0.25, 0.50)  # Средняя толщина
STAGE1_HEIGHT = (2.5, 5.0)

# Stage 2 parameters (dynamic obstacles)
STAGE2_N_DYNAMIC = (3, 6)
STAGE2_DYN_AMPLITUDE = (0.5, 1.5)
STAGE2_DYN_FREQUENCY = (0.3, 1.0)

# Training parameters
SEED = 42
N_ENVS = 4  # Вернул к 4 (было 8)
TOTAL_STEPS = 600_000
EVAL_FREQ = 20_000
SAVE_FREQ = 50_000

# PPO hyperparameters (fixed)
PPO_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 128,
    "n_epochs": 5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.15,
    "ent_coef": 0.005,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy": "MlpPolicy",
}
