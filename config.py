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
SUCCESS_DIST = 0.5
MAX_STEPS = 750  # Увеличено с 500 для большего времени на достижение цели

# Control parameters (velocity control)
VX_MAX = 1.5
VY_MAX = 1.5
VZ_MAX = 0.8
YAW_RATE_MAX = 0.8

# Raycasts
N_RAYS = 16
RAY_LENGTH = 5.0

# Reward parameters
REWARD_SUCCESS = 100.0
REWARD_CRASH = -100.0
REWARD_PROGRESS_SCALE = 6.0  # Увеличено с 3.0 для мотивации быстрого полёта
REWARD_PROXIMITY_THRESHOLD = 0.4
REWARD_PROXIMITY_SCALE = 1.2
REWARD_STEP_PENALTY = 0.002  # Уменьшено с 0.008 чтобы дрон не боялся лететь

# Curriculum parameters
CURRICULUM_WINDOW = 200
THRESHOLDS = {0: 0.90, 1: 0.75}

# Stage 1 parameters (static obstacles)
STAGE1_N_OBSTACLES = (4, 10)
STAGE1_RADIUS = (0.15, 0.40)
STAGE1_HEIGHT = (2.5, 5.0)

# Stage 2 parameters (dynamic obstacles)
STAGE2_N_DYNAMIC = (3, 6)
STAGE2_DYN_AMPLITUDE = (0.5, 1.5)
STAGE2_DYN_FREQUENCY = (0.3, 1.0)

# Training parameters
SEED = 42
N_ENVS = 4
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
