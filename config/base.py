"""
Base configuration - common parameters for all stages.
"""

# ===== SIMULATION PARAMETERS =====
PYB_FREQ = 240  # Physics simulation frequency (Hz)
CTRL_FREQ = 30  # Control frequency (Hz) - how often agent makes decisions
MAX_STEPS = 1200  # Episode length in control steps (1200/30 = 40 seconds)

# ===== DRONE CONTROL PARAMETERS =====
VX_MAX = 3.0  # Was 1.5
VY_MAX = 3.0  # Was 1.5
VZ_MAX = 1.6  # Was 0.8
YAW_RATE_MAX = 0.8

# ===== RAYCASTS =====
N_RAYS = 20  # Was 16
RAY_LENGTH = 5.0

# ===== TRAINING PARAMETERS =====
SEED = 42
N_ENVS = 8  # Number of parallel environments

# ===== PPO HYPERPARAMETERS =====
PPO_PARAMS = {
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy": "MlpPolicy",
    "use_sde": True,
    "sde_sample_freq": 8,
}

# ===== REWARD PARAMETERS (COMMON) =====
REWARD_SUCCESS = 500
REWARD_CRASH = -500.0
REWARD_WAYPOINT = 50.0
REWARD_PROGRESS_SCALE = 10.0
REWARD_VELOCITY_SCALE = 8.0
REWARD_HEADING_SCALE = 10.0
REWARD_ACTION_SMOOTHNESS_SCALE = 0.5
REWARD_YAW_PENALTY_SCALE = 15.0
REWARD_PROXIMITY_THRESHOLD = 3.0
REWARD_PROXIMITY_SCALE = 5.0
REWARD_STEP_PENALTY = 0.0015
REWARD_EXPLORATION_BONUS = 0.0
REWARD_EFFICIENCY_BONUS = True
REWARD_EFFICIENCY_SCALE = 200.0
EXPLORATION_GRID_SIZE = 1.0

# ===== COMMON PARAMETERS =====
MIN_CLEARANCE = 1.0
SUCCESS_DIST = 0.4
WAYPOINT_THRESHOLD = 0.3
