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
SUCCESS_DIST = 0.4  # Реальная цель - чуть мягче чем waypoints (0.3м)
WAYPOINT_THRESHOLD = 0.3  # Радиус достижения waypoint (строгий)

# Simulation frequencies
PYB_FREQ = 240  # Physics simulation frequency (Hz)
CTRL_FREQ = 30  # Control frequency (Hz) - how often agent makes decisions
MAX_STEPS = 1200  # Episode length in control steps (1200/30 = 40 seconds)

# Control parameters (velocity control)
VX_MAX = 1.5
VY_MAX = 1.5
VZ_MAX = 0.8
YAW_RATE_MAX = 0.8

# Raycasts
N_RAYS = 16
RAY_LENGTH = 5.0  # Откат к 5.0

# Reward parameters
REWARD_SUCCESS = 500
REWARD_CRASH = -500.0  # Сбалансировано с success (было -100)
REWARD_WAYPOINT = 50.0  # Уменьшено с 150: соотношение 50/500=10% (разумнее чем 30%)
REWARD_PROGRESS_SCALE = 10.0 # награда за сближение
REWARD_VELOCITY_SCALE = 4.0  # Увеличено с 2.0 - сильнее награждаем за полет к цели
REWARD_HEADING_SCALE = 5.0  # Увеличено с 2.0 - сильнее награждаем за правильное направление
REWARD_ACTION_SMOOTHNESS_SCALE = 0.5  # Штраф за резкие изменения действий
REWARD_YAW_PENALTY_SCALE = 0.5  # Штраф за избыточное вращение
REWARD_PROXIMITY_THRESHOLD = 3.0  # Расширенная зона для раннего притяжения
REWARD_PROXIMITY_SCALE = 5.0  # Уменьшено с 20.0 - используется в proximity reward
REWARD_STEP_PENALTY = 0.0015
REWARD_EXPLORATION_BONUS = 0.0  # Награда за посещение новой клетки (отключено)
REWARD_EFFICIENCY_BONUS = True  # Бонус за быстрое достижение цели
REWARD_EFFICIENCY_SCALE = 200.0  # Масштаб бонуса за эффективность
EXPLORATION_GRID_SIZE = 1.0  # Размер клетки сетки в метрах

# Stage 1 parameters (static obstacles)
STAGE1_N_OBSTACLES = (8, 12)  # Умеренное количество
STAGE1_RADIUS = (0.25, 0.50)  # Средняя толщина
STAGE1_HEIGHT = (2.5, 5.0)

# Training parameters
SEED = 42
N_ENVS = 8  # Увеличено с 4 для ускорения обучения (DRL-DroneNavigation использует 12)
TOTAL_STEPS = 600_000
EVAL_FREQ = 20_000
SAVE_FREQ = 50_000

# PPO hyperparameters (fixed)
PPO_PARAMS = {
    "learning_rate": 1e-4,        # Уменьшено с 3e-4 для стабильности
    "n_steps": 2048,
    "batch_size": 256,        # Вернули с 512
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,        # Вернули с 0.15
    "ent_coef": 0.01,         # Вернули с 0.0005 - критично для exploration!
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy": "MlpPolicy",
    "use_sde": True,            # State Dependent Exploration
    "sde_sample_freq": 8,       # Loquercio 2021: SDE лучше для UAV control
}

# ─────────────────────────────────────────────
# DEBUG & MONITORING PARAMETERS
# ─────────────────────────────────────────────

DEBUG_MODE = True  # По умолчанию выключен для production (можно включить через --debug в train.py)

# ─── Декомпозиция reward ─────────────────────
# Видеть из чего складывается суммарная награда
LOG_REWARD_COMPONENTS = True
# Выводит отдельно:
#   reward_progress    — сколько заработал на прогрессе
#   reward_velocity    — сколько на velocity к цели
#   reward_proximity   — сколько на близости к цели
#   reward_obstacle    — сколько потерял на штрафах за препятствия
#   reward_step_pen    — сколько потерял на step penalty
#   reward_exploration — сколько на exploration
#   reward_terminal    — финальный бонус (success/crash/timeout)

# ─── Навигационные метрики ───────────────────
LOG_NAVIGATION_METRICS = True
#   path_efficiency    — длина_прямой / длина_пути [0..1]
#                        1.0 = летел прямо
#                        0.3 = петлял сильно
#   avg_speed          — средняя скорость м/с
#   avg_heading_error  — средний угол к цели (градусы)
#                        <15° = хорошо, >45° = проблема
#   time_facing_goal   — % времени летел в сторону цели

# ─── Метрики следования RRT* пути ────────────
LOG_PATH_FOLLOWING_METRICS = True
#   avg_cross_track_error  — среднее отклонение от пути (метры)
#                            <0.3м = хорошо следует
#                            >1.0м = плохо следует
#   max_cross_track_error  — максимальное отклонение
#   path_following_score   — % времени когда отклонение <0.5м
#                            >80% = хорошо
CROSS_TRACK_ERROR_THRESHOLD = 0.5  # Порог для path_following_score (метры)

# ─── Метрики эпизода ─────────────────────────
LOG_EPISODE_METRICS = True
#   start_distance     — начальная дист. до цели
#   final_distance     — конечная дист. до цели (при timeout)
#   min_distance       — минимальная дист. за эпизод
#                        если min=0.5 но timeout → цель была рядом!
#   closest_obstacle   — минимальное расстояние до препятствия
#   n_near_misses      — сколько раз был < MIN_CLEARANCE от препятствия

# ─── Расширенные метрики эпизода ─────────────
LOG_EXTENDED_EPISODE_METRICS = True
#   avg_clearance      — средняя дистанция до ближайшего препятствия
#   hovering_time      — % времени с низкой скоростью (<0.2 м/с)
#   goal_seeking_ratio — % времени когда летел в сторону цели (угол <90°)
#   spinning_time      — % времени с высоким yaw_rate (>0.5)
HOVERING_SPEED_THRESHOLD = 0.2  # м/с - порог для определения зависания
SPINNING_YAW_THRESHOLD = 0.5    # рад/с - порог для определения вращения

# ─── PPO внутренние метрики ──────────────────
LOG_PPO_INTERNALS = True
#   policy_entropy     — энтропия политики
#                        высокая → агент не уверен
#                        низкая → агент детерминирован
#   value_loss         — ошибка critic
#   policy_loss        — ошибка actor
#   approx_kl          — насколько изменилась политика
#                        >0.02 → слишком большой шаг
#   clip_fraction      — % действий обрезанных clip_range
#                        >0.3 → clip_range слишком мал
#   explained_variance — насколько critic предсказывает return
#                        <0.5 → critic плохо обучен
#                        >0.9 → хорошо

# ─── Анализ timeout эпизодов ─────────────────
LOG_TIMEOUT_ANALYSIS = True
#   timeout_avg_final_dist  — средняя дист. до цели при timeout
#                             ~1-2м → почти долетел, не хватило шагов
#                             ~8-10м → вообще не летел к цели
#   timeout_avg_min_dist    — минимальная дист. за timeout эпизод
#   timeout_start_dist      — начальные дистанции timeout эпизодов
#                             коррелирует ли timeout с дальним стартом?
#   timeout_near_goal       — % timeout когда был близко к цели (<2м)
#   timeout_stuck           — % timeout когда застрял (мало движения)
TIMEOUT_NEAR_GOAL_THRESHOLD = 2.0  # метры - порог "близко к цели"
TIMEOUT_STUCK_MOVEMENT_THRESHOLD = 5.0  # метры - минимальное пройденное расстояние

# ─── Анализ действий агента ──────────────────
LOG_ACTION_STATS = True
#   action_mean        — средние значения vx, vy, vz, yaw
#                        если vx_mean ≈ 0 → агент не движется
#   action_std         — разброс действий
#                        низкий std → детерминированная политика
#   action_smoothness  — среднее |a_t - a_{t-1}|
#                        высокое → дёрганые движения

# ─── Частота логирования ─────────────────────
LOG_INTERVAL_EPISODES = 10      # каждые N эпизодов
LOG_INTERVAL_STEPS = 5000       # или каждые N шагов
LOG_TENSORBOARD = True          # писать в TensorBoard
LOG_CSV = True                  # писать в CSV для анализа
LOG_CSV_PATH = "logs/debug_metrics.csv"


