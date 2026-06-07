import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config
from envs.nav_aviary import NavAviary
from envs.raycasts import RaycastSensor
from scenarios.stage_pretrain import StagePretrainScenario
from training.train import make_env_pretrain, sync_env_runtime_config


class _SafeRaycastSensor:
    n_rays = 20
    ray_directions = RaycastSensor().ray_directions

    def cast_rays(self, *args, **kwargs):
        return np.ones(self.n_rays, dtype=np.float32)


def _nav_stub(cfg=None):
    cfg = cfg or load_config("pretrain")
    env = NavAviary.__new__(NavAviary)
    env.scenario = SimpleNamespace(
        config=cfg,
        obstacle_type="cylinders",
        current_obstacle_type="cylinders",
        obstacles=[],
    )
    env.arena_size_x = cfg.ARENA_SIZE_X
    env.arena_size_y = cfg.ARENA_SIZE_Y
    env.arena_height = cfg.ARENA_HEIGHT
    env.num_drones = 1
    env.swarm_mode = False
    env.CLIENT = 0
    env.raycast_sensor = RaycastSensor(ray_length=cfg.RAY_LENGTH)
    env.prev_action = np.zeros(4, dtype=np.float32)
    env.prev_prev_action = np.zeros(4, dtype=np.float32)
    env.visited_cells = set()
    env._goal_hold_counter = 0
    env._prev_raycasts = None
    env.reward_components = {}
    env.episode_min_obstacle_dist = float("inf")
    env.episode_near_misses = 0
    env.episode_min_goal_dist = float("inf")
    env.episode_heading_errors = []
    env.episode_speeds = []
    env.episode_clearances = []
    env.episode_goal_seeking_steps = 0
    env.episode_total_steps = 0
    env.episode_yaw_rates = []
    env._get_ray_ignore_ids = lambda: None
    return env


def test_pretrain_start_goal_samples_both_corridor_directions():
    scenario = StagePretrainScenario(obstacle_type="empty", seed=123)
    pairs = [scenario._generate_start_goal_zones() for _ in range(100)]

    forward = sum(goal[1] > start[1] for start, goal in pairs)
    backward = sum(goal[1] < start[1] for start, goal in pairs)

    assert forward > 0
    assert backward > 0


def test_pretrain_goal_bias_respects_terminal_wall_clearance(monkeypatch):
    scenario = StagePretrainScenario(obstacle_type="empty", seed=456)
    monkeypatch.setattr(scenario.config, "GOAL_CORNER_BIAS", 1.0, raising=False)
    monkeypatch.setattr(scenario.config, "GOAL_END_Y_BIAS", 1.0, raising=False)

    pairs = [scenario._generate_start_goal_zones() for _ in range(100)]
    half_x = scenario.config.ARENA_SIZE_X / 2.0
    half_y = scenario.config.ARENA_SIZE_Y / 2.0
    clearance = scenario.config.GOAL_WALL_CLEARANCE

    for _, goal in pairs:
        assert half_x - abs(goal[0]) >= clearance - 1e-9
        assert half_y - abs(goal[1]) >= clearance - 1e-9
        assert 0.75 <= abs(goal[0]) <= 0.95


def test_observation_contains_goal_in_body_frame_and_enhanced_features():
    cfg = load_config("pretrain")
    env = _nav_stub(cfg)

    drone_pos = np.array([0.0, 0.0, 1.0])
    drone_quat = np.array([0.0, 0.0, 0.0, 1.0])
    drone_vel = np.array([0.0, 0.5, 0.0])
    goal = np.array([0.0, 2.0, 1.5])
    prev_action = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    raycasts = np.linspace(0.2, 1.0, env.raycast_sensor.n_rays, dtype=np.float32)
    max_dist = np.sqrt(env.arena_size_x ** 2 + env.arena_size_y ** 2 + env.arena_height ** 2)

    obs = env._build_observation_vector(
        0,
        drone_pos,
        drone_quat,
        drone_vel,
        goal,
        prev_action,
        raycasts,
        max_dist,
    )

    assert obs.shape == (13 + env.raycast_sensor.n_rays + 4 + env.raycast_sensor.n_rays,)
    np.testing.assert_allclose(obs[:3], (goal - drone_pos) / max_dist, atol=1e-6)
    np.testing.assert_allclose(obs[3], np.linalg.norm(goal - drone_pos) / max_dist, atol=1e-6)
    np.testing.assert_allclose(obs[9:13], prev_action, atol=1e-6)
    np.testing.assert_allclose(obs[13:33], raycasts, atol=1e-6)

    yaw_quat = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    rotated_obs = env._build_observation_vector(
        0,
        drone_pos,
        yaw_quat,
        drone_vel,
        goal,
        prev_action,
        raycasts,
        max_dist,
    )
    np.testing.assert_allclose(rotated_obs[:3], np.array([2.0, 0.0, 0.5]) / max_dist, atol=1e-6)

    enhanced_start = 13 + env.raycast_sensor.n_rays
    assert obs[enhanced_start] > 0.0  # radial velocity toward the goal
    assert 0.0 <= obs[enhanced_start + 1] <= 1.0  # lateral speed
    assert 0.0 <= obs[enhanced_start + 2] <= 1.0  # braking ratio
    assert 0.0 <= obs[enhanced_start + 3] <= 1.0  # boundary clearance
    np.testing.assert_allclose(obs[enhanced_start + 4:], np.zeros(env.raycast_sensor.n_rays), atol=1e-6)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_real_pretrain_env_reset_exposes_goal_and_physical_walls():
    scenario = StagePretrainScenario(obstacle_type="empty", seed=5)
    sync_env_runtime_config(scenario)
    env = NavAviary(scenario=scenario, gui=False)
    try:
        obs, _ = env.reset()
        state = env._getDroneStateVector(0)
        rays = env.raycast_sensor.cast_rays(
            state[:3],
            state[3:7],
            env.CLIENT,
            ignore_body_ids=env._get_ray_ignore_ids(),
        )

        assert obs.shape == env.observation_space.shape
        assert env.start_pos is not None
        assert env.goal_pos is not None
        assert env.start_pos[1] * env.goal_pos[1] < 0.0
        assert len(env._arena_wall_body_ids) >= 4
        assert np.min(rays) < 1.0
    finally:
        env.close()


def _reward_for_state(cfg, pos, vel, goal, prev_dist):
    env = _nav_stub(cfg)
    env.raycast_sensor = _SafeRaycastSensor()
    env.goal_pos = np.asarray(goal, dtype=float)
    env.start_pos = np.array([0.0, -4.5, 1.0], dtype=float)
    env.prev_dist_to_goal = float(prev_dist)

    state = np.zeros(16, dtype=float)
    state[:3] = np.asarray(pos, dtype=float)
    state[3:7] = np.array([0.0, 0.0, 0.0, 1.0])
    state[10:13] = np.asarray(vel, dtype=float)
    env._getDroneStateVector = lambda drone_id: state

    reward = env._computeReward()
    return reward, env.reward_components


def test_reward_uses_pretrain_config_and_discourages_near_goal_overshoot():
    cfg = load_config("pretrain")

    env = _nav_stub(cfg)
    assert env._near_goal_boundary_penalty_value(curr_dist=0.4, boundary_dist=0.0) < 0.0
    assert env._near_goal_boundary_penalty_value(curr_dist=0.4, boundary_dist=1.0) == 0.0
    assert env._hard_obstacle_penalty_value(min_dist=0.1) < -1.0

    far_reward, far_components = _reward_for_state(
        cfg,
        pos=np.array([0.0, -4.0, 1.0]),
        vel=np.array([0.0, 0.5, 0.0]),
        goal=np.array([0.0, 0.0, 1.0]),
        prev_dist=4.1,
    )
    near_reward, near_components = _reward_for_state(
        cfg,
        pos=np.array([0.0, -0.5, 1.0]),
        vel=np.array([0.0, 0.5, 0.0]),
        goal=np.array([0.0, 0.0, 1.0]),
        prev_dist=0.6,
    )
    away_reward, away_components = _reward_for_state(
        cfg,
        pos=np.array([0.0, -0.5, 1.0]),
        vel=np.array([0.0, -0.5, 0.0]),
        goal=np.array([0.0, 0.0, 1.0]),
        prev_dist=0.6,
    )
    hover_reward, hover_components = _reward_for_state(
        cfg,
        pos=np.array([0.0, -0.5, 1.0]),
        vel=np.zeros(3),
        goal=np.array([0.0, 0.0, 1.0]),
        prev_dist=0.5,
    )
    settle_reward, settle_components = _reward_for_state(
        cfg,
        pos=np.array([0.0, -0.5, 1.0]),
        vel=np.array([0.0, 0.05, 0.0]),
        goal=np.array([0.0, 0.0, 1.0]),
        prev_dist=0.55,
    )

    assert near_components["velocity"] < far_components["velocity"] * 0.4
    assert near_components["near_goal_speed"] < 0.0
    assert away_components["near_goal_away"] < 0.0
    assert hover_components["near_goal_stall"] < 0.0
    assert settle_components["near_goal_capture"] > near_components["near_goal_capture"]
    assert hover_reward < 0.0
    assert settle_reward > hover_reward
    assert near_reward > away_reward
    assert far_reward > away_reward


def test_timeout_penalty_targets_near_goal_miss_and_regression():
    cfg = load_config("pretrain")
    env = _nav_stub(cfg)

    far_timeout = env._timeout_near_goal_penalty_value(final_dist=4.0, min_goal_dist=4.0)
    close_timeout = env._timeout_near_goal_penalty_value(final_dist=0.55, min_goal_dist=0.55)
    overshoot_timeout = env._timeout_near_goal_penalty_value(final_dist=1.50, min_goal_dist=0.55)

    assert far_timeout == 0.0
    assert close_timeout < 0.0
    assert overshoot_timeout < close_timeout


def test_pretrain_success_allows_slow_stable_capture_inside_hold_radius():
    cfg = load_config("pretrain")
    env = _nav_stub(cfg)
    env.goal_pos = np.array([0.0, 0.0, 1.0], dtype=float)

    state = np.zeros(16, dtype=float)
    state[3:7] = np.array([0.0, 0.0, 0.0, 1.0])
    env._getDroneStateVector = lambda drone_id: state

    hold_dist = (cfg.SUCCESS_DIST + cfg.SUCCESS_HOLD_RADIUS) / 2.0
    assert cfg.SUCCESS_HOLD_RADIUS > cfg.SUCCESS_DIST

    state[:3] = np.array([0.0, -(cfg.SUCCESS_HOLD_RADIUS + 0.01), 1.0], dtype=float)
    state[10:13] = np.array([0.0, 0.05, 0.0], dtype=float)
    assert not env._check_goal_success(cfg.SUCCESS_HOLD_RADIUS + 0.01, update_counter=True)

    state[:3] = np.array([0.0, -hold_dist, 1.0], dtype=float)
    state[10:13] = np.array([0.0, 1.0, 0.0], dtype=float)
    assert not env._check_goal_success(hold_dist, update_counter=True)
    assert env._goal_hold_counter == 0

    state[10:13] = np.array([0.0, 0.05, 0.0], dtype=float)
    for _ in range(cfg.SUCCESS_HOLD_STEPS - 1):
        assert not env._check_goal_success(hold_dist, update_counter=True)
    assert env._check_goal_success(hold_dist, update_counter=True)

    state[:3] = np.array([0.0, -(cfg.SUCCESS_DIST - 0.01), 1.0], dtype=float)
    state[10:13] = np.array([0.0, 1.0, 0.0], dtype=float)
    assert env._check_goal_success(cfg.SUCCESS_DIST - 0.01, update_counter=True)


def test_sync_env_runtime_config_updates_worker_visible_runtime_module(monkeypatch):
    import config as runtime_config

    cfg = load_config("pretrain")
    monkeypatch.setattr(runtime_config, "REWARD_BOUNDARY_THRESHOLD", 999.0, raising=False)

    sync_env_runtime_config(SimpleNamespace(config=cfg))

    assert runtime_config.REWARD_BOUNDARY_THRESHOLD == cfg.REWARD_BOUNDARY_THRESHOLD == 0.0
    assert runtime_config.REWARD_OBSTACLE_HARD_THRESHOLD == cfg.REWARD_OBSTACLE_HARD_THRESHOLD


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_pretrain_subprocess_env_uses_pretrain_reward_config():
    from multiprocessing import get_all_start_methods

    from stable_baselines3.common.vec_env import SubprocVecEnv

    cfg = load_config("pretrain")
    start_methods = get_all_start_methods()
    start_method = "fork" if "fork" in start_methods else "spawn"
    vec_env = SubprocVecEnv(
        [make_env_pretrain(0, seed=77, obstacle_type="empty", config=cfg)],
        start_method=start_method,
    )
    try:
        hard_penalty = vec_env.env_method("_hard_obstacle_penalty_value", 0.1)[0]
        boundary_penalty = vec_env.env_method("_near_goal_boundary_penalty_value", 0.4, 0.0)[0]
        ray_sensor = vec_env.get_attr("raycast_sensor")[0]

        assert hard_penalty < -1.0
        assert boundary_penalty < 0.0
        assert ray_sensor.ray_length == cfg.RAY_LENGTH
    finally:
        vec_env.close()
