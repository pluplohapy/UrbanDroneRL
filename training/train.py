"""
Unified training script for drone navigation.
Supports Stage 0 (empty) and Stage 1 (static obstacles with RRT* planner).

Usage:
    python training/train.py --stage 0                    # Train Stage 0
    python training/train.py --stage 1                    # Train Stage 1 with RRT*
    python training/train.py --stage 1 --no-planner       # Train Stage 1 without planner
    python training/train.py --stage 0 --timesteps 1000000  # Custom timesteps
    python training/train.py --stage 0 --watch            # Continuous watch: fixed map + trajectory
    python training/train.py --stage pretrain --watch --swarm-drones 8  # 8 drones in one map
"""

import os
import sys
import argparse
import json
import heapq
import shutil
import numpy as np
import torch
import warnings
from datetime import datetime
from collections import Counter, deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv, sync_envs_normalization
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

try:
    from sb3_contrib import RecurrentPPO
except ImportError:  # Keep standard PPO usable if sb3-contrib is not installed.
    RecurrentPPO = None

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.nav_aviary import NavAviary
from envs.nav_aviary_planner import NavAviaryWithPlanner
from scenarios.stage0_empty import Stage0Scenario
from scenarios.stage1_static import Stage1Scenario
from config import load_config
from config.runtime_sync import sync_runtime_config


ALGO_CHOICES = ("ppo", "recurrent_ppo")
PRETRAIN_OBSTACLE_CHOICES = (
    "random",
    "dynamic_mix",
    "empty",
    "cylinders",
    "spheres",
    "crossing_spheres",
    "walls",
    "beams",
    "boxes",
    "gates",
    "slalom",
    "city_blocks",
    "swinging_sticks",
)


def get_algorithm_class(algo: str):
    if algo == "ppo":
        return PPO
    if algo == "recurrent_ppo":
        if RecurrentPPO is None:
            raise ImportError(
                "RecurrentPPO requires sb3-contrib. Install it with: pip install sb3-contrib"
            )
        return RecurrentPPO
    raise ValueError(f"Unknown algorithm: {algo}")


def algorithm_display_name(algo: str) -> str:
    return "RecurrentPPO" if algo == "recurrent_ppo" else "PPO"


def is_recurrent_algorithm(algo: str) -> bool:
    return algo == "recurrent_ppo"


def sync_env_runtime_config(scenario):
    """
    Keep worker-process runtime config aligned with the scenario config.

    SubprocVecEnv starts child Python processes. Those processes import the
    global ``config`` module independently, so the main-process
    sync_runtime_config(...) call is not enough for reward/action code that
    still reads global config constants inside the environment.
    """
    scenario_config = getattr(scenario, "config", None)
    if scenario_config is not None:
        sync_runtime_config(scenario_config)


def build_algorithm_params(ppo_params: dict, algo: str) -> dict:
    params = dict(ppo_params)
    if not is_recurrent_algorithm(algo):
        return params

    params["policy"] = "MlpLstmPolicy"
    # gSDE is a nice PPO exploration tool, but it is an unnecessary moving part
    # for the first recurrent baseline and can interact noisily with LSTM state.
    params["use_sde"] = False
    params.pop("sde_sample_freq", None)

    policy_kwargs = dict(params.get("policy_kwargs") or {})
    policy_kwargs.setdefault("lstm_hidden_size", 128)
    policy_kwargs.setdefault("n_lstm_layers", 1)
    params["policy_kwargs"] = policy_kwargs
    return params


def predict_with_optional_state(model, obs, deterministic: bool, lstm_states=None, episode_starts=None):
    if RecurrentPPO is not None and isinstance(model, RecurrentPPO):
        if episode_starts is None:
            episode_starts = np.ones((obs.shape[0],), dtype=bool)
        return model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=deterministic
        )

    action, _ = model.predict(obs, deterministic=deterministic)
    return action, None


class TrainingDiagnosticsLogger:
    """Compact structured diagnostics writer for training runs."""

    def __init__(
        self,
        enabled: bool,
        base_dir: str,
        run_name: str,
        stage: str,
        use_planner: bool,
        config,
        sample_every: int = 10,
        milestone_window: int = 100,
        bad_top_k: int = 300,
        extra_config: dict | None = None
    ):
        self.enabled = enabled
        self.closed = False
        self.run_dir = None
        self._episodes_fh = None
        self._milestones_fh = None

        if not self.enabled:
            return

        self.config = config
        self.sample_every = max(1, int(sample_every))
        self.milestone_window = max(10, int(milestone_window))
        self.bad_top_k = max(10, int(bad_top_k))
        self.bad_heap = []  # Min-heap of (severity, episode_idx, record)
        self.failure_reasons = Counter()
        self.window = deque(maxlen=self.milestone_window)
        self.milestones_written = 0
        self.total_episodes = 0
        self.outcome_counts = Counter()
        self.global_metric_sums = {}
        self.global_metric_counts = {}
        self.outcome_metric_sums = {
            "success": {},
            "crash": {},
            "timeout": {}
        }
        self.outcome_metric_counts = {
            "success": {},
            "crash": {},
            "timeout": {}
        }
        self.reward_component_sums = {}
        self.reward_component_counts = {}
        self.reward_component_bad_sums = {}
        self.reward_component_bad_counts = {}
        self.crash_with_reward_count = 0
        self.crash_positive_count = 0
        self.out_of_bounds_crash_with_reward_count = 0
        self.out_of_bounds_crash_positive_count = 0
        self.contact_crash_with_reward_count = 0
        self.contact_crash_positive_count = 0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_run_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in run_name)
        self.run_dir = os.path.join(base_dir, f"{safe_run_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.episodes_path = os.path.join(self.run_dir, "episodes_compact.jsonl")
        self.milestones_path = os.path.join(self.run_dir, "milestones.jsonl")
        self.bad_path = os.path.join(self.run_dir, "bad_episodes_top.jsonl")
        self.summary_path = os.path.join(self.run_dir, "summary_blocks.json")
        self.readme_path = os.path.join(self.run_dir, "README.txt")
        self.run_config_path = os.path.join(self.run_dir, "run_config.json")

        self._episodes_fh = open(self.episodes_path, "w", encoding="utf-8")
        self._milestones_fh = open(self.milestones_path, "w", encoding="utf-8")

        run_config = {
            "created_at": datetime.now().isoformat(),
            "stage": stage,
            "use_planner": bool(use_planner),
            "pyb_freq": int(config.PYB_FREQ),
            "ctrl_freq": int(config.CTRL_FREQ),
            "max_steps": int(config.MAX_STEPS),
            "success_dist": float(config.SUCCESS_DIST),
            "min_clearance": float(config.MIN_CLEARANCE),
            "timeout_near_goal_threshold": float(config.TIMEOUT_NEAR_GOAL_THRESHOLD),
            "sample_every": int(self.sample_every),
            "milestone_window": int(self.milestone_window),
            "bad_top_k": int(self.bad_top_k),
            "reward_scales": {
                "progress": float(config.REWARD_PROGRESS_SCALE),
                "velocity": float(config.REWARD_VELOCITY_SCALE),
                "heading": float(config.REWARD_HEADING_SCALE),
                "yaw_penalty": float(config.REWARD_YAW_PENALTY_SCALE),
                "smoothness": float(config.REWARD_ACTION_SMOOTHNESS_SCALE),
                "proximity": float(config.REWARD_PROXIMITY_SCALE),
                "obstacle_approach": float(config.REWARD_OBSTACLE_APPROACH_SCALE),
                "obstacle_hard_threshold": float(getattr(config, "REWARD_OBSTACLE_HARD_THRESHOLD", 0.0)),
                "obstacle_hard_scale": float(getattr(config, "REWARD_OBSTACLE_HARD_SCALE", 0.0)),
                "boundary_threshold": float(config.REWARD_BOUNDARY_THRESHOLD),
                "boundary_scale": float(config.REWARD_BOUNDARY_SCALE),
                "boundary_outward": float(getattr(config, "REWARD_BOUNDARY_OUTWARD_SCALE", 0.0)),
                "near_goal_radius": float(getattr(config, "REWARD_NEAR_GOAL_RADIUS", 0.0)),
                "near_goal_progress": float(getattr(config, "REWARD_NEAR_GOAL_PROGRESS_SCALE", 0.0)),
                "near_goal_stall": float(getattr(config, "REWARD_NEAR_GOAL_STALL_SCALE", 0.0)),
                "near_goal_away": float(getattr(config, "REWARD_NEAR_GOAL_AWAY_SCALE", 0.0)),
                "near_goal_speed": float(getattr(config, "REWARD_NEAR_GOAL_SPEED_SCALE", 0.0)),
                "near_goal_boundary": float(getattr(config, "REWARD_NEAR_GOAL_BOUNDARY_SCALE", 0.0)),
                "step_penalty": float(config.REWARD_STEP_PENALTY),
                "success": float(config.REWARD_SUCCESS),
                "crash": float(config.REWARD_CRASH),
                "out_of_bounds_extra": float(config.REWARD_OUT_OF_BOUNDS_EXTRA),
                "collision_extra": float(config.REWARD_COLLISION_EXTRA)
            },
            "safety_shield": {
                "enabled": bool(getattr(config, "SAFETY_SHIELD_ENABLED", False)),
                "soft_clearance": float(getattr(config, "SAFETY_SHIELD_SOFT_CLEARANCE", 0.0)),
                "hard_clearance": float(getattr(config, "SAFETY_SHIELD_HARD_CLEARANCE", 0.0)),
                "avoid_gain": float(getattr(config, "SAFETY_SHIELD_AVOID_GAIN", 0.0)),
                "brake_gain": float(getattr(config, "SAFETY_SHIELD_BRAKE_GAIN", 0.0)),
                "max_yaw": float(getattr(config, "SAFETY_SHIELD_MAX_YAW", 1.0)),
                "topk": int(getattr(config, "SAFETY_SHIELD_TOPK", 1)),
                "vertical_gain": float(getattr(config, "SAFETY_SHIELD_VERTICAL_GAIN", 0.0)),
                "hard_brake_scale": float(getattr(config, "SAFETY_SHIELD_HARD_BRAKE_SCALE", 0.0))
            }
        }
        if extra_config is not None:
            run_config["run"] = extra_config

        with open(self.run_config_path, "w", encoding="utf-8") as fh:
            json.dump(run_config, fh, ensure_ascii=False, indent=2)

        with open(self.readme_path, "w", encoding="utf-8") as fh:
            fh.write(
                "Training diagnostics files:\n"
                "1) run_config.json - immutable run setup and reward scales\n"
                "2) episodes_compact.jsonl - sampled successful episodes + all failures\n"
                "3) milestones.jsonl - rolling-window snapshots for trend analysis\n"
                "4) bad_episodes_top.jsonl - worst failure episodes ranked by severity\n"
                "5) summary_blocks.json - final aggregated diagnostics grouped by blocks\n"
            )

    @staticmethod
    def _safe_float(value):
        try:
            if value is None:
                return None
            value = float(value)
            if np.isnan(value) or np.isinf(value):
                return None
            return value
        except Exception:
            return None

    @staticmethod
    def _to_float_list(value):
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
            return [float(x) for x in arr.tolist()]
        except Exception:
            return None

    def _mean_from_maps(self, sums_map, counts_map, key):
        count = counts_map.get(key, 0)
        if count <= 0:
            return None
        return float(sums_map.get(key, 0.0) / count)

    def _update_metric_map(self, sums_map, counts_map, key, value):
        value = self._safe_float(value)
        if value is None:
            return
        sums_map[key] = sums_map.get(key, 0.0) + value
        counts_map[key] = counts_map.get(key, 0) + 1

    def _infer_outcome(self, info):
        if bool(info.get("is_success", False)):
            return "success"
        if bool(info.get("is_crash", False)):
            return "crash"
        return "timeout"

    def _infer_failure_reason(self, record):
        outcome = record["outcome"]
        if outcome == "success":
            return "success"

        if outcome == "crash":
            if bool(record.get("out_of_bounds", False)):
                return "out_of_bounds"
            if bool(record.get("has_contact", False)):
                return "collision"
            closest = record.get("closest_obstacle")
            if closest is not None and closest < self.config.MIN_CLEARANCE:
                return "low_clearance_crash"
            return "crash_unknown"

        # timeout
        min_goal = record.get("min_goal_distance")
        progress_ratio = record.get("progress_ratio")
        avg_speed = record.get("avg_speed")
        if min_goal is not None and min_goal <= self.config.TIMEOUT_NEAR_GOAL_THRESHOLD:
            return "timeout_near_goal"
        if progress_ratio is not None and progress_ratio < 0.2:
            return "timeout_no_progress"
        if avg_speed is not None and avg_speed < self.config.HOVERING_SPEED_THRESHOLD:
            return "timeout_stuck_hovering"
        return "timeout_other"

    def _failure_severity(self, record):
        outcome = record["outcome"]
        reason = record["failure_reason"]
        score = 0.0

        if outcome == "crash":
            score += 3.0
        elif outcome == "timeout":
            score += 2.0

        if reason == "out_of_bounds":
            score += 1.5
        elif reason == "collision":
            score += 1.0
        elif reason == "timeout_near_goal":
            score += 1.2
        elif reason == "timeout_no_progress":
            score += 0.8
        elif reason == "timeout_stuck_hovering":
            score += 1.0

        dist_final = record.get("dist_to_goal")
        start_dist = record.get("start_distance")
        if dist_final is not None and start_dist is not None and start_dist > 1e-6:
            score += min(2.0, max(0.0, dist_final / start_dist))

        closest = record.get("closest_obstacle")
        if closest is not None:
            score += max(0.0, self.config.MIN_CLEARANCE - closest)

        reward = record.get("episode_reward")
        if reward is not None:
            score += max(0.0, min(2.0, -reward / 1000.0))

        return float(score)

    def _record_milestone(self, episode_idx, timesteps):
        if self.total_episodes % self.milestone_window != 0 or len(self.window) == 0:
            return

        outcomes = [x["outcome"] for x in self.window]
        rewards = [x["episode_reward"] for x in self.window if x["episode_reward"] is not None]
        final_dists = [x["dist_to_goal"] for x in self.window if x["dist_to_goal"] is not None]
        progress = [x["progress_ratio"] for x in self.window if x["progress_ratio"] is not None]
        crash_rewards = [
            x["episode_reward"]
            for x in self.window
            if x["outcome"] == "crash" and x["episode_reward"] is not None
        ]

        snapshot = {
            "episode": int(episode_idx),
            "timesteps": int(timesteps),
            "window_size": int(len(self.window)),
            "rates": {
                "success": float(np.mean([1.0 if o == "success" else 0.0 for o in outcomes])),
                "crash": float(np.mean([1.0 if o == "crash" else 0.0 for o in outcomes])),
                "timeout": float(np.mean([1.0 if o == "timeout" else 0.0 for o in outcomes]))
            },
            "means": {
                "reward": float(np.mean(rewards)) if rewards else None,
                "dist_to_goal": float(np.mean(final_dists)) if final_dists else None,
                "progress_ratio": float(np.mean(progress)) if progress else None
            },
            "alignment": {
                "positive_crash_rate": float(np.mean([1.0 if r > 0.0 else 0.0 for r in crash_rewards])) if crash_rewards else None
            }
        }
        self._milestones_fh.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        self._milestones_fh.flush()
        self.milestones_written += 1

    def log_episode(self, episode_idx, timesteps, info):
        if not self.enabled:
            return

        outcome = self._infer_outcome(info)
        episode_reward = self._safe_float(info.get("episode", {}).get("r"))
        episode_length = self._safe_float(info.get("episode", {}).get("l"))
        dist_to_goal = self._safe_float(info.get("dist_to_goal"))
        start_distance = self._safe_float(info.get("start_distance"))
        min_goal_distance = self._safe_float(info.get("min_goal_distance"))
        closest_obstacle = self._safe_float(info.get("closest_obstacle"))
        min_ray_dist = self._safe_float(info.get("min_ray_dist"))
        boundary_dist = self._safe_float(info.get("boundary_dist"))
        shield_last_min_dist = self._safe_float(info.get("shield_last_min_dist"))
        shield_intervention_ratio = self._safe_float(info.get("shield_intervention_ratio"))
        avg_speed = self._safe_float(info.get("avg_speed"))
        avg_heading_error = self._safe_float(info.get("avg_heading_error"))
        path_efficiency = self._safe_float(info.get("path_efficiency"))
        action_smoothness = self._safe_float(info.get("action_smoothness"))
        hovering_time = self._safe_float(info.get("hovering_time"))
        spinning_time = self._safe_float(info.get("spinning_time"))
        goal_seeking_ratio = self._safe_float(info.get("goal_seeking_ratio"))
        obstacle_type = info.get("obstacle_type")

        progress_ratio = None
        if start_distance is not None and start_distance > 1e-6 and dist_to_goal is not None:
            progress_ratio = float((start_distance - dist_to_goal) / start_distance)

        record = {
            "episode": int(episode_idx),
            "timesteps": int(timesteps),
            "outcome": outcome,
            "obstacle_type": str(obstacle_type) if obstacle_type is not None else None,
            "obstacle_count": info.get("obstacle_count"),
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "dist_to_goal": dist_to_goal,
            "start_distance": start_distance,
            "min_goal_distance": min_goal_distance,
            "progress_ratio": progress_ratio,
            "closest_obstacle": closest_obstacle,
            "min_ray_dist": min_ray_dist,
            "boundary_dist": boundary_dist,
            "shield_last_min_dist": shield_last_min_dist,
            "shield_intervention_ratio": shield_intervention_ratio,
            "avg_speed": avg_speed,
            "avg_heading_error": avg_heading_error,
            "path_efficiency": path_efficiency,
            "action_smoothness": action_smoothness,
            "hovering_time": hovering_time,
            "spinning_time": spinning_time,
            "goal_seeking_ratio": goal_seeking_ratio,
            "start_pos": self._to_float_list(info.get("start_pos")),
            "goal_pos": self._to_float_list(info.get("goal_pos")),
            "final_pos": self._to_float_list(info.get("final_pos")),
            "out_of_bounds": bool(info.get("out_of_bounds", False)),
            "has_contact": bool(info.get("has_contact", False)),
        }

        reward_components = info.get("reward_components")
        if isinstance(reward_components, dict):
            compact_components = {}
            for k, v in reward_components.items():
                fv = self._safe_float(v)
                if fv is not None:
                    compact_components[k] = fv
                    self._update_metric_map(self.reward_component_sums, self.reward_component_counts, k, fv)
            if compact_components:
                record["reward_components"] = compact_components

        record["failure_reason"] = self._infer_failure_reason(record)
        self.failure_reasons[record["failure_reason"]] += 1

        if outcome == "crash" and episode_reward is not None:
            self.crash_with_reward_count += 1
            if episode_reward > 0.0:
                self.crash_positive_count += 1

            if record.get("out_of_bounds", False):
                self.out_of_bounds_crash_with_reward_count += 1
                if episode_reward > 0.0:
                    self.out_of_bounds_crash_positive_count += 1

            if record.get("has_contact", False):
                self.contact_crash_with_reward_count += 1
                if episode_reward > 0.0:
                    self.contact_crash_positive_count += 1

        is_bad = outcome != "success"
        if is_bad and "reward_components" in record:
            for k, v in record["reward_components"].items():
                self._update_metric_map(self.reward_component_bad_sums, self.reward_component_bad_counts, k, v)

        # Global stats
        self.total_episodes += 1
        self.outcome_counts[outcome] += 1
        for metric_key in [
            "episode_reward",
            "episode_length",
            "dist_to_goal",
            "min_goal_distance",
            "progress_ratio",
            "closest_obstacle",
            "boundary_dist",
            "shield_last_min_dist",
            "shield_intervention_ratio",
            "avg_speed",
            "avg_heading_error",
            "path_efficiency",
            "action_smoothness",
            "hovering_time",
            "spinning_time",
            "goal_seeking_ratio",
        ]:
            self._update_metric_map(self.global_metric_sums, self.global_metric_counts, metric_key, record.get(metric_key))
            self._update_metric_map(
                self.outcome_metric_sums[outcome],
                self.outcome_metric_counts[outcome],
                metric_key,
                record.get(metric_key)
            )

        # Sampled compact stream: keep all failures + sampled successes
        should_write = is_bad or (record["episode"] % self.sample_every == 0)
        if should_write:
            self._episodes_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Keep only top-K most severe failures
        if is_bad:
            severity = self._failure_severity(record)
            rec_with_score = dict(record)
            rec_with_score["severity"] = severity
            item = (severity, int(record["episode"]), rec_with_score)
            if len(self.bad_heap) < self.bad_top_k:
                heapq.heappush(self.bad_heap, item)
            else:
                if severity > self.bad_heap[0][0]:
                    heapq.heapreplace(self.bad_heap, item)

        self.window.append(record)
        self._record_milestone(record["episode"], record["timesteps"])

        # Lightweight flush cadence
        if self.total_episodes % 20 == 0:
            self._episodes_fh.flush()

    def close(self):
        if (not self.enabled) or self.closed:
            return
        self.closed = True

        if self._episodes_fh is not None:
            self._episodes_fh.flush()
            self._episodes_fh.close()
        if self._milestones_fh is not None:
            # Ensure there is at least one snapshot even for short runs.
            if self.milestones_written == 0 and len(self.window) > 0:
                outcomes = [x["outcome"] for x in self.window]
                rewards = [x["episode_reward"] for x in self.window if x["episode_reward"] is not None]
                final_dists = [x["dist_to_goal"] for x in self.window if x["dist_to_goal"] is not None]
                progress = [x["progress_ratio"] for x in self.window if x["progress_ratio"] is not None]
                last = self.window[-1]
                snapshot = {
                    "episode": int(last["episode"]),
                    "timesteps": int(last["timesteps"]),
                    "window_size": int(len(self.window)),
                    "rates": {
                        "success": float(np.mean([1.0 if o == "success" else 0.0 for o in outcomes])),
                        "crash": float(np.mean([1.0 if o == "crash" else 0.0 for o in outcomes])),
                        "timeout": float(np.mean([1.0 if o == "timeout" else 0.0 for o in outcomes]))
                    },
                    "means": {
                        "reward": float(np.mean(rewards)) if rewards else None,
                        "dist_to_goal": float(np.mean(final_dists)) if final_dists else None,
                        "progress_ratio": float(np.mean(progress)) if progress else None
                    },
                    "alignment": {
                        "positive_crash_rate": float(np.mean([
                            1.0 if x["episode_reward"] > 0.0 else 0.0
                            for x in self.window
                            if x["outcome"] == "crash" and x["episode_reward"] is not None
                        ])) if any(x["outcome"] == "crash" and x["episode_reward"] is not None for x in self.window) else None
                    },
                    "final_window_snapshot": True
                }
                self._milestones_fh.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
                self._milestones_fh.flush()

            self._milestones_fh.flush()
            self._milestones_fh.close()

        # Write ranked bad episodes
        ranked_bad = sorted(self.bad_heap, key=lambda x: (x[0], x[1]), reverse=True)
        with open(self.bad_path, "w", encoding="utf-8") as fh:
            for _, _, record in ranked_bad:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        def metric_means(sums_map, counts_map):
            result = {}
            for key in sorted(sums_map.keys()):
                mean_val = self._mean_from_maps(sums_map, counts_map, key)
                if mean_val is not None:
                    result[key] = mean_val
            return result

        summary = {
            "total_episodes": int(self.total_episodes),
            "outcomes": {
                "success": int(self.outcome_counts.get("success", 0)),
                "crash": int(self.outcome_counts.get("crash", 0)),
                "timeout": int(self.outcome_counts.get("timeout", 0)),
                "success_rate": float(self.outcome_counts.get("success", 0) / max(1, self.total_episodes)),
                "crash_rate": float(self.outcome_counts.get("crash", 0) / max(1, self.total_episodes)),
                "timeout_rate": float(self.outcome_counts.get("timeout", 0) / max(1, self.total_episodes)),
            },
            "failure_reason_breakdown": dict(self.failure_reasons),
            "global_means": metric_means(self.global_metric_sums, self.global_metric_counts),
            "outcome_means": {
                outcome: metric_means(self.outcome_metric_sums[outcome], self.outcome_metric_counts[outcome])
                for outcome in ("success", "crash", "timeout")
            },
            "reward_component_means": metric_means(self.reward_component_sums, self.reward_component_counts),
            "reward_component_means_bad_only": metric_means(self.reward_component_bad_sums, self.reward_component_bad_counts),
            "reward_alignment": {
                "positive_crash_rate": float(self.crash_positive_count / self.crash_with_reward_count) if self.crash_with_reward_count > 0 else None,
                "positive_out_of_bounds_crash_rate": float(self.out_of_bounds_crash_positive_count / self.out_of_bounds_crash_with_reward_count) if self.out_of_bounds_crash_with_reward_count > 0 else None,
                "positive_contact_crash_rate": float(self.contact_crash_positive_count / self.contact_crash_with_reward_count) if self.contact_crash_with_reward_count > 0 else None,
                "counts": {
                    "crash_with_reward": int(self.crash_with_reward_count),
                    "crash_positive": int(self.crash_positive_count),
                    "out_of_bounds_crash_with_reward": int(self.out_of_bounds_crash_with_reward_count),
                    "out_of_bounds_crash_positive": int(self.out_of_bounds_crash_positive_count),
                    "contact_crash_with_reward": int(self.contact_crash_with_reward_count),
                    "contact_crash_positive": int(self.contact_crash_positive_count)
                }
            },
            "bad_episode_top_k": int(len(ranked_bad)),
            "files": {
                "episodes_compact": self.episodes_path,
                "milestones": self.milestones_path,
                "bad_episodes_top": self.bad_path
            }
        }

        with open(self.summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)


class ProgressCallback(BaseCallback):
    """Callback for displaying training progress."""

    def __init__(self, stage, use_planner, config, diag_logger=None, verbose=0):
        super().__init__(verbose)
        self.stage = stage
        self.use_planner = use_planner
        self.config = config
        self.diag_logger = diag_logger
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_crashes = []
        self.episode_timeouts = []
        self.episode_lengths = []
        self.swarm_mode_detected = False
        self.swarm_respawns = []
        self.swarm_successes = []
        self.swarm_crashes = []

        # Planner metrics (only for Stage 1 with planner)
        if use_planner:
            self.planning_failures = []
            self.n_waypoints = []

        # Debug metrics storage
        if config.DEBUG_MODE:
            self.reward_components = {k: [] for k in [
                'progress', 'velocity', 'proximity', 'obstacle', 'obstacle_approach',
                'obstacle_hard', 'boundary', 'boundary_outward', 'near_goal_stall',
                'near_goal_progress', 'near_goal_away', 'near_goal_speed', 'near_goal_boundary',
                'step_penalty', 'exploration', 'terminal',
                'efficiency_bonus', 'yaw_penalty', 'heading', 'smoothness'
            ]}
            self.navigation_metrics = {'path_efficiency': [], 'avg_heading_error': [], 'avg_speed': []}
            self.episode_metrics = {'start_distance': [], 'min_goal_distance': [], 'closest_obstacle': [], 'n_near_misses': []}
            self.action_stats = {'action_mean': [], 'action_std': [], 'action_smoothness': []}
            self.timeout_metrics = {'final_dist': [], 'min_dist': [], 'start_dist': []}

            # Extended episode metrics
            if self.config.LOG_EXTENDED_EPISODE_METRICS:
                self.extended_metrics = {
                    'avg_clearance': [],
                    'hovering_time': [],
                    'goal_seeking_ratio': [],
                    'spinning_time': []
                }

            # Path following metrics (for planner)
            if use_planner and self.config.LOG_PATH_FOLLOWING_METRICS:
                self.path_following_metrics = {
                    'avg_cross_track_error': [],
                    'max_cross_track_error': [],
                    'path_following_score': []
                }

    def _on_step(self) -> bool:
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_count += 1
                    ep_reward = info["episode"]["r"]
                    ep_length = info["episode"]["l"]
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)

                    if self.diag_logger is not None:
                        self.diag_logger.log_episode(
                            episode_idx=self.episode_count,
                            timesteps=self.num_timesteps,
                            info=info
                        )

                    # Track success/crash/timeout
                    if "is_success" in info:
                        if info.get("swarm_mode", False):
                            self.swarm_mode_detected = True
                            self.swarm_respawns.append(info.get("swarm_respawns", 0))
                            self.swarm_successes.append(info.get("swarm_successes", 0))
                            self.swarm_crashes.append(info.get("swarm_crashes", 0))

                        is_success = info["is_success"]
                        is_crash = info.get("is_crash", False)
                        is_timeout = not is_success and not is_crash

                        self.episode_successes.append(1 if is_success else 0)
                        self.episode_crashes.append(1 if is_crash else 0)
                        self.episode_timeouts.append(1 if is_timeout else 0)

                        # Planner metrics
                        if self.use_planner:
                            if "planning_failed" in info:
                                self.planning_failures.append(1 if info["planning_failed"] else 0)
                            if "n_waypoints" in info:
                                self.n_waypoints.append(info["n_waypoints"])

                        # Collect debug metrics
                        if self.config.DEBUG_MODE:
                            # Reward components
                            if self.config.LOG_REWARD_COMPONENTS and 'reward_components' in info:
                                for k, v in info['reward_components'].items():
                                    if k in self.reward_components:
                                        self.reward_components[k].append(v)

                            # Navigation metrics
                            if self.config.LOG_NAVIGATION_METRICS:
                                if 'path_efficiency' in info:
                                    self.navigation_metrics['path_efficiency'].append(info['path_efficiency'])
                                if 'avg_heading_error' in info:
                                    self.navigation_metrics['avg_heading_error'].append(info['avg_heading_error'])
                                if 'avg_speed' in info:
                                    self.navigation_metrics['avg_speed'].append(info['avg_speed'])

                            # Episode metrics
                            if self.config.LOG_EPISODE_METRICS:
                                if 'start_distance' in info:
                                    self.episode_metrics['start_distance'].append(info['start_distance'])
                                if 'min_goal_distance' in info:
                                    self.episode_metrics['min_goal_distance'].append(info['min_goal_distance'])
                                if 'closest_obstacle' in info:
                                    self.episode_metrics['closest_obstacle'].append(info['closest_obstacle'])
                                if 'n_near_misses' in info:
                                    self.episode_metrics['n_near_misses'].append(info['n_near_misses'])

                            # Action stats
                            if self.config.LOG_ACTION_STATS:
                                if 'action_mean' in info:
                                    self.action_stats['action_mean'].append(info['action_mean'])
                                if 'action_smoothness' in info:
                                    self.action_stats['action_smoothness'].append(info['action_smoothness'])

                            # Timeout analysis
                            if self.config.LOG_TIMEOUT_ANALYSIS and is_timeout:
                                self.timeout_metrics['final_dist'].append(info.get('dist_to_goal', 0))
                                self.timeout_metrics['min_dist'].append(info.get('min_goal_distance', 0))
                                self.timeout_metrics['start_dist'].append(info.get('start_distance', 0))

                            # Extended episode metrics
                            if self.config.LOG_EXTENDED_EPISODE_METRICS:
                                if 'avg_clearance' in info:
                                    self.extended_metrics['avg_clearance'].append(info['avg_clearance'])
                                if 'hovering_time' in info:
                                    self.extended_metrics['hovering_time'].append(info['hovering_time'])
                                if 'goal_seeking_ratio' in info:
                                    self.extended_metrics['goal_seeking_ratio'].append(info['goal_seeking_ratio'])
                                if 'spinning_time' in info:
                                    self.extended_metrics['spinning_time'].append(info['spinning_time'])

                            # Path following metrics
                            if self.use_planner and self.config.LOG_PATH_FOLLOWING_METRICS:
                                if 'avg_cross_track_error' in info:
                                    self.path_following_metrics['avg_cross_track_error'].append(info['avg_cross_track_error'])
                                if 'max_cross_track_error' in info:
                                    self.path_following_metrics['max_cross_track_error'].append(info['max_cross_track_error'])
                                if 'path_following_score' in info:
                                    self.path_following_metrics['path_following_score'].append(info['path_following_score'])

                    if self.episode_count % self.config.LOG_INTERVAL_EPISODES == 0:
                        self._print_progress()

        return True

    def _on_training_end(self) -> None:
        if self.diag_logger is not None:
            self.diag_logger.close()

    def _print_progress(self):
        """Print training progress."""
        recent_rewards = self.episode_rewards[-10:]
        recent_successes = self.episode_successes[-min(50, len(self.episode_successes)):]
        recent_crashes = self.episode_crashes[-min(50, len(self.episode_crashes)):]
        recent_timeouts = self.episode_timeouts[-min(50, len(self.episode_timeouts)):]
        recent_lengths = self.episode_lengths[-10:]

        avg_reward = np.mean(recent_rewards)
        success_rate = np.mean(recent_successes) if recent_successes else 0.0
        crash_rate = np.mean(recent_crashes) if recent_crashes else 0.0
        timeout_rate = np.mean(recent_timeouts) if recent_timeouts else 0.0
        avg_length = np.mean(recent_lengths)

        # Basic output
        print(f"Episode {self.episode_count:4d} | Steps: {self.num_timesteps:7d}")
        print(f"  Outcomes : S={success_rate:4.0%} | C={crash_rate:4.0%} | T={timeout_rate:4.0%}")

        if self.swarm_mode_detected and len(self.swarm_respawns) > 0:
            recent_n = min(10, len(self.swarm_respawns))
            avg_respawns = np.mean(self.swarm_respawns[-recent_n:])
            avg_swarm_success = np.mean(self.swarm_successes[-recent_n:]) if len(self.swarm_successes) > 0 else 0
            avg_swarm_crash = np.mean(self.swarm_crashes[-recent_n:]) if len(self.swarm_crashes) > 0 else 0
            print(f"  Swarm    : respawns={avg_respawns:.1f} | successes={avg_swarm_success:.1f} | crashes={avg_swarm_crash:.1f}")

        # Planner metrics
        if self.use_planner and len(self.planning_failures) > 0:
            recent_failures = self.planning_failures[-min(50, len(self.planning_failures)):]
            failure_rate = np.mean(recent_failures)
            print(f"  Planner  : failures={failure_rate:4.0%}", end="")

            if len(self.n_waypoints) > 0:
                recent_wp = self.n_waypoints[-min(50, len(self.n_waypoints)):]
                avg_wp = np.mean(recent_wp)
                print(f" | avg_waypoints={avg_wp:.1f}")
            else:
                print()

        if self.config.DEBUG_MODE:
            # Reward components
            if self.config.LOG_REWARD_COMPONENTS and len(self.reward_components['progress']) > 0:
                recent_n = min(50, len(self.reward_components['progress']))
                prog = np.mean(self.reward_components['progress'][-recent_n:])
                vel = np.mean(self.reward_components['velocity'][-recent_n:])
                prox = np.mean(self.reward_components['proximity'][-recent_n:])
                obst = np.mean(self.reward_components['obstacle'][-recent_n:])
                for key in ('obstacle_approach', 'obstacle_hard'):
                    if len(self.reward_components[key]) > 0:
                        obst += np.mean(self.reward_components[key][-recent_n:])
                bound = np.mean(self.reward_components['boundary'][-recent_n:]) if len(self.reward_components['boundary']) > 0 else 0
                bound_out = np.mean(self.reward_components['boundary_outward'][-recent_n:]) if len(self.reward_components['boundary_outward']) > 0 else 0
                near = 0.0
                for key in ('near_goal_stall', 'near_goal_progress', 'near_goal_away', 'near_goal_speed', 'near_goal_boundary'):
                    if len(self.reward_components[key]) > 0:
                        near += np.mean(self.reward_components[key][-recent_n:])
                step = np.mean(self.reward_components['step_penalty'][-recent_n:])
                term = np.mean(self.reward_components['terminal'][-recent_n:]) if len(self.reward_components['terminal']) > 0 else 0
                print(f"  Reward   : total={avg_reward:7.1f} | prog={prog:5.1f} | vel={vel:4.1f} | prox={prox:4.1f} | obst={obst:5.1f} | bound={bound + bound_out:5.1f} | near={near:5.1f} | term={term:5.1f}")

            # Navigation metrics
            if self.config.LOG_NAVIGATION_METRICS and len(self.navigation_metrics['path_efficiency']) > 0:
                recent_n = min(50, len(self.navigation_metrics['path_efficiency']))
                eff = np.mean(self.navigation_metrics['path_efficiency'][-recent_n:])
                heading = np.mean(self.navigation_metrics['avg_heading_error'][-recent_n:]) if len(self.navigation_metrics['avg_heading_error']) > 0 else 0
                speed = np.mean(self.navigation_metrics['avg_speed'][-recent_n:]) if len(self.navigation_metrics['avg_speed']) > 0 else 0
                print(f"  Navigate : efficiency={eff:.2f} | heading_err={heading:.1f}° | speed={speed:.2f}m/s")

            # Timeout analysis
            if self.config.LOG_TIMEOUT_ANALYSIS and len(self.timeout_metrics['final_dist']) > 0:
                final = np.mean(self.timeout_metrics['final_dist'])
                min_d = np.mean(self.timeout_metrics['min_dist'])

                # Calculate timeout_near_goal and timeout_stuck
                near_goal_count = sum(1 for d in self.timeout_metrics['min_dist'] if d < self.config.TIMEOUT_NEAR_GOAL_THRESHOLD)
                near_goal_pct = near_goal_count / len(self.timeout_metrics['min_dist']) if len(self.timeout_metrics['min_dist']) > 0 else 0

                print(f"  Timeouts : final_dist={final:.1f}m | min_dist={min_d:.1f}m | near_goal={near_goal_pct:.0%}")

            # Extended episode metrics
            if self.config.LOG_EXTENDED_EPISODE_METRICS and len(self.extended_metrics['avg_clearance']) > 0:
                recent_n = min(50, len(self.extended_metrics['avg_clearance']))
                clearance = np.mean(self.extended_metrics['avg_clearance'][-recent_n:])
                hovering = np.mean(self.extended_metrics['hovering_time'][-recent_n:]) if len(self.extended_metrics['hovering_time']) > 0 else 0
                goal_seek = np.mean(self.extended_metrics['goal_seeking_ratio'][-recent_n:]) if len(self.extended_metrics['goal_seeking_ratio']) > 0 else 0
                spinning = np.mean(self.extended_metrics['spinning_time'][-recent_n:]) if len(self.extended_metrics['spinning_time']) > 0 else 0
                print(f"  Behavior : clearance={clearance:.2f}m | hovering={hovering:.0%} | goal_seek={goal_seek:.0%} | spinning={spinning:.0%}")

            # Path following metrics
            if self.use_planner and self.config.LOG_PATH_FOLLOWING_METRICS and len(self.path_following_metrics['avg_cross_track_error']) > 0:
                recent_n = min(50, len(self.path_following_metrics['avg_cross_track_error']))
                avg_cte = np.mean(self.path_following_metrics['avg_cross_track_error'][-recent_n:])
                max_cte = np.mean(self.path_following_metrics['max_cross_track_error'][-recent_n:])
                pf_score = np.mean(self.path_following_metrics['path_following_score'][-recent_n:])
                print(f"  PathFollow: avg_CTE={avg_cte:.2f}m | max_CTE={max_cte:.2f}m | score={pf_score:.0%}")

            # Action stats
            if self.config.LOG_ACTION_STATS and len(self.action_stats['action_smoothness']) > 0:
                recent_n = min(50, len(self.action_stats['action_smoothness']))
                smooth = np.mean(self.action_stats['action_smoothness'][-recent_n:])
                if len(self.navigation_metrics['avg_speed']) > 0:
                    speed = np.mean(self.navigation_metrics['avg_speed'][-recent_n:])
                    print(f"  Actions  : speed={speed:.2f} | smoothness={smooth:.3f}")
        else:
            # Simple output when debug is off
            print(f"  Reward: {avg_reward:7.2f} | Length: {avg_length:5.1f}")

        print()  # Empty line for readability


class SuccessRateEvalCallback(BaseCallback):
    """
    Periodic evaluation callback.
    Best checkpoint criterion: higher success_rate, then higher mean_reward.
    """

    def __init__(
        self,
        eval_env,
        best_model_save_path: str,
        log_path: str,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 20,
        deterministic: bool = True,
        load_existing_best: bool = True,
        run_id: str | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.eval_freq = max(1, int(eval_freq))
        self.n_eval_episodes = max(1, int(n_eval_episodes))
        self.deterministic = bool(deterministic)
        self.run_id = run_id
        self.best_success_rate = -np.inf
        self.best_mean_reward = -np.inf
        self.eval_history_path = os.path.join(self.log_path, "eval_history.jsonl")
        self.loaded_existing_best = False
        self.loaded_existing_best_timesteps = None

        os.makedirs(self.best_model_save_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        if load_existing_best:
            self._load_existing_best()

    def _load_existing_best(self):
        best_model_path = os.path.join(self.best_model_save_path, "best_model.zip")
        best_normalize_path = os.path.join(self.best_model_save_path, "best_model_vecnormalize.pkl")
        if not (os.path.exists(best_model_path) and os.path.exists(best_normalize_path)):
            return
        if not os.path.exists(self.eval_history_path):
            return

        best_row = None
        with open(self.eval_history_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    success_rate = float(row.get("success_rate", -np.inf))
                    mean_reward = float(row.get("mean_reward", -np.inf))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue

                if (
                    best_row is None
                    or success_rate > float(best_row.get("success_rate", -np.inf)) + 1e-12
                    or (
                        abs(success_rate - float(best_row.get("success_rate", -np.inf))) <= 1e-12
                        and mean_reward > float(best_row.get("mean_reward", -np.inf)) + 1e-9
                    )
                ):
                    best_row = row

        if best_row is None:
            return

        self.best_success_rate = float(best_row.get("success_rate", -np.inf))
        self.best_mean_reward = float(best_row.get("mean_reward", -np.inf))
        self.loaded_existing_best_timesteps = best_row.get("timesteps")
        self.loaded_existing_best = True

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        if isinstance(self.training_env, VecNormalize):
            try:
                sync_envs_normalization(self.training_env, self.eval_env)
            except Exception as exc:
                if self.verbose >= 1:
                    print(f"[EVAL][WARN] Could not sync VecNormalize stats: {exc}")

        rewards = []
        lengths = []
        outcomes = []
        crash_oob = 0
        crash_contact = 0

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            ep_reward = 0.0
            ep_len = 0
            last_info = {}
            lstm_states = None
            episode_starts = np.ones((self.eval_env.num_envs,), dtype=bool)

            while not done:
                action, lstm_states = predict_with_optional_state(
                    self.model,
                    obs,
                    deterministic=self.deterministic,
                    lstm_states=lstm_states,
                    episode_starts=episode_starts
                )
                step_result = self.eval_env.step(action)

                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = bool(terminated[0] or truncated[0])
                else:
                    obs, reward, done_vec, info = step_result
                    done = bool(done_vec[0])

                ep_reward += float(reward[0])
                ep_len += 1
                if info and len(info) > 0:
                    last_info = info[0]
                episode_starts = np.array([done], dtype=bool)

            is_success = bool(last_info.get("is_success", False))
            is_crash = bool(last_info.get("is_crash", False))
            out_of_bounds = bool(last_info.get("out_of_bounds", False))
            has_contact = bool(last_info.get("has_contact", False))

            if is_success:
                outcome = "success"
            elif is_crash or out_of_bounds:
                outcome = "crash"
                if out_of_bounds:
                    crash_oob += 1
                elif has_contact:
                    crash_contact += 1
            else:
                outcome = "timeout"

            rewards.append(ep_reward)
            lengths.append(ep_len)
            outcomes.append(outcome)

        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        mean_ep_length = float(np.mean(lengths)) if lengths else 0.0
        success_rate = float(np.mean([1.0 if o == "success" else 0.0 for o in outcomes])) if outcomes else 0.0
        crash_rate = float(np.mean([1.0 if o == "crash" else 0.0 for o in outcomes])) if outcomes else 0.0
        timeout_rate = float(np.mean([1.0 if o == "timeout" else 0.0 for o in outcomes])) if outcomes else 0.0

        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_ep_length", mean_ep_length)
        self.logger.record("eval/success_rate", success_rate)
        self.logger.record("eval/crash_rate", crash_rate)
        self.logger.record("eval/timeout_rate", timeout_rate)
        self.logger.record("time/total_timesteps", self.num_timesteps)
        self.logger.dump(self.num_timesteps)

        eval_row = {
            "created_at": datetime.now().isoformat(),
            "run_id": self.run_id,
            "timesteps": int(self.num_timesteps),
            "mean_reward": mean_reward,
            "mean_ep_length": mean_ep_length,
            "success_rate": success_rate,
            "crash_rate": crash_rate,
            "timeout_rate": timeout_rate,
            "crash_out_of_bounds": int(crash_oob),
            "crash_contact": int(crash_contact),
            "episodes": int(self.n_eval_episodes),
            "deterministic": bool(self.deterministic),
        }
        with open(self.eval_history_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(eval_row, ensure_ascii=False) + "\n")

        is_better = (
            success_rate > self.best_success_rate + 1e-12
            or (
                abs(success_rate - self.best_success_rate) <= 1e-12
                and mean_reward > self.best_mean_reward + 1e-9
            )
        )
        if is_better:
            self.best_success_rate = success_rate
            self.best_mean_reward = mean_reward
            self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(os.path.join(self.best_model_save_path, "best_model_vecnormalize.pkl"))
            if self.verbose >= 1:
                print("New best checkpoint by success_rate!")
                print(
                    f"[EVAL] steps={self.num_timesteps} | success={success_rate:.1%} | "
                    f"reward={mean_reward:.2f} | len={mean_ep_length:.1f}"
                )
        elif self.verbose >= 1:
            print(
                f"[EVAL] steps={self.num_timesteps} | success={success_rate:.1%} | "
                f"reward={mean_reward:.2f} | len={mean_ep_length:.1f}"
            )

        return True


def make_env_stage0(rank, seed=0, use_planner=True, config=None, gui=False, watch_fps=None,
                    fixed_map=False, show_paths=False, swarm_drones=1):
    """Create Stage 0 environment (empty arena)."""
    def _init():
        scenario = Stage0Scenario(seed=seed + rank)
        sync_env_runtime_config(scenario)

        if use_planner:
            env = NavAviaryWithPlanner(
                scenario=scenario,
                gui=gui,
                watch_fps=watch_fps,
                fixed_map=fixed_map,
                show_trajectory=show_paths,
                use_planner=True,
                replan_freq=0,
                waypoint_threshold=config.WAYPOINT_THRESHOLD,
                planner_params={
                    'max_iter': config.RRT_MAX_ITER,
                    'step_size': config.RRT_STEP_SIZE,
                    'goal_bias': config.RRT_GOAL_BIAS,
                    'rewire_radius': config.RRT_REWIRE_RADIUS,
                    'collision_check_resolution': config.RRT_COLLISION_RESOLUTION,
                    'verbose': 0
                }
            )
        else:
            env = NavAviary(
                scenario=scenario,
                gui=gui,
                watch_fps=watch_fps,
                fixed_map=fixed_map,
                show_trajectory=show_paths,
                num_drones=swarm_drones
            )

        env = Monitor(env)
        return env
    return _init


def make_env_stage1(rank, seed=0, use_planner=True, config=None, gui=False, watch_fps=None,
                    fixed_map=False, show_paths=False, swarm_drones=1):
    """Create Stage 1 environment (static obstacles)."""
    def _init():
        scenario = Stage1Scenario(seed=seed + rank)
        sync_env_runtime_config(scenario)

        if use_planner:
            env = NavAviaryWithPlanner(
                scenario=scenario,
                gui=gui,
                watch_fps=watch_fps,
                fixed_map=fixed_map,
                show_trajectory=show_paths,
                use_planner=True,
                replan_freq=0,
                waypoint_threshold=config.WAYPOINT_THRESHOLD,
                planner_params={
                    'max_iter': config.RRT_MAX_ITER,
                    'step_size': config.RRT_STEP_SIZE,
                    'goal_bias': config.RRT_GOAL_BIAS,
                    'rewire_radius': config.RRT_REWIRE_RADIUS,
                    'collision_check_resolution': config.RRT_COLLISION_RESOLUTION,
                    'verbose': 0
                }
            )
        else:
            env = NavAviary(
                scenario=scenario,
                gui=gui,
                watch_fps=watch_fps,
                fixed_map=fixed_map,
                show_trajectory=show_paths,
                num_drones=swarm_drones
            )

        env = Monitor(env)
        return env
    return _init


def make_env_pretrain(rank, seed=0, obstacle_type='random', config=None, gui=False, watch_fps=None,
                      fixed_map=False, show_paths=False, swarm_drones=1):
    """Create Pretrain environment (diverse obstacles, no planner)."""
    def _init():
        from scenarios.stage_pretrain import StagePretrainScenario

        scenario = StagePretrainScenario(
            obstacle_type=obstacle_type,
            seed=seed + rank
        )
        sync_env_runtime_config(scenario)

        # Pretrain NEVER uses planner
        env = NavAviary(
            scenario=scenario,
            gui=gui,
            watch_fps=watch_fps,
            fixed_map=fixed_map,
            show_trajectory=show_paths,
            num_drones=swarm_drones
        )

        env = Monitor(env)
        return env
    return _init


def apply_loaded_model_hyperparams(model, ppo_params):
    """
    Apply a safe subset of PPO hyperparameters after loading a checkpoint.
    This keeps continued training aligned with current config values.
    """
    if not isinstance(ppo_params, dict):
        return []

    applied = []

    if "learning_rate" in ppo_params:
        lr = float(ppo_params["learning_rate"])
        model.learning_rate = lr
        model.lr_schedule = get_schedule_fn(lr)
        for group in model.policy.optimizer.param_groups:
            group["lr"] = lr
        applied.append(f"learning_rate={lr:g}")

    if "target_kl" in ppo_params:
        model.target_kl = float(ppo_params["target_kl"])
        applied.append(f"target_kl={float(ppo_params['target_kl']):g}")

    if "clip_range" in ppo_params:
        model.clip_range = get_schedule_fn(float(ppo_params["clip_range"]))
        applied.append(f"clip_range={float(ppo_params['clip_range']):g}")

    if "ent_coef" in ppo_params:
        model.ent_coef = float(ppo_params["ent_coef"])
        applied.append(f"ent_coef={float(ppo_params['ent_coef']):g}")

    if "vf_coef" in ppo_params:
        model.vf_coef = float(ppo_params["vf_coef"])
        applied.append(f"vf_coef={float(ppo_params['vf_coef']):g}")

    if "max_grad_norm" in ppo_params:
        model.max_grad_norm = float(ppo_params["max_grad_norm"])
        applied.append(f"max_grad_norm={float(ppo_params['max_grad_norm']):g}")

    return applied


def main():
    parser = argparse.ArgumentParser(description='Train drone navigation')
    parser.add_argument('--stage', type=str, required=True, choices=['0', '1', 'pretrain'],
                        help='Training stage: 0 (empty), 1 (obstacles), pretrain (diverse)')
    parser.add_argument('--algo', type=str, default='ppo', choices=ALGO_CHOICES,
                        help='Policy optimizer: ppo or recurrent_ppo (LSTM)')
    parser.add_argument('--obstacle-type', type=str, default='random',
                        choices=PRETRAIN_OBSTACLE_CHOICES,
                        help='Obstacle type for pretrain stage (random, dynamic_mix, or specific type)')
    parser.add_argument('--no-planner', action='store_true',
                        help='Disable RRT* planner (enabled by default for stage 0/1, always disabled for pretrain)')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Training timesteps to run; when continuing, this is additional budget')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Override PPO/RecurrentPPO learning rate for this run')
    parser.add_argument('--seed', type=int, default=None,
                        help='Training RNG/base environment seed (default: config.SEED)')
    parser.add_argument('--n-envs', type=int, default=None,
                        help='Number of parallel environments (default: from config)')
    parser.add_argument('--continue', dest='continue_training', action='store_true',
                        help='Continue training from checkpoint')
    parser.add_argument('--init-model', type=str, default=None,
                        help='Optional checkpoint .zip path to initialize/continue from')
    parser.add_argument('--init-normalize', type=str, default=None,
                        help='Optional VecNormalize .pkl path for --init-model')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (detailed logging and metrics)')
    parser.add_argument('--watch', action='store_true',
                        help='Watch training in PyBullet GUI (forces n-envs=1, fixed map, trajectory rendering)')
    parser.add_argument('--watch-fps', type=float, default=30.0,
                        help='Target FPS for watch mode (default: 30)')
    parser.add_argument('--fixed-map', action='store_true',
                        help='Keep same start/goal/obstacles across episodes')
    parser.add_argument('--show-paths', action='store_true',
                        help='Draw drone trajectory in GUI during training')
    parser.add_argument('--no-show-paths', action='store_true',
                        help='Disable trajectory drawing even in watch mode')
    parser.add_argument('--swarm-drones', type=int, default=1,
                        help='Number of drones in one shared map (parallel in one env)')
    parser.add_argument('--safety-shield', dest='safety_shield', action='store_true',
                        help='Enable safety shield during training')
    parser.add_argument('--no-safety-shield', dest='safety_shield', action='store_false',
                        help='Disable safety shield during training (recommended for training from scratch)')
    parser.add_argument('--diag', dest='diag', action='store_true',
                        help='Enable structured diagnostics logs (default: enabled)')
    parser.add_argument('--no-diag', dest='diag', action='store_false',
                        help='Disable structured diagnostics logs')
    parser.add_argument('--diag-dir', type=str, default='logs/training_diagnostics',
                        help='Directory for structured diagnostics logs')
    parser.add_argument('--diag-sample-every', type=int, default=10,
                        help='Write every Nth successful episode to compact diagnostics stream')
    parser.add_argument('--diag-window', type=int, default=100,
                        help='Rolling window size for milestone snapshots')
    parser.add_argument('--diag-bad-topk', type=int, default=300,
                        help='Keep top-K worst failure episodes in diagnostics')
    parser.add_argument('--eval', dest='eval_enabled', action='store_true',
                        help='Enable periodic evaluation and best-checkpoint saving')
    parser.add_argument('--no-eval', dest='eval_enabled', action='store_false',
                        help='Disable periodic evaluation and best-checkpoint saving (default)')
    parser.add_argument('--eval-freq', type=int, default=50_000,
                        help='Evaluation frequency in environment steps (default: 50000)')
    parser.add_argument('--eval-episodes', type=int, default=20,
                        help='Number of episodes per evaluation pass (default: 20)')
    parser.add_argument('--eval-seed', type=int, default=12345,
                        help='Base seed for evaluation environment (default: 12345)')
    parser.add_argument('--eval-deterministic', dest='eval_deterministic', action='store_true',
                        help='Use deterministic policy during periodic eval (default)')
    parser.add_argument('--eval-stochastic', dest='eval_deterministic', action='store_false',
                        help='Use stochastic policy during periodic eval')
    parser.add_argument('--reset-best', action='store_true',
                        help='Ignore existing eval history and allow overwriting best checkpoint from this run')
    parser.add_argument('--save-final-to-main', dest='save_final_to_main', action='store_true',
                        help='Save the last policy to the standard model path at the end')
    parser.add_argument('--no-save-final-to-main', dest='save_final_to_main', action='store_false',
                        help='Keep the standard model path unchanged; final policy is saved as last checkpoint only')
    parser.add_argument('--promote-best-to-main', action='store_true',
                        help='Copy the best eval checkpoint to the standard model path after training')
    parser.set_defaults(
        diag=True,
        safety_shield=None,
        eval_enabled=False,
        eval_deterministic=True,
        save_final_to_main=None
    )

    args = parser.parse_args()
    algo = args.algo
    algo_cls = get_algorithm_class(algo)
    algo_name = algorithm_display_name(algo)

    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error("--learning-rate must be > 0")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be >= 0")
    if args.promote_best_to_main and not args.eval_enabled:
        parser.error("--promote-best-to-main requires periodic eval; remove --no-eval")

    # Load config for the specified stage
    config = load_config(args.stage)
    algo_params = build_algorithm_params(config.PPO_PARAMS, algo)
    if args.learning_rate is not None:
        algo_params["learning_rate"] = float(args.learning_rate)
    train_seed = int(config.SEED if args.seed is None else args.seed)
    config.SEED = train_seed

    # Safety shield toggle (training-friendly default: disabled unless explicitly enabled)
    if args.safety_shield is not None:
        config.SAFETY_SHIELD_ENABLED = bool(args.safety_shield)

    # Enable debug mode if requested
    if args.debug:
        config.DEBUG_MODE = True
        print("[DEBUG] Debug mode enabled")

    # Critical: keep env runtime module aligned with stage-specific config values.
    sync_runtime_config(config)

    # Determine configuration
    stage = args.stage

    # Planner logic: pretrain never uses planner, others use by default unless --no-planner
    if stage == 'pretrain':
        use_planner = False
    else:
        use_planner = not args.no_planner

    # Set default n_envs
    n_envs = args.n_envs if args.n_envs is not None else config.N_ENVS
    watch_mode = args.watch
    fixed_map = args.fixed_map or watch_mode
    show_paths = (args.show_paths or watch_mode) and (not args.no_show_paths)
    swarm_drones = args.swarm_drones
    save_final_to_main = args.save_final_to_main
    if save_final_to_main is None:
        save_final_to_main = not (args.continue_training and args.eval_enabled)

    if swarm_drones < 1:
        parser.error("--swarm-drones must be >= 1")
    if args.diag_sample_every < 1:
        parser.error("--diag-sample-every must be >= 1")
    if args.diag_window < 10:
        parser.error("--diag-window must be >= 10")
    if args.diag_bad_topk < 10:
        parser.error("--diag-bad-topk must be >= 10")
    if args.eval_freq < 1:
        parser.error("--eval-freq must be >= 1")
    if args.eval_episodes < 1:
        parser.error("--eval-episodes must be >= 1")

    if swarm_drones > 1:
        if use_planner:
            parser.error("Swarm mode currently supports only non-planner envs. Use --no-planner or --stage pretrain.")
        if n_envs != 1:
            print(f"[SWARM] Requested n_envs={n_envs}, forcing n_envs=1 (swarm parallelism is inside one env)")
            n_envs = 1
        if not fixed_map:
            print("[SWARM] Enabling fixed map for shared-map swarm training")
            fixed_map = True

    if watch_mode:
        if args.watch_fps <= 0:
            parser.error("--watch-fps must be > 0")
        if n_envs != 1:
            print(f"[WATCH] Requested n_envs={n_envs}, forcing n_envs=1 for GUI mode")
            n_envs = 1
        if swarm_drones > 1 and args.watch_fps <= 30:
            print("[WATCH] For smoother swarm rendering use --watch-fps 60..120")

    # Set default timesteps
    if args.timesteps is None:
        if stage == '0':
            timesteps = 500_000
        elif stage == '1':
            timesteps = 1_500_000
        else:  # pretrain
            timesteps = 500_000
    else:
        timesteps = args.timesteps

    # Model paths
    if stage == '0':
        if use_planner:
            model_path = "models/ppo_drone_nav_stage0_planner"
            normalize_path = "models/vec_normalize_stage0_planner.pkl"
            log_name = "PPO_stage0_planner"
        else:
            model_path = "models/ppo_drone_nav_stage0"
            normalize_path = "models/vec_normalize_stage0.pkl"
            log_name = "PPO_stage0"
    elif stage == '1':
        if use_planner:
            model_path = "models/ppo_drone_nav_stage1_planner"
            normalize_path = "models/vec_normalize_stage1_planner.pkl"
            log_name = "PPO_stage1_planner"
        else:
            model_path = "models/ppo_drone_nav_stage1"
            normalize_path = "models/vec_normalize_stage1.pkl"
            log_name = "PPO_stage1"
    else:  # pretrain
        obstacle_type = args.obstacle_type
        model_path = f"models/ppo_pretrain_{obstacle_type}"
        normalize_path = f"models/vec_normalize_pretrain_{obstacle_type}.pkl"
        log_name = f"PPO_pretrain_{obstacle_type}"

    if algo != "ppo":
        model_dir, model_base = os.path.split(model_path)
        if model_base.startswith("ppo_"):
            model_base = model_base.replace("ppo_", f"{algo}_", 1)
        else:
            model_base = f"{algo}_{model_base}"
        model_path = os.path.join(model_dir, model_base)

        normalize_dir, normalize_base = os.path.split(normalize_path)
        if normalize_base.startswith("vec_normalize_"):
            normalize_base = f"vec_normalize_{algo}_{normalize_base[len('vec_normalize_'):]}"
        else:
            normalize_base = f"{algo}_{normalize_base}"
        normalize_path = os.path.join(normalize_dir, normalize_base)

        if log_name.startswith("PPO_"):
            log_name = f"{algo_name}_{log_name[len('PPO_'):]}"
        else:
            log_name = f"{algo_name}_{log_name}"

    # Use separate artifacts for enhanced observations because policy input shape changes.
    enhanced_obs = bool(getattr(config, "USE_ENHANCED_OBS", False))
    if enhanced_obs:
        obs_suffix = "_enhanced_obs"
        model_path = f"{model_path}{obs_suffix}"
        if normalize_path.endswith(".pkl"):
            normalize_path = normalize_path[:-4] + f"{obs_suffix}.pkl"
        else:
            normalize_path = f"{normalize_path}{obs_suffix}"
        log_name = f"{log_name}{obs_suffix}"

    # Use separate checkpoints/logs for swarm runs to avoid shape mismatch with single-drone artifacts.
    if swarm_drones > 1:
        swarm_suffix = f"_swarm{swarm_drones}"
        model_path = f"{model_path}{swarm_suffix}"
        if normalize_path.endswith(".pkl"):
            normalize_path = normalize_path[:-4] + f"{swarm_suffix}.pkl"
        else:
            normalize_path = f"{normalize_path}{swarm_suffix}"
        log_name = f"{log_name}{swarm_suffix}"

    # Print configuration
    print("=" * 60)
    print(f"DRONE NAVIGATION TRAINING - STAGE {stage.upper()}")
    if stage == 'pretrain':
        print(f"Mode: PRETRAIN (NO PLANNER) - Obstacle type: {args.obstacle_type}")
    else:
        print(f"Mode: {'WITH RRT* PLANNER' if use_planner else 'WITHOUT PLANNER'}")
    print("=" * 60)
    print(f"\n[CONFIG]")
    print(f"  Stage: {stage}")
    print(f"  Algorithm: {algo_name}")
    if stage == 'pretrain':
        print(f"  Obstacle type: {args.obstacle_type}")
    print(f"  Use planner: {use_planner}")
    print(f"  Total timesteps: {timesteps:,}")
    print(f"  Training seed: {train_seed}")
    print(f"  Learning rate: {float(algo_params.get('learning_rate', 0.0)):g}")
    print(f"  Parallel envs: {n_envs}")
    print(f"  Watch mode: {watch_mode}")
    print(f"  Fixed map: {fixed_map}")
    print(f"  Show paths: {show_paths}")
    print(f"  Enhanced observations: {enhanced_obs}")
    print(f"  Swarm drones: {swarm_drones}")
    print(f"  Diagnostics logs: {args.diag}")
    print(f"  Periodic eval: {args.eval_enabled}")
    if args.eval_enabled:
        print(f"  Eval frequency: {args.eval_freq:,} env steps")
        print(f"  Eval episodes: {args.eval_episodes}")
        print(f"  Eval deterministic: {args.eval_deterministic}")
        print(f"  Preserve existing best: {not args.reset_best}")
        print(f"  Promote best to main: {args.promote_best_to_main}")
    print(f"  Save final to main: {save_final_to_main}")
    print(f"  Safety shield: {bool(getattr(config, 'SAFETY_SHIELD_ENABLED', False))}")
    if watch_mode:
        print(f"  Watch FPS: {args.watch_fps}")
    print(f"  Arena: {config.ARENA_SIZE_X}x{config.ARENA_SIZE_Y}x{config.ARENA_HEIGHT}m")
    print(f"  Model path: {model_path}")

    if use_planner:
        print(f"\n[CONFIG] RRT* Planner Parameters (OPTIMIZED):")
        print(f"  max_iter: {config.RRT_MAX_ITER}")
        print(f"  step_size: {config.RRT_STEP_SIZE}m")
        print(f"  goal_bias: {config.RRT_GOAL_BIAS}")
        print(f"  rewire_radius: {config.RRT_REWIRE_RADIUS}m")
        print(f"  collision_check_resolution: {config.RRT_COLLISION_RESOLUTION}m")
        print(f"  max_segment_length: {config.RRT_MAX_SEGMENT_LENGTH}m")

    if stage == 'pretrain':
        print(f"\n[CONFIG] Pretrain Obstacle Types:")
        for obs_type, params in config.OBSTACLE_TYPES.items():
            dynamic_str = " (DYNAMIC)" if params.get('dynamic', False) else ""
            print(f"  - {params['name']}{dynamic_str}")
        if args.obstacle_type == 'random':
            print(f"  Training on: ALL TYPES (random)")
        else:
            print(f"  Training on: {args.obstacle_type.upper()} only")

    # Check for checkpoints
    checkpoint_exists = os.path.exists(f"{model_path}.zip") and os.path.exists(normalize_path)
    checkpoint_model_path = args.init_model if args.init_model else f"{model_path}.zip"
    checkpoint_normalize_path = args.init_normalize if args.init_normalize else normalize_path

    if args.continue_training:
        print(f"  Continue from model: {checkpoint_model_path}")
        print(f"  Continue from normalize: {checkpoint_normalize_path}")

    # Check for transfer learning (Stage 0 -> Stage 1)
    stage0_model = "models/ppo_drone_nav_stage0_planner.zip"
    stage0_normalize = "models/vec_normalize_stage0_planner.pkl"
    can_transfer = (
        algo == "ppo"
        and stage == '1'
        and os.path.exists(stage0_model)
        and os.path.exists(stage0_normalize)
    )

    np.random.seed(train_seed)
    torch.manual_seed(train_seed)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Create environments
    print(f"\n[SETUP] Creating environments...")
    if stage == '0':
        env_fns = [make_env_stage0(i, train_seed, use_planner, config,
                                   gui=watch_mode, watch_fps=args.watch_fps,
                                   fixed_map=fixed_map, show_paths=show_paths,
                                   swarm_drones=swarm_drones)
                   for i in range(n_envs)]
    elif stage == '1':
        env_fns = [make_env_stage1(i, train_seed, use_planner, config,
                                   gui=watch_mode, watch_fps=args.watch_fps,
                                   fixed_map=fixed_map, show_paths=show_paths,
                                   swarm_drones=swarm_drones)
                   for i in range(n_envs)]
    else:  # pretrain
        env_fns = [make_env_pretrain(i, train_seed, args.obstacle_type, config,
                                     gui=watch_mode, watch_fps=args.watch_fps,
                                     fixed_map=fixed_map, show_paths=show_paths,
                                     swarm_drones=swarm_drones)
                   for i in range(n_envs)]

    if watch_mode:
        vec_env = DummyVecEnv(env_fns)
        print("✓ DummyVecEnv created (watch mode)")
    else:
        vec_env = SubprocVecEnv(env_fns)

    # Load or create model
    if args.continue_training:
        if not (os.path.exists(checkpoint_model_path) and os.path.exists(checkpoint_normalize_path)):
            parser.error(
                "Requested --continue but checkpoint files are missing: "
                f"model={checkpoint_model_path}, normalize={checkpoint_normalize_path}"
            )

        print(f"\n[LOAD] Continuing training from checkpoint...")
        vec_env = VecNormalize.load(checkpoint_normalize_path, vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
        print("✓ VecNormalize stats loaded")

        model = algo_cls.load(checkpoint_model_path, env=vec_env)
        print("✓ Model loaded")
        updated = apply_loaded_model_hyperparams(model, algo_params)
        if updated:
            print(f"✓ Applied {algo_name} overrides for continued training: " + ", ".join(updated))
        print(f"\nStarting from {model.num_timesteps} steps")

    elif stage == '1' and can_transfer and not checkpoint_exists:
        print(f"\n[LOAD] Using Stage 0 model for transfer learning...")
        vec_env = VecNormalize.load(stage0_normalize, vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
        print("✓ VecNormalize stats loaded from Stage 0")

        model = algo_cls.load(stage0_model, env=vec_env)
        print("✓ Stage 0 model loaded for transfer learning")
        updated = apply_loaded_model_hyperparams(model, algo_params)
        if updated:
            print(f"✓ Applied {algo_name} overrides for transfer training: " + ", ".join(updated))
        print(f"\nStarting from {model.num_timesteps} steps")

    else:
        print(f"\n[SETUP] Starting from scratch...")
        if args.init_normalize:
            if not os.path.exists(args.init_normalize):
                parser.error(f"--init-normalize file not found: {args.init_normalize}")
            vec_env = VecNormalize.load(args.init_normalize, vec_env)
            vec_env.training = True
            vec_env.norm_reward = True
            print(f"✓ VecNormalize stats initialized from {args.init_normalize}")
        else:
            vec_env = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0
            )
            print("✓ VecNormalize created")
        print("✓ Environments created")

        print(f"\n[SETUP] Creating {algo_name} model...")
        model = algo_cls(
            **algo_params,
            env=vec_env,
            tensorboard_log="./logs/",
            verbose=1,
            device="auto"
        )
        print("✓ Model created")

    # Create callback
    diag_logger = TrainingDiagnosticsLogger(
        enabled=args.diag,
        base_dir=args.diag_dir,
        run_name=log_name,
        stage=stage,
        use_planner=use_planner,
        config=config,
        sample_every=args.diag_sample_every,
        milestone_window=args.diag_window,
        bad_top_k=args.diag_bad_topk,
        extra_config={
            "algo": algo,
            "seed": int(train_seed),
            "watch_mode": watch_mode,
            "n_envs": int(n_envs),
            "swarm_drones": int(swarm_drones),
            "safety_shield_enabled": bool(getattr(config, "SAFETY_SHIELD_ENABLED", False)),
            "obstacle_type": args.obstacle_type if stage == "pretrain" else None,
            "continue_training": bool(args.continue_training),
            "init_model": checkpoint_model_path if args.continue_training else None,
            "init_normalize": checkpoint_normalize_path if args.continue_training else None,
            "timesteps_target": int(timesteps),
            "learning_rate": float(algo_params.get("learning_rate", 0.0)),
            "eval_enabled": bool(args.eval_enabled),
            "eval_freq_env_steps": int(args.eval_freq),
            "eval_episodes": int(args.eval_episodes),
            "eval_seed": int(args.eval_seed),
            "eval_deterministic": bool(args.eval_deterministic),
            "reset_best": bool(args.reset_best),
            "save_final_to_main": bool(save_final_to_main),
            "promote_best_to_main": bool(args.promote_best_to_main)
        }
    )
    if args.diag and diag_logger.run_dir is not None:
        print(f"[DIAG] Run dir: {diag_logger.run_dir}")

    progress_callback = ProgressCallback(
        stage=stage,
        use_planner=use_planner,
        config=config,
        diag_logger=diag_logger
    )
    callbacks = [progress_callback]

    eval_env = None
    eval_callback = None
    best_model_dir = os.path.join("models", "best_checkpoints", log_name)
    best_model_path = os.path.join(best_model_dir, "best_model.zip")
    best_normalize_path = os.path.join(best_model_dir, "best_model_vecnormalize.pkl")
    last_model_dir = os.path.join("models", "last_checkpoints", log_name)
    last_model_path = os.path.join(last_model_dir, "last_model")
    last_normalize_path = os.path.join(last_model_dir, "last_model_vecnormalize.pkl")
    eval_log_dir = os.path.join("logs", "eval", log_name)

    if args.eval_enabled:
        print("\n[EVAL] Setting up periodic evaluation...")
        eval_seed = int(args.eval_seed)

        if stage == '0':
            eval_env_fn = make_env_stage0(
                rank=0,
                seed=eval_seed,
                use_planner=use_planner,
                config=config,
                gui=False,
                watch_fps=None,
                fixed_map=False,
                show_paths=False,
                swarm_drones=1
            )
        elif stage == '1':
            eval_env_fn = make_env_stage1(
                rank=0,
                seed=eval_seed,
                use_planner=use_planner,
                config=config,
                gui=False,
                watch_fps=None,
                fixed_map=False,
                show_paths=False,
                swarm_drones=1
            )
        else:
            eval_env_fn = make_env_pretrain(
                rank=0,
                seed=eval_seed,
                obstacle_type=args.obstacle_type,
                config=config,
                gui=False,
                watch_fps=None,
                fixed_map=False,
                show_paths=False,
                swarm_drones=1
            )

        eval_env = DummyVecEnv([eval_env_fn])
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=float(getattr(vec_env, "clip_obs", 10.0)),
            clip_reward=10.0
        )
        eval_env.training = False
        eval_env.norm_reward = False

        os.makedirs(best_model_dir, exist_ok=True)
        os.makedirs(eval_log_dir, exist_ok=True)
        eval_freq_rollouts = max(args.eval_freq // n_envs, 1)

        eval_callback = SuccessRateEvalCallback(
            eval_env=eval_env,
            best_model_save_path=best_model_dir,
            log_path=eval_log_dir,
            eval_freq=eval_freq_rollouts,
            n_eval_episodes=args.eval_episodes,
            deterministic=args.eval_deterministic,
            load_existing_best=not args.reset_best,
            run_id=os.path.basename(diag_logger.run_dir) if diag_logger.run_dir else None,
            verbose=1
        )
        callbacks.append(eval_callback)

        print(f"✓ Eval env ready (seed={eval_seed})")
        print(f"✓ Eval callback frequency: every {eval_freq_rollouts} rollout steps (~{args.eval_freq:,} env steps)")
        print(f"✓ Best model path: {best_model_path}")
        print("✓ Best-checkpoint criterion: success_rate (tie-break: mean_reward)")
        if eval_callback.loaded_existing_best:
            print(
                "✓ Existing best preserved: "
                f"success={eval_callback.best_success_rate:.1%}, "
                f"reward={eval_callback.best_mean_reward:.2f}, "
                f"steps={eval_callback.loaded_existing_best_timesteps}"
            )
        elif args.reset_best:
            print("⚠ Existing best history ignored for this run (--reset-best)")

    print(f"\n[TRAINING] Starting training...")
    print("-" * 60)

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=False,
            tb_log_name=log_name
        )
        print("\n" + "-" * 60)
        print("✓ Training completed")

    except KeyboardInterrupt:
        print("\n\n[INFO] Training interrupted by user")

    # Save the last policy separately so a post-peak policy does not silently
    # replace the best evaluated artifact.
    print(f"\n[SAVE] Saving model...")
    os.makedirs(last_model_dir, exist_ok=True)
    model.save(last_model_path)
    vec_env.save(last_normalize_path)
    print(f"✓ Last model saved to {last_model_path}.zip")
    print(f"✓ Last normalization saved to {last_normalize_path}")

    if args.promote_best_to_main:
        if os.path.exists(best_model_path) and os.path.exists(best_normalize_path):
            shutil.copyfile(best_model_path, f"{model_path}.zip")
            shutil.copyfile(best_normalize_path, normalize_path)
            print(f"✓ Best eval model promoted to {model_path}.zip")
            print(f"✓ Best eval normalization promoted to {normalize_path}")
        else:
            print("⚠ Could not promote best checkpoint: best model or normalization file is missing")
    elif save_final_to_main:
        model.save(model_path)
        vec_env.save(normalize_path)
        print(f"✓ Final model saved to {model_path}.zip")
        print(f"✓ Final normalization saved to {normalize_path}")
    else:
        print(f"✓ Main model preserved: {model_path}.zip")
        print(f"✓ Main normalization preserved: {normalize_path}")

    if args.eval_enabled:
        if os.path.exists(best_model_path):
            print(f"✓ Best eval model saved to {best_model_path}")
        else:
            print("⚠ Periodic eval was enabled, but best_model.zip was not produced")

    if diag_logger is not None:
        diag_logger.close()

    if eval_env is not None:
        eval_env.close()
    vec_env.close()

    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)
    print(f"\nTotal training steps: {model.num_timesteps:,}")
    if args.diag and diag_logger.run_dir is not None:
        print(f"[DIAGNOSTICS] Structured logs saved to: {diag_logger.run_dir}")

    # Next steps
    print("\n[NEXT STEPS]")
    if stage == '0':
        print("  1. Visualize Stage 0:")
        if use_planner:
            print(f"     python visualize_planner.py --model {model_path}")
        else:
            print(f"     python visualize.py --model {model_path}")
        print("\n  2. Train Stage 1 with planner:")
        print("     python training/train.py --stage 1")
    elif stage == '1':
        print("  1. Visualize Stage 1:")
        if use_planner:
            print(f"     python visualize_planner.py --model {model_path}")
        else:
            print(f"     python visualize.py --model {model_path}")
        print("\n  2. Compare with/without planner:")
        print("     python compare_planner.py")
    else:  # pretrain
        print("  1. Visualize Pretrain:")
        print(f"     python visualization/visualize.py --algo {algo} --model {model_path} --stage pretrain")
        print("\n  2. Preview scenario generation:")
        print(f"     python visualization/preview_scenarios.py --stage pretrain --obstacle-type {args.obstacle_type}")
        print("\n  3. Train on different obstacle type:")
        print(f"     python training/train.py --algo {algo} --stage pretrain --obstacle-type spheres")


if __name__ == "__main__":
    main()
