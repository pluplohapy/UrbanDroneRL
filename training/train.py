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
import numpy as np
import torch
import warnings
from datetime import datetime
from collections import Counter, deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

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
                "step_penalty": float(config.REWARD_STEP_PENALTY),
                "success": float(config.REWARD_SUCCESS),
                "crash": float(config.REWARD_CRASH)
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
        avg_speed = self._safe_float(info.get("avg_speed"))
        avg_heading_error = self._safe_float(info.get("avg_heading_error"))
        path_efficiency = self._safe_float(info.get("path_efficiency"))
        action_smoothness = self._safe_float(info.get("action_smoothness"))
        hovering_time = self._safe_float(info.get("hovering_time"))
        spinning_time = self._safe_float(info.get("spinning_time"))
        goal_seeking_ratio = self._safe_float(info.get("goal_seeking_ratio"))

        progress_ratio = None
        if start_distance is not None and start_distance > 1e-6 and dist_to_goal is not None:
            progress_ratio = float((start_distance - dist_to_goal) / start_distance)

        record = {
            "episode": int(episode_idx),
            "timesteps": int(timesteps),
            "outcome": outcome,
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "dist_to_goal": dist_to_goal,
            "start_distance": start_distance,
            "min_goal_distance": min_goal_distance,
            "progress_ratio": progress_ratio,
            "closest_obstacle": closest_obstacle,
            "min_ray_dist": min_ray_dist,
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
            self.reward_components = {k: [] for k in ['progress', 'velocity', 'proximity', 'obstacle', 'step_penalty', 'exploration', 'terminal', 'efficiency_bonus', 'yaw_penalty', 'heading', 'smoothness']}
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
                step = np.mean(self.reward_components['step_penalty'][-recent_n:])
                term = np.mean(self.reward_components['terminal'][-recent_n:]) if len(self.reward_components['terminal']) > 0 else 0
                print(f"  Reward   : total={avg_reward:7.1f} | prog={prog:5.1f} | vel={vel:4.1f} | prox={prox:4.1f} | obst={obst:5.1f} | term={term:5.1f}")

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


def make_env_stage0(rank, seed=0, use_planner=True, config=None, gui=False, watch_fps=None,
                    fixed_map=False, show_paths=False, swarm_drones=1):
    """Create Stage 0 environment (empty arena)."""
    def _init():
        scenario = Stage0Scenario(seed=seed + rank)

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


def main():
    parser = argparse.ArgumentParser(description='Train drone navigation')
    parser.add_argument('--stage', type=str, required=True, choices=['0', '1', 'pretrain'],
                        help='Training stage: 0 (empty), 1 (obstacles), pretrain (diverse)')
    parser.add_argument('--obstacle-type', type=str, default='random',
                        choices=['random', 'empty', 'cylinders', 'spheres', 'walls', 'beams', 'boxes', 'swinging_sticks'],
                        help='Obstacle type for pretrain stage (default: random, includes empty)')
    parser.add_argument('--no-planner', action='store_true',
                        help='Disable RRT* planner (enabled by default for stage 0/1, always disabled for pretrain)')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Total training timesteps (default: 500k for Stage 0, 1.5M for Stage 1, 500k for pretrain)')
    parser.add_argument('--n-envs', type=int, default=None,
                        help='Number of parallel environments (default: from config)')
    parser.add_argument('--continue', dest='continue_training', action='store_true',
                        help='Continue training from checkpoint')
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
    parser.set_defaults(diag=True)

    args = parser.parse_args()

    # Load config for the specified stage
    config = load_config(args.stage)

    # Enable debug mode if requested
    if args.debug:
        config.DEBUG_MODE = True
        print("[DEBUG] Debug mode enabled")

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

    if swarm_drones < 1:
        parser.error("--swarm-drones must be >= 1")
    if args.diag_sample_every < 1:
        parser.error("--diag-sample-every must be >= 1")
    if args.diag_window < 10:
        parser.error("--diag-window must be >= 10")
    if args.diag_bad_topk < 10:
        parser.error("--diag-bad-topk must be >= 10")

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
    if stage == 'pretrain':
        print(f"  Obstacle type: {args.obstacle_type}")
    print(f"  Use planner: {use_planner}")
    print(f"  Total timesteps: {timesteps:,}")
    print(f"  Parallel envs: {n_envs}")
    print(f"  Watch mode: {watch_mode}")
    print(f"  Fixed map: {fixed_map}")
    print(f"  Show paths: {show_paths}")
    print(f"  Swarm drones: {swarm_drones}")
    print(f"  Diagnostics logs: {args.diag}")
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

    # Check for transfer learning (Stage 0 -> Stage 1)
    stage0_model = "models/ppo_drone_nav_stage0_planner.zip"
    stage0_normalize = "models/vec_normalize_stage0_planner.pkl"
    can_transfer = (stage == 1) and os.path.exists(stage0_model) and os.path.exists(stage0_normalize)

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Create environments
    print(f"\n[SETUP] Creating environments...")
    if stage == '0':
        env_fns = [make_env_stage0(i, config.SEED, use_planner, config,
                                   gui=watch_mode, watch_fps=args.watch_fps,
                                   fixed_map=fixed_map, show_paths=show_paths,
                                   swarm_drones=swarm_drones)
                   for i in range(n_envs)]
    elif stage == '1':
        env_fns = [make_env_stage1(i, config.SEED, use_planner, config,
                                   gui=watch_mode, watch_fps=args.watch_fps,
                                   fixed_map=fixed_map, show_paths=show_paths,
                                   swarm_drones=swarm_drones)
                   for i in range(n_envs)]
    else:  # pretrain
        env_fns = [make_env_pretrain(i, config.SEED, args.obstacle_type, config,
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
    if args.continue_training and checkpoint_exists:
        print(f"\n[LOAD] Continuing training from checkpoint...")
        vec_env = VecNormalize.load(normalize_path, vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
        print("✓ VecNormalize stats loaded")

        model = PPO.load(f"{model_path}.zip", env=vec_env)
        print("✓ Model loaded")
        print(f"\nStarting from {model.num_timesteps} steps")

    elif stage == 1 and can_transfer and not checkpoint_exists:
        print(f"\n[LOAD] Using Stage 0 model for transfer learning...")
        vec_env = VecNormalize.load(stage0_normalize, vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
        print("✓ VecNormalize stats loaded from Stage 0")

        model = PPO.load(stage0_model, env=vec_env)
        print("✓ Stage 0 model loaded for transfer learning")
        print(f"\nStarting from {model.num_timesteps} steps")

    else:
        print(f"\n[SETUP] Starting from scratch...")
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
        print("✓ Environments created")

        print(f"\n[SETUP] Creating PPO model...")
        model = PPO(
            **config.PPO_PARAMS,
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
            "watch_mode": watch_mode,
            "n_envs": int(n_envs),
            "swarm_drones": int(swarm_drones),
            "obstacle_type": args.obstacle_type if stage == "pretrain" else None,
            "continue_training": bool(args.continue_training),
            "timesteps_target": int(timesteps)
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

    print(f"\n[TRAINING] Starting training...")
    print("-" * 60)

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=progress_callback,
            progress_bar=False,
            reset_num_timesteps=False,
            tb_log_name=log_name
        )
        print("\n" + "-" * 60)
        print("✓ Training completed")

    except KeyboardInterrupt:
        print("\n\n[INFO] Training interrupted by user")

    # Save model
    print(f"\n[SAVE] Saving model...")
    model.save(model_path)
    vec_env.save(normalize_path)
    print(f"✓ Model saved to {model_path}.zip")
    print(f"✓ Normalization saved to {normalize_path}")

    if diag_logger is not None:
        diag_logger.close()

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
        print(f"     python visualization/visualize.py --model {model_path} --stage pretrain")
        print("\n  2. Preview scenario generation:")
        print(f"     python visualization/preview_scenarios.py --stage pretrain --obstacle-type {args.obstacle_type}")
        print("\n  3. Train on different obstacle type:")
        print("     python training/train.py --stage pretrain --obstacle-type spheres")


if __name__ == "__main__":
    main()
