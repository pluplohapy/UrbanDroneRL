
import argparse
import json
import os
import sys
from typing import Any
from collections import defaultdict

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.nav_aviary import NavAviary
from envs.nav_aviary_planner import NavAviaryWithPlanner
from scenarios.stage0_empty import Stage0Scenario
from scenarios.stage1_static import Stage1Scenario
from scenarios.stage_pretrain import StagePretrainScenario
from config import load_config, apply_pretrain_obstacle_overrides
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
    "city_dynamic",
    "construction_site_dynamic",
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

def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    except Exception:
        return None

def _build_env(stage: str, obstacle_type: str, seed: int, use_planner: bool, config):
    if stage == "pretrain":
        if use_planner:
            raise ValueError("Planner mode is not supported for pretrain stage")
        scenario = StagePretrainScenario(obstacle_type=obstacle_type, seed=seed)
        return NavAviary(scenario=scenario, gui=False)

    if stage == "1":
        scenario = Stage1Scenario(seed=seed)
    else:
        scenario = Stage0Scenario(seed=seed)

    if use_planner:
        return NavAviaryWithPlanner(
            scenario=scenario,
            gui=False,
            use_planner=True,
            replan_freq=0,
            waypoint_threshold=config.WAYPOINT_THRESHOLD,
            planner_params={
                "max_iter": config.RRT_MAX_ITER,
                "step_size": config.RRT_STEP_SIZE,
                "goal_bias": config.RRT_GOAL_BIAS,
                "rewire_radius": config.RRT_REWIRE_RADIUS,
                "collision_check_resolution": config.RRT_COLLISION_RESOLUTION,
                "verbose": 0
            }
        )

    return NavAviary(scenario=scenario, gui=False)

def _summarize(records, timeout_near_goal_threshold: float):
    total = len(records)
    success = sum(1 for r in records if r["outcome"] == "success")
    crash = sum(1 for r in records if r["outcome"] == "crash")
    timeout = sum(1 for r in records if r["outcome"] == "timeout")

    crash_oob = sum(1 for r in records if r["crash_reason"] == "out_of_bounds")
    crash_contact = sum(1 for r in records if r["crash_reason"] == "contact")
    crash_other = sum(1 for r in records if r["crash_reason"] == "other")
    timeout_near_goal = sum(
        1
        for r in records
        if r["outcome"] == "timeout"
        and r["min_goal_dist"] is not None
        and r["min_goal_dist"] <= timeout_near_goal_threshold
    )
    crash_near_goal = sum(
        1
        for r in records
        if r["outcome"] == "crash"
        and r["min_goal_dist"] is not None
        and r["min_goal_dist"] <= timeout_near_goal_threshold
    )
    oob_near_goal = sum(
        1
        for r in records
        if r["crash_reason"] == "out_of_bounds"
        and r["min_goal_dist"] is not None
        and r["min_goal_dist"] <= timeout_near_goal_threshold
    )

    steps = [r["steps"] for r in records]
    rewards = [r["episode_reward"] for r in records if r["episode_reward"] is not None]
    final_dist = [r["final_dist"] for r in records if r["final_dist"] is not None]
    min_goal_dist = [r["min_goal_dist"] for r in records if r["min_goal_dist"] is not None]

    def _rate(count: int) -> float:
        return float(count / total) if total > 0 else 0.0

    summary = {
        "episodes": total,
        "success": success,
        "crash": crash,
        "timeout": timeout,
        "success_rate": _rate(success),
        "crash_rate": _rate(crash),
        "timeout_rate": _rate(timeout),
        "crash_breakdown": {
            "out_of_bounds": crash_oob,
            "contact": crash_contact,
            "other": crash_other
        },
        "timeout_near_goal": timeout_near_goal,
        "crash_near_goal": crash_near_goal,
        "out_of_bounds_near_goal": oob_near_goal,
        "means": {
            "steps": float(np.mean(steps)) if steps else None,
            "reward": float(np.mean(rewards)) if rewards else None,
            "final_dist": float(np.mean(final_dist)) if final_dist else None,
            "min_goal_dist": float(np.mean(min_goal_dist)) if min_goal_dist else None
        }
    }

    by_obstacle_type = {}
    grouped = defaultdict(list)
    for record in records:
        grouped[record.get("obstacle_type") or "unknown"].append(record)
    if len(grouped) > 1 or "unknown" not in grouped:
        for obstacle_type, group_records in sorted(grouped.items()):
            by_obstacle_type[obstacle_type] = _summarize_basic(group_records)
    summary["by_obstacle_type"] = by_obstacle_type
    return summary

def _summarize_basic(records):
    total = len(records)
    if total == 0:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "crash_rate": 0.0,
            "timeout_rate": 0.0,
            "crash_breakdown": {"out_of_bounds": 0, "contact": 0, "other": 0},
            "mean_steps": None,
            "mean_final_dist": None,
        }

    success = sum(1 for r in records if r["outcome"] == "success")
    crash = sum(1 for r in records if r["outcome"] == "crash")
    timeout = sum(1 for r in records if r["outcome"] == "timeout")
    final_dist = [r["final_dist"] for r in records if r["final_dist"] is not None]
    return {
        "episodes": total,
        "success_rate": float(success / total),
        "crash_rate": float(crash / total),
        "timeout_rate": float(timeout / total),
        "crash_breakdown": {
            "out_of_bounds": sum(1 for r in records if r["crash_reason"] == "out_of_bounds"),
            "contact": sum(1 for r in records if r["crash_reason"] == "contact"),
            "other": sum(1 for r in records if r["crash_reason"] == "other"),
        },
        "mean_steps": float(np.mean([r["steps"] for r in records])),
        "mean_final_dist": float(np.mean(final_dist)) if final_dist else None,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained drone navigation policy")
    parser.add_argument("--algo", type=str, default="ppo", choices=ALGO_CHOICES,
                        help="Policy optimizer used by the checkpoint")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model (.zip can be omitted)")
    parser.add_argument("--normalize", type=str, default=None,
                        help="Path to VecNormalize stats (.pkl)")
    parser.add_argument("--stage", type=str, default="pretrain", choices=["0", "1", "pretrain"],
                        help="Evaluation stage")
    parser.add_argument("--obstacle-type", type=str, default="random",
                        choices=PRETRAIN_OBSTACLE_CHOICES,
                        help="Obstacle type for pretrain stage")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Base seed for scenario RNG")
    parser.add_argument("--planner", action="store_true",
                        help="Use planner environment for stage 0/1")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy instead of deterministic")
    parser.add_argument("--safety-shield", dest="safety_shield", action="store_true",
                        help="Enable safety shield during evaluation")
    parser.add_argument("--no-safety-shield", dest="safety_shield", action="store_false",
                        help="Disable safety shield during evaluation (default)")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Optional path to save per-episode records and summary as JSON")
    parser.set_defaults(safety_shield=False)

    args = parser.parse_args()
    algo_cls = get_algorithm_class(args.algo)

    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    if args.stage == "pretrain" and args.planner:
        parser.error("--planner is not supported for stage pretrain")

    config = load_config(args.stage)
    if args.stage == "pretrain":
        config = apply_pretrain_obstacle_overrides(config, args.obstacle_type)
    shield_enabled = bool(args.safety_shield)
    config.SAFETY_SHIELD_ENABLED = shield_enabled
    sync_runtime_config(config)

    print("=" * 60)
    print("DRONE NAVIGATION EVALUATION")
    print("=" * 60)
    print(f"[CONFIG] algo={args.algo}, stage={args.stage}, obstacle_type={args.obstacle_type}, planner={args.planner}")
    print(f"[CONFIG] episodes={args.episodes}, deterministic={not args.stochastic}, safety_shield={shield_enabled}")

    env = DummyVecEnv([
        lambda: _build_env(
            stage=args.stage,
            obstacle_type=args.obstacle_type,
            seed=int(args.seed),
            use_planner=bool(args.planner),
            config=config
        )
    ])

    if args.normalize:
        if os.path.exists(args.normalize):
            env = VecNormalize.load(args.normalize, env)
            env.training = False
            env.norm_reward = False
            print(f"✓ Normalization loaded: {args.normalize}")
        else:
            print(f"⚠ Normalization file not found, continuing without it: {args.normalize}")

    model = algo_cls.load(args.model, env=env)
    print(f"✓ Model loaded: {args.model}")

    records = []
    deterministic = not args.stochastic

    def _fmt(value: float | None, pattern: str) -> str:
        if value is None:
            return "n/a"
        return format(value, pattern)

    try:
        for episode_idx in range(1, args.episodes + 1):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            steps = 0
            last_info = {}
            lstm_states = None
            episode_starts = np.ones((env.num_envs,), dtype=bool)

            while not done:
                action, lstm_states = predict_with_optional_state(
                    model,
                    obs,
                    deterministic=deterministic,
                    lstm_states=lstm_states,
                    episode_starts=episode_starts
                )
                step_result = env.step(action)

                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = bool(terminated[0] or truncated[0])
                else:
                    obs, reward, done_vec, info = step_result
                    done = bool(done_vec[0])

                ep_reward += float(reward[0])
                steps += 1
                if info and len(info) > 0:
                    last_info = info[0]
                episode_starts = np.array([done], dtype=bool)

            is_success = bool(last_info.get("is_success", False))
            is_crash = bool(last_info.get("is_crash", False))
            out_of_bounds = bool(last_info.get("out_of_bounds", False))
            has_contact = bool(last_info.get("has_contact", False))

            if is_success:
                outcome = "success"
                crash_reason = "none"
            elif is_crash or out_of_bounds:
                outcome = "crash"
                if out_of_bounds:
                    crash_reason = "out_of_bounds"
                elif has_contact:
                    crash_reason = "contact"
                else:
                    crash_reason = "other"
            else:
                outcome = "timeout"
                crash_reason = "none"

            final_dist = _safe_float(last_info.get("dist_to_goal"))
            min_goal_dist = _safe_float(last_info.get("min_goal_distance"))

            record = {
                "episode": int(episode_idx),
                "obstacle_type": last_info.get("obstacle_type"),
                "obstacle_count": last_info.get("obstacle_count"),
                "outcome": outcome,
                "crash_reason": crash_reason,
                "steps": int(steps),
                "episode_reward": _safe_float(ep_reward),
                "final_dist": final_dist,
                "min_goal_dist": min_goal_dist,
                "out_of_bounds": out_of_bounds,
                "has_contact": has_contact
            }
            records.append(record)

            result_label = outcome.upper()
            if outcome == "crash":
                result_label = f"CRASH/{crash_reason}"
            print(
                f"EP {episode_idx:03d} | {result_label:16s} | "
                f"map={str(record['obstacle_type'] or 'unknown'):15s} | "
                f"steps={steps:4d} | reward={ep_reward:9.2f} | "
                f"final_dist={final_dist if final_dist is not None else float('nan'):.2f}"
            )

        summary = _summarize(records, timeout_near_goal_threshold=float(config.TIMEOUT_NEAR_GOAL_THRESHOLD))

        print("\n" + "-" * 60)
        print("SUMMARY")
        print("-" * 60)
        print(
            f"Success: {summary['success']}/{summary['episodes']} ({summary['success_rate']:.1%}) | "
            f"Crash: {summary['crash']}/{summary['episodes']} ({summary['crash_rate']:.1%}) | "
            f"Timeout: {summary['timeout']}/{summary['episodes']} ({summary['timeout_rate']:.1%})"
        )
        print(
            "Crash breakdown: "
            f"out_of_bounds={summary['crash_breakdown']['out_of_bounds']}, "
            f"contact={summary['crash_breakdown']['contact']}, "
            f"other={summary['crash_breakdown']['other']}"
        )
        print(
            f"Timeout near goal (<= {float(config.TIMEOUT_NEAR_GOAL_THRESHOLD):.2f}m): "
            f"{summary['timeout_near_goal']}"
        )
        print(
            f"Crash near goal (<= {float(config.TIMEOUT_NEAR_GOAL_THRESHOLD):.2f}m): "
            f"{summary['crash_near_goal']} | oob={summary['out_of_bounds_near_goal']}"
        )
        print(
            f"Mean steps: {_fmt(summary['means']['steps'], '.1f')} | "
            f"Mean reward: {_fmt(summary['means']['reward'], '.2f')} | "
            f"Mean final dist: {_fmt(summary['means']['final_dist'], '.2f')} | "
            f"Mean min dist: {_fmt(summary['means']['min_goal_dist'], '.2f')}"
        )
        if summary.get("by_obstacle_type"):
            print("\nBy obstacle type:")
            for obstacle_type, item in summary["by_obstacle_type"].items():
                print(
                    f"  {obstacle_type:15s} | n={item['episodes']:3d} | "
                    f"S={item['success_rate']:.1%} | C={item['crash_rate']:.1%} | "
                    f"contact={item['crash_breakdown']['contact']:3d} | "
                    f"oob={item['crash_breakdown']['out_of_bounds']:3d} | "
                    f"steps={_fmt(item['mean_steps'], '.1f')} | "
                    f"final={_fmt(item['mean_final_dist'], '.2f')}"
                )

        if args.json_out:
            output_payload = {
                "config": {
                    "model": args.model,
                    "algo": args.algo,
                    "normalize": args.normalize,
                    "stage": args.stage,
                    "obstacle_type": args.obstacle_type,
                    "episodes": args.episodes,
                    "seed": int(args.seed),
                    "planner": bool(args.planner),
                    "deterministic": bool(deterministic),
                    "safety_shield": bool(shield_enabled)
                },
                "summary": summary,
                "episodes": records
            }
            output_dir = os.path.dirname(args.json_out)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as fh:
                json.dump(output_payload, fh, ensure_ascii=False, indent=2)
            print(f"✓ Saved JSON report: {args.json_out}")

    finally:
        env.close()

if __name__ == "__main__":
    main()
