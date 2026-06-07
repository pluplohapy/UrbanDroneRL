
import numpy as np
import time
import os
import sys
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.nav_aviary import NavAviary
from scenarios.stage0_empty import Stage0Scenario
from scenarios.stage1_static import Stage1Scenario
from scenarios.stage_pretrain import StagePretrainScenario
from envs.visualization_utils import draw_arena_boundaries, draw_goal_marker
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

def _is_pybullet_disconnect_error(exc: Exception) -> bool:
    return "Not connected to physics server" in str(exc)

def _visualization_client_connected(env) -> bool:
    try:
        return bool(p.isConnected(env.envs[0].CLIENT))
    except Exception:
        return False

def visualize_flight(model_path="models/ppo_drone_nav_test",
                     vec_normalize_path="models/vec_normalize_test.pkl",
                     n_episodes=5,
                     deterministic=True,
                     algo="ppo",
                     stage=0,
                     obstacle_type='random',
                     watch_fps=60.0,
                     show_paths=True,
                     safety_shield=True):
    print("=" * 60)
    print("DRONE NAVIGATION VISUALIZATION")
    print("=" * 60)
    print(f"[CONFIG] algo={algo}, watch_fps={watch_fps}, show_paths={show_paths}, safety_shield={safety_shield}")
    if watch_fps <= 0:
        raise ValueError("watch_fps must be > 0")

    stage_key = 'pretrain' if stage == 'pretrain' else str(stage)
    stage_config = load_config(stage_key)
    if stage == 'pretrain':
        stage_config = apply_pretrain_obstacle_overrides(stage_config, obstacle_type)

    stage_config.SAFETY_SHIELD_ENABLED = bool(safety_shield)
    sync_runtime_config(stage_config)

    print("\n[SETUP] Creating visualization environment...")

    def make_env():
        if stage == 'pretrain':
            scenario = StagePretrainScenario(obstacle_type=obstacle_type, seed=42)
            print(f"✓ Using Pretrain stage (obstacle_type={obstacle_type})")
        elif stage == 1:
            scenario = Stage1Scenario(seed=42)
            print("✓ Using Stage 1 (static obstacles)")
        else:
            scenario = Stage0Scenario(seed=42)
            print("✓ Using Stage 0 (empty)")
        env = NavAviary(
            scenario=scenario,
            gui=True,
            watch_fps=watch_fps,
            show_trajectory=show_paths
        )
        return env

    env = DummyVecEnv([make_env])

    try:
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
        print("✓ Normalization stats loaded")
    except FileNotFoundError:
        print("⚠ Normalization stats not found, using unnormalized environment")

    print(f"\n[LOAD] Loading model from {model_path}...")
    try:
        model = get_algorithm_class(algo).load(model_path, env=env)
        print("✓ Model loaded")
    except FileNotFoundError:
        print(f"✗ Model not found at {model_path}")
        print("Please train the model first with: python train_test.py")
        env.close()
        return

    print("\n[INFO] Controls:")
    print("  - Close PyBullet window to stop")
    print("  - Watch the drone navigate to the goal")
    print("\n" + "-" * 60)

    stop_visualization = False
    for episode in range(n_episodes):
        if stop_visualization:
            break

        obs = env.reset()
        episode_reward = 0
        step = 0
        done = False

        print(f"\nEpisode {episode + 1}/{n_episodes}")
        print("-" * 60)

        draw_arena_boundaries(
            env.envs[0].CLIENT,
            stage_config.ARENA_SIZE_X,
            stage_config.ARENA_SIZE_Y,
            stage_config.ARENA_HEIGHT
        )

        goal_pos = env.envs[0].goal_pos

        draw_goal_marker(goal_pos, env.envs[0].CLIENT)

        trajectory = []

        actions_log = []
        velocities_log = []
        heading_errors_log = []
        distances_log = []
        lstm_states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)

        while not done:
            if not _visualization_client_connected(env):
                print("\n[INFO] PyBullet window closed; stopping visualization.")
                stop_visualization = True
                break

            action, lstm_states = predict_with_optional_state(
                model,
                obs,
                deterministic=deterministic,
                lstm_states=lstm_states,
                episode_starts=episode_starts
            )
            actions_log.append(action[0].copy())

            try:
                curr_pos = env.envs[0]._getDroneStateVector(0)[:3]
                curr_vel = env.envs[0]._getDroneStateVector(0)[10:13]
            except p.error as exc:
                if _is_pybullet_disconnect_error(exc):
                    print("\n[INFO] PyBullet window closed; stopping visualization.")
                    stop_visualization = True
                    break
                raise
            trajectory.append(curr_pos.copy())
            velocities_log.append(curr_vel.copy())

            dist_to_goal = np.linalg.norm(goal_pos - curr_pos)
            distances_log.append(dist_to_goal)

            goal_direction = (goal_pos - curr_pos) / (dist_to_goal + 1e-6)
            speed = np.linalg.norm(curr_vel)
            if speed > 0.1:
                vel_direction = curr_vel / speed
                heading_error = np.arccos(np.clip(np.dot(vel_direction, goal_direction), -1, 1))
                heading_errors_log.append(np.degrees(heading_error))
            else:
                heading_errors_log.append(90.0)

            if step % 30 == 0:
                print(f"  Step {step:3d} | Pos: [{curr_pos[0]:6.2f}, {curr_pos[1]:6.2f}, {curr_pos[2]:6.2f}] | "
                      f"Dist: {dist_to_goal:5.2f}m | Speed: {speed:4.2f}m/s | Heading: {heading_errors_log[-1]:5.1f}°")
                print(f"           | Action: [{action[0][0]:5.2f}, {action[0][1]:5.2f}, {action[0][2]:5.2f}, {action[0][3]:5.2f}] | "
                      f"Vel: [{curr_vel[0]:5.2f}, {curr_vel[1]:5.2f}, {curr_vel[2]:5.2f}]")

            try:
                step_result = env.step(action)
            except p.error as exc:
                if _is_pybullet_disconnect_error(exc):
                    print("\n[INFO] PyBullet window closed; stopping visualization.")
                    stop_visualization = True
                    break
                raise

            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = [terminated[0] or truncated[0]]
            else:
                obs, reward, done, info = step_result
                terminated = done
                truncated = [False]

            episode_reward += reward[0]
            step += 1
            episode_starts = np.array(done, dtype=bool)

            if done[0]:

                final_pos = np.array(info[0].get("final_pos", curr_pos), dtype=float)
                pos_after_step = final_pos

                is_success = info[0].get("is_success", False)
                is_crash = info[0].get("is_crash", False)
                has_contact = bool(info[0].get("has_contact", False))

                arena_x = env.envs[0].arena_size_x / 2
                arena_y = env.envs[0].arena_size_y / 2
                arena_z = env.envs[0].arena_height

                out_of_bounds_before = (
                    abs(final_pos[0]) > arena_x or
                    abs(final_pos[1]) > arena_y or
                    final_pos[2] < 0.1 or
                    final_pos[2] > arena_z
                )

                out_of_bounds_after = bool(info[0].get("out_of_bounds", False))

                print(f"\n  [DEBUG] Pos BEFORE step: [{final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f}]")
                print(f"  [DEBUG] Pos AFTER step:  [{pos_after_step[0]:.2f}, {pos_after_step[1]:.2f}, {pos_after_step[2]:.2f}]")
                print(f"  [DEBUG] Arena bounds: X=±{arena_x:.1f}, Y=±{arena_y:.1f}, Z=0.1-{arena_z:.1f}")
                print(f"  [DEBUG] Out of bounds BEFORE: {out_of_bounds_before}")
                print(f"  [DEBUG] Out of bounds AFTER: {out_of_bounds_after}")
                print(f"  [DEBUG] is_crash={is_crash}, is_success={is_success}, has_contact={has_contact}, step={step}")

                out_of_bounds = out_of_bounds_before or out_of_bounds_after

                if is_success:
                    result = "SUCCESS ✓"
                elif is_crash or out_of_bounds:
                    result = "CRASH ✗"
                    if out_of_bounds:
                        result += f" (OUT OF BOUNDS: pos=[{final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f}])"
                else:
                    result = "TIMEOUT"

                min_dist = min(distances_log)
                avg_speed = np.mean([np.linalg.norm(v) for v in velocities_log])
                avg_heading = np.mean(heading_errors_log) if heading_errors_log else 0

                straight_dist = np.linalg.norm(goal_pos - trajectory[0])
                path_length = sum(np.linalg.norm(trajectory[i] - trajectory[i-1])
                                 for i in range(1, len(trajectory)))
                path_efficiency = straight_dist / (path_length + 1e-6)

                actions_array = np.array(actions_log)
                action_mean = np.mean(actions_array, axis=0)
                action_std = np.std(actions_array, axis=0)
                action_smoothness = np.mean([np.linalg.norm(actions_log[i] - actions_log[i-1])
                                            for i in range(1, len(actions_log))]) if len(actions_log) > 1 else 0

                print(f"\n  Result: {result}")
                print(f"  Steps: {step}")
                print(f"  Episode reward: {episode_reward:.2f}")
                print(f"  Final distance: {info[0].get('dist_to_goal', 0):.2f}m")
                print(f"  Min distance: {min_dist:.2f}m")
                print(f"  Avg speed: {avg_speed:.2f}m/s")
                print(f"  Avg heading error: {avg_heading:.1f}°")
                print(f"  Path efficiency: {path_efficiency:.2f}")
                print(f"  Action mean: [{action_mean[0]:.2f}, {action_mean[1]:.2f}, {action_mean[2]:.2f}, {action_mean[3]:.2f}]")
                print(f"  Action std: [{action_std[0]:.2f}, {action_std[1]:.2f}, {action_std[2]:.2f}, {action_std[3]:.2f}]")
                print(f"  Action smoothness: {action_smoothness:.3f}")
                print(f"  Trajectory length: {len(trajectory)} points")

                if episode < n_episodes - 1:
                    print("\n  Next episode in 2 seconds...")
                    time.sleep(2)

    try:
        env.close()
    except p.error as exc:
        if not _is_pybullet_disconnect_error(exc):
            raise
    print("\n" + "=" * 60)
    print("VISUALIZATION FINISHED")
    print("=" * 60)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize trained drone navigation")
    parser.add_argument("--algo", type=str, default="ppo", choices=ALGO_CHOICES,
                        help="Policy optimizer used by the checkpoint")
    parser.add_argument("--model", type=str, default="models/ppo_drone_nav_test",
                        help="Path to trained model")
    parser.add_argument("--normalize", type=str, default="models/vec_normalize_test.pkl",
                        help="Path to VecNormalize stats")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to visualize")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy instead of deterministic")
    parser.add_argument("--stage", type=str, default="0",
                        help="Stage: 0=empty, 1=static obstacles, pretrain=pretrain")
    parser.add_argument("--obstacle-type", type=str, default="random",
                        choices=PRETRAIN_OBSTACLE_CHOICES,
                        help="Obstacle type for pretrain stage")
    parser.add_argument("--watch-fps", type=float, default=60.0,
                        help="Target GUI FPS (same as train --watch-fps)")
    parser.add_argument("--show-paths", action="store_true",
                        help="Draw trajectory lines in GUI")
    parser.add_argument("--no-show-paths", action="store_true",
                        help="Disable trajectory lines in GUI")
    parser.add_argument("--safety-shield", dest="safety_shield", action="store_true",
                        help="Enable safety shield during visualization (default)")
    parser.add_argument("--no-safety-shield", dest="safety_shield", action="store_false",
                        help="Disable safety shield during visualization")
    parser.set_defaults(safety_shield=True)

    args = parser.parse_args()
    show_paths = args.show_paths or (not args.no_show_paths)

    if args.stage == 'pretrain':
        stage = 'pretrain'
    else:
        stage = int(args.stage)

    visualize_flight(
        model_path=args.model,
        vec_normalize_path=args.normalize,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        algo=args.algo,
        stage=stage,
        obstacle_type=args.obstacle_type,
        watch_fps=args.watch_fps,
        show_paths=show_paths,
        safety_shield=args.safety_shield
    )
