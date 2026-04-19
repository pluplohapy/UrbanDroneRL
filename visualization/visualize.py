"""
Visualization script for trained drone navigation model.
Shows the drone flying in PyBullet GUI.
"""

import numpy as np
import time
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from envs.nav_aviary import NavAviary
from scenarios.stage0_empty import Stage0Scenario
from scenarios.stage1_static import Stage1Scenario
from scenarios.stage_pretrain import StagePretrainScenario
from envs.visualization_utils import draw_arena_boundaries, draw_goal_marker
from config import load_config


def visualize_flight(model_path="models/ppo_drone_nav_test",
                     vec_normalize_path="models/vec_normalize_test.pkl",
                     n_episodes=5,
                     deterministic=True,
                     stage=0,
                     obstacle_type='random'):
    """
    Visualize trained model flying in PyBullet GUI.

    Args:
        model_path: Path to saved model
        vec_normalize_path: Path to VecNormalize stats
        n_episodes: Number of episodes to visualize
        deterministic: Use deterministic policy
        stage: 0 for empty, 1 for static obstacles, 'pretrain' for pretrain
        obstacle_type: Obstacle type for pretrain stage
    """
    print("=" * 60)
    print("DRONE NAVIGATION VISUALIZATION")
    print("=" * 60)

    # Create environment with GUI FIRST
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
        env = NavAviary(scenario=scenario, gui=True)  # GUI enabled
        return env

    env = DummyVecEnv([make_env])

    # Disable PyBullet warnings
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=env.envs[0].CLIENT)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=env.envs[0].CLIENT)

    # Load normalization stats
    try:
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
        print("✓ Normalization stats loaded")
    except FileNotFoundError:
        print("⚠ Normalization stats not found, using unnormalized environment")

    # Load model AFTER environment is set up
    print(f"\n[LOAD] Loading model from {model_path}...")
    try:
        model = PPO.load(model_path, env=env)
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

    # Run episodes
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        step = 0
        done = False

        print(f"\nEpisode {episode + 1}/{n_episodes}")
        print("-" * 60)

        # Get config for current stage
        if stage == 'pretrain':
            stage_config = load_config('pretrain')
        elif stage == 1:
            stage_config = load_config('1')
        else:
            stage_config = load_config('0')

        # Draw arena boundaries with correct size
        draw_arena_boundaries(
            env.envs[0].CLIENT,
            stage_config.ARENA_SIZE_X,
            stage_config.ARENA_SIZE_Y,
            stage_config.ARENA_HEIGHT
        )

        # Get goal position for visualization
        goal_pos = env.envs[0].goal_pos

        # Draw goal marker
        draw_goal_marker(goal_pos, env.envs[0].CLIENT)

        # Track trajectory
        trajectory = []
        prev_pos = None

        # Debug tracking
        actions_log = []
        velocities_log = []
        heading_errors_log = []
        distances_log = []

        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=deterministic)
            actions_log.append(action[0].copy())

            # Get current state
            curr_pos = env.envs[0]._getDroneStateVector(0)[:3]
            curr_vel = env.envs[0]._getDroneStateVector(0)[10:13]
            trajectory.append(curr_pos.copy())
            velocities_log.append(curr_vel.copy())

            # Calculate metrics
            dist_to_goal = np.linalg.norm(goal_pos - curr_pos)
            distances_log.append(dist_to_goal)

            # Heading error
            goal_direction = (goal_pos - curr_pos) / (dist_to_goal + 1e-6)
            speed = np.linalg.norm(curr_vel)
            if speed > 0.1:
                vel_direction = curr_vel / speed
                heading_error = np.arccos(np.clip(np.dot(vel_direction, goal_direction), -1, 1))
                heading_errors_log.append(np.degrees(heading_error))
            else:
                heading_errors_log.append(90.0)

            # Update camera to follow drone (third person view)
            p.resetDebugVisualizerCamera(
                cameraDistance=5.0,
                cameraYaw=50,
                cameraPitch=-35,
                cameraTargetPosition=curr_pos,
                physicsClientId=env.envs[0].CLIENT
            )

            # Print detailed info every 30 steps
            if step % 30 == 0:
                print(f"  Step {step:3d} | Pos: [{curr_pos[0]:6.2f}, {curr_pos[1]:6.2f}, {curr_pos[2]:6.2f}] | "
                      f"Dist: {dist_to_goal:5.2f}m | Speed: {speed:4.2f}m/s | Heading: {heading_errors_log[-1]:5.1f}°")
                print(f"           | Action: [{action[0][0]:5.2f}, {action[0][1]:5.2f}, {action[0][2]:5.2f}, {action[0][3]:5.2f}] | "
                      f"Vel: [{curr_vel[0]:5.2f}, {curr_vel[1]:5.2f}, {curr_vel[2]:5.2f}]")

            # Draw trajectory line
            if prev_pos is not None:
                p.addUserDebugLine(
                    prev_pos,
                    curr_pos,
                    [1, 0, 0],  # Red color
                    5,
                    0,
                    physicsClientId=env.envs[0].CLIENT
                )
            prev_pos = curr_pos

            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            step += 1

            # Slow down for visualization
            time.sleep(0.03)  # ~30 FPS

            if done[0]:
                result = "SUCCESS ✓" if info[0].get("is_success", False) else \
                         "CRASH ✗" if info[0].get("is_crash", False) else \
                         "TIMEOUT"

                # Calculate episode statistics
                min_dist = min(distances_log)
                avg_speed = np.mean([np.linalg.norm(v) for v in velocities_log])
                avg_heading = np.mean(heading_errors_log) if heading_errors_log else 0

                # Path efficiency
                straight_dist = np.linalg.norm(goal_pos - trajectory[0])
                path_length = sum(np.linalg.norm(trajectory[i] - trajectory[i-1])
                                 for i in range(1, len(trajectory)))
                path_efficiency = straight_dist / (path_length + 1e-6)

                # Action statistics
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

                # Pause between episodes
                if episode < n_episodes - 1:
                    print("\n  Next episode in 2 seconds...")
                    time.sleep(2)

    env.close()
    print("\n" + "=" * 60)
    print("VISUALIZATION FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize trained drone navigation")
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
                        help="Obstacle type for pretrain stage")

    args = parser.parse_args()

    # Convert stage to appropriate type
    if args.stage == 'pretrain':
        stage = 'pretrain'
    else:
        stage = int(args.stage)

    visualize_flight(
        model_path=args.model,
        vec_normalize_path=args.normalize,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        stage=stage,
        obstacle_type=args.obstacle_type
    )
