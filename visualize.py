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
import config


def visualize_flight(model_path="models/ppo_drone_nav_test",
                     vec_normalize_path="models/vec_normalize_test.pkl",
                     n_episodes=5,
                     deterministic=True):
    """
    Visualize trained model flying in PyBullet GUI.

    Args:
        model_path: Path to saved model
        vec_normalize_path: Path to VecNormalize stats
        n_episodes: Number of episodes to visualize
        deterministic: Use deterministic policy
    """
    print("=" * 60)
    print("DRONE NAVIGATION VISUALIZATION")
    print("=" * 60)

    # Load model
    print(f"\n[LOAD] Loading model from {model_path}...")
    try:
        model = PPO.load(model_path)
        print("✓ Model loaded")
    except FileNotFoundError:
        print(f"✗ Model not found at {model_path}")
        print("Please train the model first with: python train_test.py")
        return

    # Create environment with GUI
    print("\n[SETUP] Creating visualization environment...")

    def make_env():
        scenario = Stage0Scenario(seed=42)
        env = NavAviary(scenario=scenario, gui=True)  # GUI enabled
        return env

    env = DummyVecEnv([make_env])

    # Load normalization stats
    try:
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
        print("✓ Normalization stats loaded")
    except FileNotFoundError:
        print("⚠ Normalization stats not found, using unnormalized environment")

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

        # Get goal position for visualization
        goal_pos = env.envs[0].goal_pos

        # Draw goal marker in PyBullet
        p.addUserDebugLine(
            [goal_pos[0] - 0.2, goal_pos[1], goal_pos[2]],
            [goal_pos[0] + 0.2, goal_pos[1], goal_pos[2]],
            [0, 1, 0], 3, 0,
            physicsClientId=env.envs[0].CLIENT
        )
        p.addUserDebugLine(
            [goal_pos[0], goal_pos[1] - 0.2, goal_pos[2]],
            [goal_pos[0], goal_pos[1] + 0.2, goal_pos[2]],
            [0, 1, 0], 3, 0,
            physicsClientId=env.envs[0].CLIENT
        )
        p.addUserDebugLine(
            [goal_pos[0], goal_pos[1], goal_pos[2] - 0.2],
            [goal_pos[0], goal_pos[1], goal_pos[2] + 0.2],
            [0, 1, 0], 3, 0,
            physicsClientId=env.envs[0].CLIENT
        )

        # Track trajectory
        trajectory = []
        prev_pos = None

        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=deterministic)

            # Get current position
            curr_pos = env.envs[0]._getDroneStateVector(0)[:3]
            trajectory.append(curr_pos.copy())

            # Draw trajectory line
            if prev_pos is not None:
                p.addUserDebugLine(
                    prev_pos,
                    curr_pos,
                    [1, 0, 0],  # Red color
                    2,  # Line width
                    0,  # Line lifetime (0 = permanent)
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

                print(f"  Result: {result}")
                print(f"  Steps: {step}")
                print(f"  Reward: {episode_reward:.2f}")
                print(f"  Final distance: {info[0].get('dist_to_goal', 0):.2f}m")
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

    args = parser.parse_args()

    visualize_flight(
        model_path=args.model,
        vec_normalize_path=args.normalize,
        n_episodes=args.episodes,
        deterministic=not args.stochastic
    )
