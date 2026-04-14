"""
Visualize multiple flights on a fixed map to show trajectory distribution.
Shows heatmap and 3D overlay of many episodes.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from envs.nav_aviary import NavAviary
from scenarios.stage1_static import Stage1Scenario
import config


def collect_trajectories(model_path, vec_normalize_path, n_episodes=100, seed=42):
    """
    Collect multiple trajectories on the same fixed map.

    Args:
        model_path: Path to trained model
        vec_normalize_path: Path to VecNormalize stats
        n_episodes: Number of episodes to run
        seed: Fixed seed for map generation

    Returns:
        all_trajectories: List of trajectory arrays
        all_infos: List of episode info dicts
        start_pos: Start position
        goal_pos: Goal position
        obstacles: List of (position, radius) tuples
    """
    import pybullet as p

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # Create environment WITHOUT vectorization to avoid auto-reset
    scenario = Stage1Scenario(seed=seed)
    env = NavAviary(scenario=scenario, gui=False)

    # Load normalization stats manually
    vec_normalize = VecNormalize.load(vec_normalize_path, DummyVecEnv([lambda: env]))
    vec_normalize.training = False
    vec_normalize.norm_reward = False

    print(f"Collecting {n_episodes} episodes on fixed map (seed={seed})...")

    # First reset to create the map
    obs, _ = env.reset()
    start_pos = env.start_pos.copy()
    goal_pos = env.goal_pos.copy()
    obstacles = [(obs_obj.position.copy(), obs_obj.radius)
                 for obs_obj in env.scenario.obstacles]

    print(f"Map configuration:")
    print(f"  Start: [{start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}]")
    print(f"  Goal:  [{goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f}]")
    print(f"  Obstacles: {len(obstacles)}")

    all_trajectories = []
    all_infos = []

    for episode in range(n_episodes):
        # Manual reset: only reset drone position, keep obstacles
        env.start_pos = start_pos.copy()
        env.goal_pos = goal_pos.copy()
        env.INIT_XYZS = np.array([start_pos])
        env.INIT_RPYS = np.array([[0, 0, 0]])

        # Reset PyBullet drone state
        p.resetBasePositionAndOrientation(
            env.DRONE_IDS[0],
            start_pos,
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=env.CLIENT
        )
        p.resetBaseVelocity(
            env.DRONE_IDS[0],
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0],
            physicsClientId=env.CLIENT
        )

        # Reset internal state
        env.control_step_counter = 0
        env.prev_dist_to_goal = np.linalg.norm(goal_pos - start_pos)
        env.prev_action = np.zeros(4)
        env.visited_cells = set()

        # Get normalized observation
        obs = env._computeObs()
        obs_normalized = vec_normalize.normalize_obs(obs.reshape(1, -1))[0]

        # Collect trajectory - use start_pos directly since we just reset to it
        trajectory = [start_pos.copy()]
        step = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and step < config.MAX_STEPS:
            # Use deterministic policy for best performance
            action, _ = model.predict(obs_normalized, deterministic=True)

            # Step environment directly (no vectorization)
            obs, reward, terminated, truncated, info = env.step(action)
            obs_normalized = vec_normalize.normalize_obs(obs.reshape(1, -1))[0]

            pos = env._getDroneStateVector(0)[:3]
            trajectory.append(pos.copy())
            step += 1

        all_trajectories.append(np.array(trajectory))
        all_infos.append(info)

        if (episode + 1) % 10 == 0:
            success_rate = sum(i['is_success'] for i in all_infos) / len(all_infos)
            print(f"  Episode {episode + 1}/{n_episodes} | Success rate: {success_rate:.1%}")

    # Close environment
    env.close()

    # Final statistics
    n_success = sum(i['is_success'] for i in all_infos)
    n_crash = sum(i['is_crash'] for i in all_infos)
    n_timeout = n_episodes - n_success - n_crash

    print(f"\nFinal statistics:")
    print(f"  Success: {n_success}/{n_episodes} ({n_success/n_episodes:.1%})")
    print(f"  Crash:   {n_crash}/{n_episodes} ({n_crash/n_episodes:.1%})")
    print(f"  Timeout: {n_timeout}/{n_episodes} ({n_timeout/n_episodes:.1%})")

    return all_trajectories, all_infos, start_pos, goal_pos, obstacles


def plot_heatmap(all_trajectories, start_pos, goal_pos, obstacles, output_path):
    """
    Plot 2D heatmap of trajectory density (top view).
    """
    print(f"\nGenerating heatmap...")

    grid_size = 100
    heatmap = np.zeros((grid_size, grid_size))

    for traj in all_trajectories:
        for pos in traj:
            x_idx = int((pos[0] + config.ARENA_SIZE_X / 2) / config.ARENA_SIZE_X * grid_size)
            y_idx = int((pos[1] + config.ARENA_SIZE_Y / 2) / config.ARENA_SIZE_Y * grid_size)
            if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
                heatmap[y_idx, x_idx] += 1

    plt.figure(figsize=(12, 10))

    # Plot heatmap
    extent = [-config.ARENA_SIZE_X/2, config.ARENA_SIZE_X/2,
              -config.ARENA_SIZE_Y/2, config.ARENA_SIZE_Y/2]
    plt.imshow(heatmap, cmap='YlOrRd', extent=extent, origin='lower', aspect='auto')
    plt.colorbar(label='Trajectory Density')

    # Plot obstacles
    for obs_pos, obs_radius in obstacles:
        circle = plt.Circle((obs_pos[0], obs_pos[1]), obs_radius,
                           color='blue', alpha=0.3, linewidth=2, fill=True)
        plt.gca().add_patch(circle)

    # Plot start/goal
    plt.scatter(start_pos[0], start_pos[1], c='green', s=200,
               marker='o', edgecolors='black', linewidths=2, label='Start', zorder=5)
    plt.scatter(goal_pos[0], goal_pos[1], c='red', s=200,
               marker='*', edgecolors='black', linewidths=2, label='Goal', zorder=5)

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'Trajectory Density Heatmap ({len(all_trajectories)} episodes)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    plt.close()


def plot_2d_trajectories(all_trajectories, all_infos, start_pos, goal_pos, obstacles, output_path):
    """
    Plot 2D overlay of all trajectories (top view) colored by outcome.
    Green = success, Red = crash, Gray = timeout.
    """
    print(f"\nGenerating 2D trajectory overlay...")

    plt.figure(figsize=(14, 12))

    # Plot trajectories by outcome
    for traj, info in zip(all_trajectories, all_infos):
        if info['is_success']:
            color = 'green'
            alpha = 0.5
            linewidth = 1.5
            zorder = 3
        elif info['is_crash']:
            color = 'red'
            alpha = 0.5
            linewidth = 1.5
            zorder = 2
        else:  # timeout
            color = 'gray'
            alpha = 0.3
            linewidth = 1.0
            zorder = 1

        plt.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha,
                linewidth=linewidth, zorder=zorder)

    # Plot obstacles
    for obs_pos, obs_radius in obstacles:
        circle = plt.Circle((obs_pos[0], obs_pos[1]), obs_radius,
                           color='blue', alpha=0.4, linewidth=2, fill=True, zorder=4)
        plt.gca().add_patch(circle)

    # Plot start/goal
    plt.scatter(start_pos[0], start_pos[1], c='green', s=300,
               marker='o', edgecolors='black', linewidths=3, label='Start', zorder=10)
    plt.scatter(goal_pos[0], goal_pos[1], c='red', s=300,
               marker='*', edgecolors='black', linewidths=3, label='Goal', zorder=10)

    # Count outcomes
    n_success = sum(i['is_success'] for i in all_infos)
    n_crash = sum(i['is_crash'] for i in all_infos)
    n_timeout = len(all_infos) - n_success - n_crash

    plt.xlabel('X (m)', fontsize=12)
    plt.ylabel('Y (m)', fontsize=12)
    plt.title(f'2D Trajectory Overlay ({len(all_trajectories)} episodes)\n'
             f'Green=Success({n_success}), Red=Crash({n_crash}), Gray=Timeout({n_timeout})',
             fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(-config.ARENA_SIZE_X/2, config.ARENA_SIZE_X/2)
    plt.ylim(-config.ARENA_SIZE_Y/2, config.ARENA_SIZE_Y/2)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved 2D trajectory overlay to {output_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize multiple flights on fixed map")
    parser.add_argument("--model", type=str, default="models/ppo_drone_nav_stage1",
                       help="Path to trained model")
    parser.add_argument("--normalize", type=str, default="models/vec_normalize_stage1.pkl",
                       help="Path to VecNormalize stats")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of episodes to collect")
    parser.add_argument("--seed", type=int, default=42,
                       help="Seed for fixed map generation")
    parser.add_argument("--output", type=str, default="visualizations/",
                       help="Output directory for plots")

    args = parser.parse_args()

    import os
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("MULTIPLE FLIGHTS VISUALIZATION")
    print("=" * 60)

    # Collect trajectories
    all_trajectories, all_infos, start_pos, goal_pos, obstacles = collect_trajectories(
        model_path=args.model,
        vec_normalize_path=args.normalize,
        n_episodes=args.episodes,
        seed=args.seed
    )

    # Generate plots
    plot_heatmap(all_trajectories, start_pos, goal_pos, obstacles,
                output_path=f"{args.output}/heatmap_seed{args.seed}.png")

    plot_2d_trajectories(all_trajectories, all_infos, start_pos, goal_pos, obstacles,
                        output_path=f"{args.output}/trajectories_2d_seed{args.seed}.png")

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\nCheck {args.output}/ for generated plots")


if __name__ == "__main__":
    main()
