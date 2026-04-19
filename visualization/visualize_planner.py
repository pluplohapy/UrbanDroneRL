"""
Visualization script for trained drone with RRT* planner.
Shows the planned path and drone following it.
"""

import numpy as np
import argparse
import time
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from envs.nav_aviary_planner import NavAviaryWithPlanner
from scenarios.stage0_empty import Stage0Scenario
from scenarios.stage1_static import Stage1Scenario
from planners.visualization import visualize_rrt_tree_3d, visualize_rrt_tree_2d
import config


def visualize_episode(env, model, render=True, save_video=False):
    """
    Run one episode and visualize.

    Args:
        env: Environment
        model: Trained model
        render: Whether to render in PyBullet GUI
        save_video: Whether to save video

    Returns:
        Episode info dictionary
    """
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0

    # Get planner info
    planner = env.envs[0].planner
    waypoints = env.envs[0].waypoints
    start = env.envs[0].start_pos
    goal = env.envs[0].goal_pos
    obstacles = env.envs[0].scenario.obstacles
    client = env.envs[0].CLIENT

    print(f"\n[Episode] Start: {start}")
    print(f"[Episode] Goal: {goal}")
    print(f"[Episode] Waypoints: {len(waypoints)}")
    print(f"[Episode] Obstacles: {len(obstacles)}")

    # Draw goal sphere (blue, semi-transparent)
    goal_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=config.SUCCESS_DIST,
        rgbaColor=[0, 0, 1, 0.3],
        physicsClientId=client
    )
    goal_body = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=goal_visual,
        basePosition=goal,
        physicsClientId=client
    )

    # Draw planned path (thick blue line) and waypoint spheres (black)
    waypoint_threshold = env.envs[0].waypoint_threshold  # 1.5m
    if planner is not None and len(waypoints) > 1:
        # Draw waypoint spheres (black, semi-transparent)
        for i, wp in enumerate(waypoints):
            wp_visual = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=waypoint_threshold,  # 1.5m - реальная зона достижения
                rgbaColor=[0, 0, 0, 0.2],  # полупрозрачные
                physicsClientId=client
            )
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=wp_visual,
                basePosition=wp,
                physicsClientId=client
            )

        # Draw path lines
        for i in range(len(waypoints) - 1):
            p.addUserDebugLine(
                waypoints[i],
                waypoints[i + 1],
                lineColorRGB=[0, 0, 1],
                lineWidth=5,
                physicsClientId=client
            )

    # Visualize planned path
    if planner is not None and len(waypoints) > 0:
        print(f"\n[Visualizing] RRT* tree and path...")
        visualize_rrt_tree_2d(planner, waypoints, start, goal, obstacles, view='xy')
        visualize_rrt_tree_3d(planner, waypoints, start, goal, obstacles)

    trajectory = [start.copy()]
    prev_pos = start.copy()

    print(f"\n[Episode] Running...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        total_reward += reward[0]
        steps += 1

        # Track trajectory
        drone_state = env.envs[0]._getDroneStateVector(0)
        drone_pos = drone_state[:3]
        drone_vel = drone_state[10:13]
        trajectory.append(drone_pos.copy())

        # Calculate metrics
        speed = np.linalg.norm(drone_vel)
        dist_to_goal = np.linalg.norm(goal - drone_pos)
        yaw_rate = abs(action[0][3]) * config.YAW_RATE_MAX if len(action[0]) > 3 else 0

        # Draw actual trajectory (thick red line)
        p.addUserDebugLine(
            prev_pos,
            drone_pos,
            lineColorRGB=[1, 0, 0],
            lineWidth=5,
            physicsClientId=client
        )
        prev_pos = drone_pos.copy()

        if render:
            time.sleep(1.0 / 30.0)  # 30 FPS

        # Print detailed progress every 50 steps
        if 'current_waypoint_idx' in info[0]:
            current_wp_idx = info[0]['current_waypoint_idx']
            if steps % 50 == 0:
                print(f"  Step {steps}: wp {current_wp_idx+1}/{len(waypoints)} | "
                      f"pos=[{drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f}] | "
                      f"dist={dist_to_goal:.2f}m | speed={speed:.2f}m/s | "
                      f"yaw_rate={yaw_rate:.2f}rad/s | reward={reward[0]:.2f}")

    # Results
    is_success = info[0].get('is_success', False)
    is_crash = info[0].get('is_crash', False)
    dist_to_goal = info[0].get('dist_to_goal', 0)

    print(f"\n[Episode] Finished!")
    print(f"  Result: {'SUCCESS' if is_success else 'CRASH' if is_crash else 'TIMEOUT'}")
    print(f"  Steps: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final distance to goal: {dist_to_goal:.2f}m")

    # Visualize trajectory vs planned path
    if planner is not None and len(waypoints) > 0:
        print(f"\n[Visualizing] Actual trajectory vs planned path...")
        from planners.visualization import plot_path_comparison

        plot_path_comparison(
            paths=[
                ('Planned Path', waypoints),
                ('Actual Trajectory', trajectory)
            ],
            start=start,
            goal=goal,
            obstacles=obstacles
        )

    return {
        'success': is_success,
        'crash': is_crash,
        'steps': steps,
        'reward': total_reward,
        'dist_to_goal': dist_to_goal,
        'trajectory': trajectory
    }


def main():
    parser = argparse.ArgumentParser(description='Visualize trained drone with RRT* planner')
    parser.add_argument('--model', type=str, default='models/ppo_drone_nav_planner',
                        help='Path to trained model')
    parser.add_argument('--normalize', type=str, default=None,
                        help='Path to VecNormalize stats (auto-detected if not specified)')
    parser.add_argument('--stage', type=int, default=1, choices=[0, 1],
                        help='Stage to visualize (0=empty, 1=obstacles)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--n_episodes', type=int, default=5,
                        help='Number of episodes to run')
    parser.add_argument('--no-gui', action='store_true',
                        help='Disable PyBullet GUI')
    parser.add_argument('--no-planner', action='store_true',
                        help='Disable RRT* planner (direct to goal)')

    args = parser.parse_args()

    print("=" * 60)
    print("DRONE NAVIGATION VISUALIZATION WITH RRT*")
    print("=" * 60)

    # Auto-detect normalize path if not specified
    if args.normalize is None:
        # Extract stage from model path or use args.stage
        if 'stage0' in args.model:
            detected_stage = 0
        elif 'stage1' in args.model:
            detected_stage = 1
        else:
            detected_stage = args.stage

        # Determine if planner version
        if 'planner' in args.model or not args.no_planner:
            args.normalize = f'models/vec_normalize_stage{detected_stage}_planner.pkl'
        else:
            args.normalize = f'models/vec_normalize_stage{detected_stage}.pkl'

        print(f"[INFO] Auto-detected normalize path: {args.normalize}")

    # Create scenario
    if args.stage == 0:
        scenario = Stage0Scenario(seed=args.seed)
        print(f"\n[Stage 0] Empty arena")
    else:
        scenario = Stage1Scenario(seed=args.seed)
        print(f"\n[Stage 1] Static obstacles")

    # Create environment
    def make_env():
        env = NavAviaryWithPlanner(
            scenario=scenario,
            gui=not args.no_gui,
            use_planner=not args.no_planner,
            replan_freq=0,
            waypoint_threshold=config.WAYPOINT_THRESHOLD,  # 0.3м
            planner_params={
                'max_iter': 2000,
                'step_size': 1.0,
                'goal_bias': 0.15,
                'goal_threshold': 0.8,
                'rewire_radius': 3.0
            }
        )
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(args.normalize, env)
    env.training = False
    env.norm_reward = False

    print(f"✓ Environment created")

    # Load model
    model = PPO.load(args.model)
    print(f"✓ Model loaded from {args.model}")

    # Run episodes
    results = []
    for episode in range(args.n_episodes):
        print(f"\n{'='*60}")
        print(f"EPISODE {episode + 1}/{args.n_episodes}")
        print(f"{'='*60}")

        result = visualize_episode(env, model, render=not args.no_gui)
        results.append(result)

        # Reset scenario for next episode
        if episode < args.n_episodes - 1:
            scenario.seed = args.seed + episode + 1

    # Summary statistics
    print(f"\n{'='*60}")
    print(f"SUMMARY ({args.n_episodes} episodes)")
    print(f"{'='*60}")

    success_rate = sum(1 for r in results if r['success']) / len(results)
    crash_rate = sum(1 for r in results if r['crash']) / len(results)
    timeout_rate = 1.0 - success_rate - crash_rate

    avg_steps = np.mean([r['steps'] for r in results])
    avg_reward = np.mean([r['reward'] for r in results])
    avg_dist = np.mean([r['dist_to_goal'] for r in results])

    print(f"\nSuccess rate: {success_rate*100:.1f}%")
    print(f"Crash rate: {crash_rate*100:.1f}%")
    print(f"Timeout rate: {timeout_rate*100:.1f}%")
    print(f"\nAverage steps: {avg_steps:.1f}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average final distance: {avg_dist:.2f}m")

    # Success-only statistics
    successful = [r for r in results if r['success']]
    if successful:
        print(f"\nSuccessful episodes only:")
        print(f"  Average steps: {np.mean([r['steps'] for r in successful]):.1f}")
        print(f"  Average reward: {np.mean([r['reward'] for r in successful]):.2f}")

    env.close()

    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETED")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
