"""
Unified training script for drone navigation.
Supports Stage 0 (empty) and Stage 1 (static obstacles with RRT* planner).

Usage:
    python training/train.py --stage 0                    # Train Stage 0
    python training/train.py --stage 1                    # Train Stage 1 with RRT*
    python training/train.py --stage 1 --no-planner       # Train Stage 1 without planner
    python training/train.py --stage 0 --timesteps 1000000  # Custom timesteps
"""

import os
import sys
import argparse
import numpy as np
import torch
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
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


class ProgressCallback(BaseCallback):
    """Callback for displaying training progress."""

    def __init__(self, stage, use_planner, config, verbose=0):
        super().__init__(verbose)
        self.stage = stage
        self.use_planner = use_planner
        self.config = config
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_crashes = []
        self.episode_timeouts = []
        self.episode_lengths = []

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

                    # Track success/crash/timeout
                    if "is_success" in info:
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


def make_env_stage0(rank, seed=0, use_planner=True, config=None):
    """Create Stage 0 environment (empty arena)."""
    def _init():
        scenario = Stage0Scenario(seed=seed + rank)

        if use_planner:
            env = NavAviaryWithPlanner(
                scenario=scenario,
                gui=False,
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
                gui=False
            )

        env = Monitor(env)
        return env
    return _init


def make_env_stage1(rank, seed=0, use_planner=True, config=None):
    """Create Stage 1 environment (static obstacles)."""
    def _init():
        scenario = Stage1Scenario(seed=seed + rank)

        if use_planner:
            env = NavAviaryWithPlanner(
                scenario=scenario,
                gui=False,
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
                gui=False
            )

        env = Monitor(env)
        return env
    return _init


def make_env_pretrain(rank, seed=0, obstacle_type='random', config=None):
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
            gui=False
        )

        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description='Train drone navigation')
    parser.add_argument('--stage', type=str, required=True, choices=['0', '1', 'pretrain'],
                        help='Training stage: 0 (empty), 1 (obstacles), pretrain (diverse)')
    parser.add_argument('--obstacle-type', type=str, default='random',
                        choices=['random', 'cylinders', 'spheres', 'walls', 'beams', 'boxes', 'swinging_sticks'],
                        help='Obstacle type for pretrain stage (default: random)')
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
        env_fns = [make_env_stage0(i, config.SEED, use_planner, config) for i in range(n_envs)]
    elif stage == '1':
        env_fns = [make_env_stage1(i, config.SEED, use_planner, config) for i in range(n_envs)]
    else:  # pretrain
        env_fns = [make_env_pretrain(i, config.SEED, args.obstacle_type, config) for i in range(n_envs)]

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
    progress_callback = ProgressCallback(stage=stage, use_planner=use_planner, config=config)

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

    vec_env.close()

    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)
    print(f"\nTotal training steps: {model.num_timesteps:,}")

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
