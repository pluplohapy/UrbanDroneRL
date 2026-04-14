"""
Training script for Stage 1 (static obstacles).
Starts from Stage 0 checkpoint and continues training with obstacles.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch

from envs.nav_aviary import NavAviary
from scenarios.stage1_static import Stage1Scenario
import config


class ProgressCallback(BaseCallback):
    """Callback for displaying training progress with debug metrics."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_crashes = []
        self.episode_timeouts = []
        self.episode_lengths = []

        # Debug metrics storage
        if config.DEBUG_MODE:
            self.reward_components = {k: [] for k in ['progress', 'velocity', 'proximity', 'obstacle', 'step_penalty', 'exploration', 'terminal', 'efficiency_bonus', 'yaw_penalty']}
            self.navigation_metrics = {'path_efficiency': [], 'avg_heading_error': [], 'avg_speed': []}
            self.episode_metrics = {'start_distance': [], 'min_goal_distance': [], 'closest_obstacle': [], 'n_near_misses': []}
            self.action_stats = {'action_mean': [], 'action_std': [], 'action_smoothness': []}
            self.timeout_metrics = {'final_dist': [], 'min_dist': [], 'start_dist': []}

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

                        # Collect debug metrics
                        if config.DEBUG_MODE:
                            # Reward components
                            if config.LOG_REWARD_COMPONENTS and 'reward_components' in info:
                                for k, v in info['reward_components'].items():
                                    if k in self.reward_components:
                                        self.reward_components[k].append(v)

                            # Navigation metrics
                            if config.LOG_NAVIGATION_METRICS:
                                if 'path_efficiency' in info:
                                    self.navigation_metrics['path_efficiency'].append(info['path_efficiency'])
                                if 'avg_heading_error' in info:
                                    self.navigation_metrics['avg_heading_error'].append(info['avg_heading_error'])
                                if 'avg_speed' in info:
                                    self.navigation_metrics['avg_speed'].append(info['avg_speed'])

                            # Episode metrics
                            if config.LOG_EPISODE_METRICS:
                                if 'start_distance' in info:
                                    self.episode_metrics['start_distance'].append(info['start_distance'])
                                if 'min_goal_distance' in info:
                                    self.episode_metrics['min_goal_distance'].append(info['min_goal_distance'])
                                if 'closest_obstacle' in info:
                                    self.episode_metrics['closest_obstacle'].append(info['closest_obstacle'])
                                if 'n_near_misses' in info:
                                    self.episode_metrics['n_near_misses'].append(info['n_near_misses'])

                            # Action stats
                            if config.LOG_ACTION_STATS:
                                if 'action_mean' in info:
                                    self.action_stats['action_mean'].append(info['action_mean'])
                                if 'action_smoothness' in info:
                                    self.action_stats['action_smoothness'].append(info['action_smoothness'])

                            # Timeout analysis
                            if config.LOG_TIMEOUT_ANALYSIS and is_timeout:
                                self.timeout_metrics['final_dist'].append(info.get('dist_to_goal', 0))
                                self.timeout_metrics['min_dist'].append(info.get('min_goal_distance', 0))
                                self.timeout_metrics['start_dist'].append(info.get('start_distance', 0))

                    if self.episode_count % config.LOG_INTERVAL_EPISODES == 0:
                        self._print_progress()

        return True

    def _print_progress(self):
        """Print training progress with debug metrics."""
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

        if config.DEBUG_MODE:
            # Reward components
            if config.LOG_REWARD_COMPONENTS and len(self.reward_components['progress']) > 0:
                recent_n = min(50, len(self.reward_components['progress']))
                prog = np.mean(self.reward_components['progress'][-recent_n:])
                vel = np.mean(self.reward_components['velocity'][-recent_n:])
                prox = np.mean(self.reward_components['proximity'][-recent_n:])
                obst = np.mean(self.reward_components['obstacle'][-recent_n:])
                step = np.mean(self.reward_components['step_penalty'][-recent_n:])
                term = np.mean(self.reward_components['terminal'][-recent_n:]) if len(self.reward_components['terminal']) > 0 else 0
                print(f"  Reward   : total={avg_reward:7.1f} | prog={prog:5.1f} | vel={vel:4.1f} | prox={prox:4.1f} | obst={obst:5.1f} | term={term:5.1f}")

            # Navigation metrics
            if config.LOG_NAVIGATION_METRICS and len(self.navigation_metrics['path_efficiency']) > 0:
                recent_n = min(50, len(self.navigation_metrics['path_efficiency']))
                eff = np.mean(self.navigation_metrics['path_efficiency'][-recent_n:])
                heading = np.mean(self.navigation_metrics['avg_heading_error'][-recent_n:]) if len(self.navigation_metrics['avg_heading_error']) > 0 else 0
                speed = np.mean(self.navigation_metrics['avg_speed'][-recent_n:]) if len(self.navigation_metrics['avg_speed']) > 0 else 0
                print(f"  Navigate : efficiency={eff:.2f} | heading_err={heading:.1f}° | speed={speed:.2f}m/s")

            # Timeout analysis
            if config.LOG_TIMEOUT_ANALYSIS and len(self.timeout_metrics['final_dist']) > 0:
                final = np.mean(self.timeout_metrics['final_dist'])
                min_d = np.mean(self.timeout_metrics['min_dist'])
                print(f"  Timeouts : final_dist={final:.1f}m | min_dist={min_d:.1f}m")

            # Action stats
            if config.LOG_ACTION_STATS and len(self.action_stats['action_smoothness']) > 0:
                recent_n = min(50, len(self.action_stats['action_smoothness']))
                smooth = np.mean(self.action_stats['action_smoothness'][-recent_n:])
                if len(self.navigation_metrics['avg_speed']) > 0:
                    speed = np.mean(self.navigation_metrics['avg_speed'][-recent_n:])
                    print(f"  Actions  : speed={speed:.2f} | smoothness={smooth:.3f}")
        else:
            # Simple output when debug is off
            print(f"  Reward: {avg_reward:7.2f} | Length: {avg_length:5.1f}")

        print()  # Empty line for readability


def make_env(rank, seed=0):
    def _init():
        scenario = Stage1Scenario(seed=seed + rank)
        env = NavAviary(scenario=scenario, gui=False)
        env = Monitor(env)
        return env
    return _init


def main():
    import sys
    sys.stdout.flush()
    print("=" * 60, flush=True)
    print("STAGE 1 TRAINING - STATIC OBSTACLES", flush=True)
    print("=" * 60, flush=True)

    # Check if Stage 1 checkpoint exists (для дообучения)
    stage1_model = "models/ppo_drone_nav_stage1.zip"
    stage1_normalize = "models/vec_normalize_stage1.pkl"

    # Fallback to Stage 0 if Stage 1 doesn't exist
    if os.path.exists(stage1_model):
        stage0_model = stage1_model
        stage0_normalize = stage1_normalize
        print(f"\n[LOAD] Found Stage 1 checkpoint, will continue from it...")
    else:
        stage0_model = "models/ppo_drone_nav.zip"
        stage0_normalize = "models/vec_normalize.pkl"
        print(f"\n[LOAD] No Stage 1 checkpoint, starting from Stage 0...")

    if not os.path.exists(stage0_model):
        print(f"\n✗ Model not found at {stage0_model}")
        print("Please train Stage 0 first with: python train.py")
        return

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print(f"\n[CONFIG] Stage 1 parameters:")
    print(f"  Obstacles: {config.STAGE1_N_OBSTACLES[0]}-{config.STAGE1_N_OBSTACLES[1]}")
    print(f"  Radius: {config.STAGE1_RADIUS[0]}-{config.STAGE1_RADIUS[1]}m")
    print(f"  Height: {config.STAGE1_HEIGHT[0]}-{config.STAGE1_HEIGHT[1]}m (full cylinders)")

    print(f"\n[LOAD] Loading Stage 0 checkpoint...")

    # Create environments with Stage 1
    env_fns = [make_env(i, config.SEED) for i in range(config.N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)

    # Load VecNormalize stats from Stage 0
    vec_env = VecNormalize.load(stage0_normalize, vec_env)
    vec_env.training = True
    vec_env.norm_reward = True
    print("✓ VecNormalize stats loaded")

    # Load Stage 0 model
    model = PPO.load(stage0_model, env=vec_env)
    print("✓ Stage 0 model loaded")

    current_steps = model.num_timesteps
    print(f"\nStarting from {current_steps} steps (Stage 0)")

    progress_callback = ProgressCallback()

    print(f"\n[TRAINING] Starting Stage 1 training...")
    print("-" * 60)

    try:
        model.learn(
            total_timesteps=2000000,  # 3M additional steps (было 1.5M)
            callback=progress_callback,
            progress_bar=False,
            reset_num_timesteps=False
        )
        print("\n" + "-" * 60)
        print("✓ Training completed")

    except KeyboardInterrupt:
        print("\n\n[INFO] Training interrupted")

    print(f"\n[SAVE] Saving Stage 1 model...")
    model.save("models/ppo_drone_nav_stage1")
    vec_env.save("models/vec_normalize_stage1.pkl")
    print("✓ Model saved")

    vec_env.close()

    print("\n" + "=" * 60)
    print("STAGE 1 TRAINING FINISHED")
    print("=" * 60)
    print(f"\nTotal training steps: {model.num_timesteps}")
    print("\nДля визуализации:")
    print("  python visualize.py --model models/ppo_drone_nav_stage1 --normalize models/vec_normalize_stage1.pkl")


if __name__ == "__main__":
    main()
