"""
Training script for drone navigation.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch

from envs.nav_aviary import NavAviary
from scenarios.stage0_empty import Stage0Scenario
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
        scenario = Stage0Scenario(seed=seed + rank)
        env = NavAviary(scenario=scenario, gui=False)
        env = Monitor(env)
        return env
    return _init


def main():
    print("=" * 60)
    print("DRONE NAVIGATION TRAINING - STAGE 0")
    print("=" * 60)

    # Check if Stage 0 checkpoint exists (для дообучения)
    stage0_model = "models/ppo_drone_nav.zip"
    stage0_normalize = "models/vec_normalize.pkl"

    continue_training = os.path.exists(stage0_model) and os.path.exists(stage0_normalize)

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print(f"\n[CONFIG] Улучшенные параметры:")
    print(f"  REWARD_STEP_PENALTY: {config.REWARD_STEP_PENALTY} (было 0.008)")
    print(f"  REWARD_PROGRESS_SCALE: {config.REWARD_PROGRESS_SCALE} (было 3.0)")
    print(f"  MAX_STEPS: {config.MAX_STEPS} (было 500)")
    print(f"  + velocity_reward: награда за полёт к цели")

    print(f"\n[SETUP] Creating environments...")
    env_fns = [make_env(i, config.SEED) for i in range(config.N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)

    if continue_training:
        print(f"\n[LOAD] Found Stage 0 checkpoint, continuing training...")
        # Load VecNormalize stats
        vec_env = VecNormalize.load(stage0_normalize, vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
        print("✓ VecNormalize stats loaded")

        # Load model
        model = PPO.load(stage0_model, env=vec_env)
        print("✓ Stage 0 model loaded")
        print(f"\nStarting from {model.num_timesteps} steps")
    else:
        print(f"\n[SETUP] No checkpoint found, starting from scratch...")
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
            verbose=0,
            device="auto"
        )
        print("✓ Model created")

    progress_callback = ProgressCallback()

    print(f"\n[TRAINING] Starting training...")
    print("-" * 60)

    try:
        model.learn(
            total_timesteps=1500000,
            callback=progress_callback,
            progress_bar=False,
            reset_num_timesteps=False  # Продолжить счетчик шагов
        )
        print("\n" + "-" * 60)
        print("✓ Training completed")

    except KeyboardInterrupt:
        print("\n\n[INFO] Training interrupted")

    print(f"\n[SAVE] Saving model...")
    model.save("models/ppo_drone_nav")
    vec_env.save("models/vec_normalize.pkl")
    print("✓ Model saved")

    vec_env.close()

    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)
    print(f"\nTotal training steps: {model.num_timesteps}")
    print("\nДля визуализации:")
    print("  python visualize.py --model models/ppo_drone_nav --normalize models/vec_normalize.pkl")


if __name__ == "__main__":
    main()
