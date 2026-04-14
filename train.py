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
    """Callback for displaying training progress."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_count += 1
                    ep_reward = info["episode"]["r"]
                    ep_length = info["episode"]["l"]
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)

                    if "is_success" in info:
                        self.episode_successes.append(1 if info["is_success"] else 0)

                    if self.episode_count % 10 == 0:
                        recent_rewards = self.episode_rewards[-10:]
                        recent_successes = self.episode_successes[-min(50, len(self.episode_successes)):]
                        recent_lengths = self.episode_lengths[-10:]

                        avg_reward = np.mean(recent_rewards)
                        success_rate = np.mean(recent_successes) if recent_successes else 0.0
                        avg_length = np.mean(recent_lengths)

                        print(f"Episode {self.episode_count:4d} | "
                              f"Steps: {self.num_timesteps:6d} | "
                              f"Reward: {avg_reward:7.2f} | "
                              f"Success: {success_rate:5.1%} | "
                              f"Length: {avg_length:5.1f}")

        return True


def make_env(rank, seed=0):
    def _init():
        scenario = Stage0Scenario(seed=seed + rank)
        env = NavAviary(scenario=scenario, gui=False)
        env = Monitor(env)
        return env
    return _init


def main():
    print("=" * 60)
    print("DRONE NAVIGATION TRAINING - 300K STEPS")
    print("=" * 60)

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
            total_timesteps=1000000,
            callback=progress_callback,
            progress_bar=False
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
    print("\nДля визуализации:")
    print("  python visualize.py --model models/ppo_drone_nav --normalize models/vec_normalize.pkl")


if __name__ == "__main__":
    main()
