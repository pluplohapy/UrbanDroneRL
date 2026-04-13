"""
Continue training from saved checkpoint.
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
    print("CONTINUE TRAINING FROM CHECKPOINT")
    print("=" * 60)

    # Check if checkpoint exists
    model_path = "models/ppo_drone_nav.zip"
    vec_normalize_path = "models/vec_normalize.pkl"

    if not os.path.exists(model_path):
        print(f"\n✗ Model not found at {model_path}")
        print("Please train first with: python train.py")
        return

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    print(f"\n[LOAD] Loading checkpoint...")

    # Create environments
    env_fns = [make_env(i, config.SEED) for i in range(config.N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)

    # Load VecNormalize stats
    vec_env = VecNormalize.load(vec_normalize_path, vec_env)
    vec_env.training = True
    vec_env.norm_reward = True
    print("✓ VecNormalize stats loaded")

    # Load model
    model = PPO.load(model_path, env=vec_env)
    print("✓ Model loaded")

    # Get current timesteps
    current_steps = model.num_timesteps
    print(f"\nCurrent training steps: {current_steps}")

    # Ask for additional steps
    additional_steps = int(input("Additional steps to train (e.g., 300000): "))
    total_steps = current_steps + additional_steps

    print(f"\nWill train from {current_steps} to {total_steps} steps")
    print(f"Additional steps: {additional_steps}")

    progress_callback = ProgressCallback()

    print(f"\n[TRAINING] Starting training...")
    print("-" * 60)

    try:
        model.learn(
            total_timesteps=additional_steps,
            callback=progress_callback,
            progress_bar=False,
            reset_num_timesteps=False  # Important: don't reset timestep counter
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
