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
