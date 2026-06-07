
import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

from envs.nav_aviary import NavAviary
from envs.nav_aviary_planner import NavAviaryWithPlanner
from scenarios.stage1_static import Stage1Scenario
import config


class ComparisonCallback(BaseCallback):

    def __init__(self, name, verbose=0):
        super().__init__(verbose)
        self.name = name
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_lengths = []
        self.timesteps = []

    def _on_step(self) -> bool:
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])
                    self.timesteps.append(self.num_timesteps)

                    if "is_success" in info:
                        self.episode_successes.append(1 if info["is_success"] else 0)

        return True

    def get_stats(self):
        return {
            'name': self.name,
            'rewards': self.episode_rewards,
            'successes': self.episode_successes,
            'lengths': self.episode_lengths,
            'timesteps': self.timesteps
        }


def make_env_baseline(rank, seed=0):
    def _init():
        scenario = Stage1Scenario(seed=seed + rank)
        env = NavAviary(scenario=scenario, gui=False)
        env = Monitor(env)
        return env
    return _init


def make_env_planner(rank, seed=0):
    def _init():
        scenario = Stage1Scenario(seed=seed + rank)
        env = NavAviaryWithPlanner(
            scenario=scenario,
            gui=False,
            use_planner=True,
            replan_freq=0,
            waypoint_threshold=1.5,
            planner_params={
                'max_iter': 2000,
                'step_size': 1.0,
                'goal_bias': 0.15,
                'goal_threshold': 0.8,
                'rewire_radius': 3.0
            }
        )
        env = Monitor(env)
        return env
    return _init


def train_model(env_fns, name, total_timesteps=500000):
    print(f"\n{'='*60}")
    print(f"TRAINING: {name}")
    print(f"{'='*60}")


    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )


    model = PPO(
        **config.PPO_PARAMS,
        env=vec_env,
        verbose=0,
        device="auto"
    )


    callback = ComparisonCallback(name=name)
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )


    model_path = f"models/comparison_{name.lower().replace(' ', '_')}"
    model.save(model_path)
    vec_env.save(f"{model_path}_normalize.pkl")
    print(f"\n✓ Model saved to {model_path}")

    vec_env.close()

    return callback.get_stats()


def plot_comparison(stats_list):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['blue', 'red', 'green', 'orange']


    ax = axes[0, 0]
    for i, stats in enumerate(stats_list):

        window = 50
        if len(stats['rewards']) >= window:
            rewards_smooth = np.convolve(
                stats['rewards'],
                np.ones(window)/window,
                mode='valid'
            )
            timesteps_smooth = stats['timesteps'][:len(rewards_smooth)]
            ax.plot(timesteps_smooth, rewards_smooth,
                    label=stats['name'], color=colors[i % len(colors)])

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Episode Reward (smoothed)')
    ax.set_title('Training Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)


    ax = axes[0, 1]
    for i, stats in enumerate(stats_list):
        if len(stats['successes']) > 0:

            window = 50
            if len(stats['successes']) >= window:
                success_smooth = np.convolve(
                    stats['successes'],
                    np.ones(window)/window,
                    mode='valid'
                )
                timesteps_smooth = stats['timesteps'][:len(success_smooth)]
                ax.plot(timesteps_smooth, success_smooth,
                        label=stats['name'], color=colors[i % len(colors)])

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Success Rate (smoothed)')
    ax.set_title('Success Rate Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])


    ax = axes[1, 0]
    for i, stats in enumerate(stats_list):

        window = 50
        if len(stats['lengths']) >= window:
            lengths_smooth = np.convolve(
                stats['lengths'],
                np.ones(window)/window,
                mode='valid'
            )
            timesteps_smooth = stats['timesteps'][:len(lengths_smooth)]
            ax.plot(timesteps_smooth, lengths_smooth,
                    label=stats['name'], color=colors[i % len(colors)])

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Episode Length (smoothed)')
    ax.set_title('Episode Length Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)


    ax = axes[1, 1]
    names = [s['name'] for s in stats_list]
    final_rewards = [np.mean(s['rewards'][-100:]) if len(s['rewards']) >= 100 else 0
                     for s in stats_list]
    final_success = [np.mean(s['successes'][-100:]) if len(s['successes']) >= 100 else 0
                     for s in stats_list]

    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width/2, final_rewards, width, label='Avg Reward (last 100)', alpha=0.8)
    ax.bar(x + width/2, [s*1000 for s in final_success], width,
           label='Success Rate (last 100) ×1000', alpha=0.8)

    ax.set_ylabel('Value')
    ax.set_title('Final Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to comparison_results.png")
    plt.show()


def main():
    print("="*60)
    print("RRT* PLANNER COMPARISON EXPERIMENT")
    print("="*60)
    print("\nThis will train two models:")
    print("  1. Baseline (no planner)")
    print("  2. With RRT* planner")
    print("\nTraining time: ~30-60 minutes total")

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    os.makedirs("models", exist_ok=True)


    n_envs = config.N_ENVS
    total_timesteps = 500000


    print("\n" + "="*60)
    print("EXPERIMENT 1/2: Baseline (No Planner)")
    print("="*60)
    env_fns_baseline = [make_env_baseline(i, config.SEED) for i in range(n_envs)]
    stats_baseline = train_model(env_fns_baseline, "Baseline", total_timesteps)


    print("\n" + "="*60)
    print("EXPERIMENT 2/2: With RRT* Planner")
    print("="*60)
    env_fns_planner = [make_env_planner(i, config.SEED) for i in range(n_envs)]
    stats_planner = train_model(env_fns_planner, "RRT* Planner", total_timesteps)


    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    stats_list = [stats_baseline, stats_planner]

    for stats in stats_list:
        print(f"\n{stats['name']}:")
        print(f"  Total episodes: {len(stats['rewards'])}")

        if len(stats['rewards']) >= 100:
            final_reward = np.mean(stats['rewards'][-100:])
            print(f"  Final avg reward (last 100): {final_reward:.2f}")

        if len(stats['successes']) >= 100:
            final_success = np.mean(stats['successes'][-100:])
            print(f"  Final success rate (last 100): {final_success*100:.1f}%")

        if len(stats['lengths']) >= 100:
            final_length = np.mean(stats['lengths'][-100:])
            print(f"  Final avg length (last 100): {final_length:.1f}")


    print("\n" + "="*60)
    print("GENERATING COMPARISON PLOTS")
    print("="*60)
    plot_comparison(stats_list)

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED")
    print("="*60)
    print("\nResults saved:")
    print("  - Models: models/comparison_*.zip")
    print("  - Plot: comparison_results.png")


if __name__ == "__main__":
    main()