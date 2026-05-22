# Training Plot Report

Generated at: 2026-05-12T11:51:35

## Input rows
- eval history rows: 25
- milestone rows: 49
- compact episode rows: 2439
- summary rows: 3
- tensorboard scalar rows: 2430
- curriculum report rows: 0

## Generated figures
- 01_eval_history.png
- 02_training_milestones.png
- 03_policy_losses.png
- 04_optimization_diagnostics.png
- 10_rl_reward_success_curves.png
- 11_rl_loss_kl_entropy_curves.png
- 12_rl_policy_update_runtime_curves.png
- 05_reward_components.png
- 06_failure_breakdown.png
- 07_episode_diagnostics.png
- 08_behavior_by_outcome.png

## Best eval row
- run_id: Цилиндры
- timesteps: 1800000.0
- success_rate: 0.92
- mean_reward: 1144.1341331418118

## Notes for thesis use
- Plot legends are kept outside the plotting area. If a figure has too many series, the plot is left clean and labels are written to legend_labels.csv.
- Plot set: full. Full mode writes the complete diagnostics set; compact mode is available only for temporary quick previews.
- Retry X-axis stitching: enabled.
- 01_eval_history: success/crash/timeout and reward during periodic evaluation.
- 02_training_milestones: rolling success/crash/timeout, reward, progress, and final distance during training.
- 10_rl_reward_success_curves: rollout/eval reward and success/failure dynamics.
- 11_rl_loss_kl_entropy_curves: total loss, policy loss, value loss, entropy loss, KL, and explained variance.
- 12_rl_policy_update_runtime_curves: policy std, clipping, learning rate, episode length, and FPS.
- 03_policy_losses and 04_optimization_diagnostics: extra PPO optimization signals from TensorBoard.
- 05_reward_components: reward shaping contribution sanity check.
- 06_failure_breakdown: final failure reason breakdown.
- 07_episode_diagnostics and 08_behavior_by_outcome: navigation behavior and failure analysis.
