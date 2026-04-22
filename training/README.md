# Training Scripts

Unified training pipeline for drone navigation with progressive stages.

**🆕 Все stages теперь используют RRT* планировщик по умолчанию!**

## Quick Start

### Stage 0: Empty Arena with RRT*
```bash
# Train with RRT* planner (500k steps, ~30-60 min)
python training/train.py --stage 0

# Train without planner (baseline)
python training/train.py --stage 0 --no-planner

# Visualize
python visualize_planner.py --model models/ppo_drone_nav_stage0_planner
```

### Stage 1: Static Obstacles with RRT* Planner
```bash
# Train with RRT* planner (1.5M steps, ~2-3 hours)
python training/train.py --stage 1

# Train without planner (baseline comparison)
python training/train.py --stage 1 --no-planner

# Visualize
python visualize_planner.py --model models/ppo_drone_nav_stage1_planner
```

## Command Line Options

```bash
python training/train.py --stage {0,1} [OPTIONS]

Required:
  --stage {0,1}           Training stage (0=empty, 1=obstacles)

Optional:
  --no-planner            Disable RRT* planner (enabled by default for all stages)
  --timesteps N           Total training steps (default: 500k for Stage 0, 1.5M for Stage 1)
  --n-envs N              Number of parallel environments (default: 4)
  --continue              Continue training from checkpoint
  --watch                 Enable continuous GUI watch mode (forces n-envs=1, fixed map, draws trajectory)
  --watch-fps N           Watch mode FPS limit (default: 30)
  --fixed-map             Keep same start/goal/obstacles across episodes
  --show-paths            Draw drone trajectory in GUI
  --no-show-paths         Disable trajectory drawing (faster GUI)
  --swarm-drones N        Parallel drones in one shared map (e.g. 8)
```

## Examples

### Basic Training Pipeline (Recommended)
```bash
# 1. Train Stage 0 with RRT* (learn waypoint following)
python training/train.py --stage 0

# 2. Train Stage 1 with transfer learning (obstacles + RRT*)
python training/train.py --stage 1
```

### Training Without Planner (Baseline Comparison)
```bash
# Stage 0 without planner (direct flight)
python training/train.py --stage 0 --no-planner

# Stage 1 without planner (direct flight with obstacles)
python training/train.py --stage 1 --no-planner
```

### Custom Training
```bash
# Train Stage 0 with more steps
python training/train.py --stage 0 --timesteps 1000000

# Train Stage 1 with 8 parallel environments
python training/train.py --stage 1 --n-envs 8

# Continue interrupted training
python training/train.py --stage 1 --continue

# Watch training live in PyBullet GUI
python training/train.py --stage 0 --watch

# Watch mode with slower playback
python training/train.py --stage 1 --watch --watch-fps 20

# Fixed-map training without GUI (same map every episode)
python training/train.py --stage 1 --fixed-map

# 8 drones simultaneously in one map (best for your requested mode)
python training/train.py --stage pretrain --obstacle-type cylinders --watch --swarm-drones 8

# Faster GUI for swarm: no trajectory lines
python training/train.py --stage pretrain --obstacle-type cylinders --watch --swarm-drones 8 --watch-fps 90 --no-show-paths
```

### Comparison Experiments
```bash
# Train Stage 1 WITHOUT planner (for comparison)
python training/train.py --stage 1 --no-planner

# Compare results
python compare_planner.py
```

## Training Progress

### Monitor in Console
Progress is printed every 10 episodes:
```
Episode   10 | Steps:   20480
  Outcomes : S= 20% | C= 30% | T= 50%
  Planner  : failures=  5% | avg_waypoints=4.2
  Reward: 156.30 | Length: 245.0
```

**Metrics:**
- `S` - Success rate (reached goal)
- `C` - Crash rate (collision)
- `T` - Timeout rate (time limit)
- `failures` - RRT* planning failures
- `avg_waypoints` - Average waypoints per path

### Monitor with TensorBoard
```bash
# In another terminal
tensorboard --logdir logs/

# Open browser
http://localhost:6006
```

## Model Files

Models are saved to `models/`:

```
models/
├── ppo_drone_nav_stage0_planner.zip      # Stage 0 with RRT*
├── vec_normalize_stage0_planner.pkl      # Stage 0 normalization
├── ppo_drone_nav_stage0.zip              # Stage 0 without planner
├── vec_normalize_stage0.pkl              # Stage 0 normalization
├── ppo_drone_nav_stage1_planner.zip      # Stage 1 with RRT*
├── vec_normalize_stage1_planner.pkl      # Stage 1 normalization
├── ppo_drone_nav_stage1.zip              # Stage 1 without planner
└── vec_normalize_stage1.pkl              # Stage 1 normalization
```

Checkpoints are saved every 50,000 steps:
```
models/
├── ppo_drone_nav_stage0_planner_50000.zip
├── ppo_drone_nav_stage1_planner_50000.zip
├── ppo_drone_nav_stage1_planner_100000.zip
└── ...
```

## Transfer Learning

Stage 1 automatically uses Stage 0 planner model if available:

```bash
# 1. Train Stage 0 with RRT*
python training/train.py --stage 0

# 2. Stage 1 will automatically load Stage 0 for transfer learning
python training/train.py --stage 1
```

This significantly improves Stage 1 training:
- Faster convergence (50% fewer steps)
- Higher success rate (10-20% improvement)
- More stable training

## Expected Results

### Stage 0 with RRT* (Empty Arena)
- **Training time**: 30-60 minutes
- **Success rate**: 95-100%
- **Episode length**: 150-250 steps
- **Reward**: 200-300
- **Planning time**: 10-20 seconds per episode (fast, no obstacles)
- **Waypoints**: 2-4 per path

### Stage 0 without RRT* (Baseline)
- **Training time**: 30-60 minutes
- **Success rate**: 90-100%
- **Episode length**: 150-250 steps
- **Reward**: 200-300

### Stage 1 with RRT* (Static Obstacles)
- **Training time**: 2-3 hours
- **Success rate**: 60-80%
- **Episode length**: 250-400 steps
- **Reward**: 150-250
- **Planning time**: 25-40 seconds per episode
- **Waypoints**: 3-6 per path

### Stage 1 without RRT* (Baseline)
- **Training time**: 2-3 hours
- **Success rate**: 20-30% ❌
- **Episode length**: 400-600 steps
- **Reward**: 50-150

## Troubleshooting

### Training is slow
```bash
# Reduce parallel environments
python training/train.py --stage 1 --n-envs 2

# Reduce RRT* iterations (edit training/train.py line 254)
'max_iter': 500  # instead of 1000
```

### Out of memory
```bash
# Reduce parallel environments
python training/train.py --stage 1 --n-envs 2
```

### Training interrupted
```bash
# Continue from checkpoint
python training/train.py --stage 1 --continue
```

### Low success rate
- Check TensorBoard for learning curves
- Ensure Stage 0 is trained first
- Increase training steps: `--timesteps 2000000`

## Configuration

Edit `config.py` to adjust:
- Reward weights
- Environment parameters
- PPO hyperparameters
- Debug logging

## Visualization

After training, visualize the results:

```bash
# Stage 0
python visualize.py --model models/ppo_drone_nav_stage0 --episodes 5

# Stage 1 with planner
python visualize_planner.py --model models/ppo_drone_nav_stage1_planner --episodes 5

# Stage 1 without planner
python visualize.py --model models/ppo_drone_nav_stage1 --episodes 5
```

## Next Steps

After completing Stage 0 and Stage 1:

1. **Compare approaches**:
   ```bash
   python compare_planner.py
   ```

2. **Analyze results**:
   - Check `logs/` for TensorBoard data
   - Review `logs/debug_metrics.csv` for detailed metrics

3. **Prepare for Stage 2**:
   - Dynamic obstacles
   - Larger arena
   - Hierarchical planning

## Tips

1. **Always train Stage 0 first** - it provides baseline skills
2. **Use transfer learning** - Stage 1 trains faster with Stage 0 model
3. **Monitor TensorBoard** - catch issues early
4. **Save checkpoints** - training can be interrupted
5. **Compare with/without planner** - understand RRT* impact

## Support

For issues or questions:
- Check `README.md` in project root
- Review `RRT_STAR_README.md` for planner details
- See `USAGE.md` for general usage guide
