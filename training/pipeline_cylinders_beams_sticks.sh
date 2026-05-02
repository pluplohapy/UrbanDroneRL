#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
mkdir -p reports "$MPLCONFIGDIR"
ARTIFACT_TAG="${ARTIFACT_TAG:-rl3_$(date +%m%d_%H%M)}"
TRAIN_SEED="${TRAIN_SEED:-$((100000 + RANDOM * 1000 + RANDOM))}"
EVAL_SEED="${EVAL_SEED:-$((200000 + RANDOM * 1000 + RANDOM))}"

echo "[PIPELINE] Artifact tag: $ARTIFACT_TAG"
echo "[PIPELINE] Train seed: $TRAIN_SEED"
echo "[PIPELINE] Eval seed: $EVAL_SEED"

python training/curriculum_train.py \
  --algo recurrent_ppo \
  --stages cylinders,beams,swinging_sticks \
  --skip-stages "" \
  --start-from "" \
  --allow-scratch \
  --chunk-timesteps 8000000 \
  --max-rounds-per-stage 5 \
  --continue-on-fail \
  --eval-episodes 100 \
  --train-eval-episodes 50 \
  --eval-freq 500000 \
  --target-overrides cylinders=0.85,beams=0.65,swinging_sticks=0.70 \
  --n-envs 8 \
  --learning-rate 3e-5 \
  --train-seed "$TRAIN_SEED" \
  --eval-seed "$EVAL_SEED" \
  --seed-stride 1000 \
  --artifact-tag "$ARTIFACT_TAG" \
  --run-tag "$ARTIFACT_TAG" \
  --make-plots \
  --plots-dir reports/training_plots \
  --plots-formats png,svg \
  2>&1 | tee "reports/pipeline_cylinders_beams_swinging_sticks_${ARTIFACT_TAG}.log"
