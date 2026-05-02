"""
Night curriculum runner for pretrain obstacle types.

The script trains one obstacle family at a time, evaluates the current best
policy, and moves to the next family once the requested success rate is reached.
It is intentionally a thin orchestrator around training/train.py and
training/evaluate.py so normal checkpoints, VecNormalize files, diagnostics and
eval logs keep their existing format.
"""

from __future__ import annotations

import atexit
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ALGO_CHOICES = ("ppo", "recurrent_ppo")

DEFAULT_STAGES = (
    "cylinders",
    "walls",
    "boxes",
    "gates",
    "slalom",
    "spheres",
    "crossing_spheres",
    "swinging_sticks",
    "beams",
    "dynamic_mix",
    "city_blocks",
)

DEFAULT_TARGETS = {
    "cylinders": 0.80,
    "walls": 0.80,
    "boxes": 0.80,
    "gates": 0.75,
    "slalom": 0.75,
    "spheres": 0.75,
    "crossing_spheres": 0.70,
    "swinging_sticks": 0.70,
    "beams": 0.65,
    "dynamic_mix": 0.65,
    "city_blocks": 0.70,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def split_csv(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_run_tag(value: str) -> str:
    raw = str(value or "").strip()
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw).strip("_")


def algo_display_name(algo: str) -> str:
    return "RecurrentPPO" if algo == "recurrent_ppo" else "PPO"


def model_prefix(algo: str) -> str:
    return "recurrent_ppo" if algo == "recurrent_ppo" else "ppo"


def pretrain_obs_suffix() -> str:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.append(root)

    from config import load_config

    cfg = load_config("pretrain")
    return "_enhanced_obs" if getattr(cfg, "USE_ENHANCED_OBS", False) else ""


def artifact_suffix(artifact_tag: str) -> str:
    tag = safe_run_tag(artifact_tag)
    return f"_{tag}" if tag else ""


def normalize_filename(algo: str, obstacle_type: str, artifact_tag: str = "") -> str:
    if algo == "recurrent_ppo":
        base = f"vec_normalize_recurrent_ppo_pretrain_{obstacle_type}.pkl"
    else:
        base = f"vec_normalize_pretrain_{obstacle_type}.pkl"

    suffix = pretrain_obs_suffix()
    if suffix:
        base = base[:-4] + f"{suffix}.pkl"
    suffix = artifact_suffix(artifact_tag)
    if suffix:
        base = base[:-4] + f"{suffix}.pkl"
    return base


def checkpoint_candidates(root: Path, algo: str, obstacle_type: str, artifact_tag: str = "") -> list[tuple[Path, Path, str]]:
    display = algo_display_name(algo)
    prefix = model_prefix(algo)
    suffix = pretrain_obs_suffix()
    tag_suffix = artifact_suffix(artifact_tag)
    log_name = f"{display}_pretrain_{obstacle_type}{suffix}{tag_suffix}"

    return [
        (
            root / "models" / "best_checkpoints" / log_name / "best_model.zip",
            root / "models" / "best_checkpoints" / log_name / "best_model_vecnormalize.pkl",
            "best",
        ),
        (
            root / "models" / f"{prefix}_pretrain_{obstacle_type}{suffix}{tag_suffix}.zip",
            root / "models" / normalize_filename(algo, obstacle_type, artifact_tag),
            "main",
        ),
        (
            root / "models" / "last_checkpoints" / log_name / "last_model.zip",
            root / "models" / "last_checkpoints" / log_name / "last_model_vecnormalize.pkl",
            "last",
        ),
    ]


def find_checkpoint(root: Path, algo: str, obstacle_type: str, artifact_tag: str = "") -> tuple[Path, Path, str] | None:
    for model_path, normalize_path, label in checkpoint_candidates(root, algo, obstacle_type, artifact_tag):
        if model_path.exists() and normalize_path.exists():
            return model_path, normalize_path, label
    return None


def parse_target_overrides(value: str) -> dict[str, float]:
    targets = dict(DEFAULT_TARGETS)
    for item in split_csv(value):
        if "=" not in item:
            raise ValueError(f"Target override must look like obstacle=0.75, got: {item}")
        name, raw_rate = item.split("=", 1)
        rate = float(raw_rate)
        if rate < 0.0 or rate > 1.0:
            raise ValueError(f"Target success rate must be in [0, 1], got: {item}")
        targets[name.strip()] = rate
    return targets


def run_command(root: Path, command: list[str], dry_run: bool) -> None:
    print("\n[CMD] " + " ".join(command), flush=True)
    if dry_run:
        return
    completed = subprocess.run(command, cwd=root)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}")


def evaluate_policy(
    root: Path,
    algo: str,
    model_path: Path,
    normalize_path: Path,
    obstacle_type: str,
    episodes: int,
    seed: int,
    reports_dir: Path,
    tag: str,
    dry_run: bool,
) -> float:
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"eval_{obstacle_type}_{tag}.json"
    command = [
        sys.executable,
        "training/evaluate.py",
        "--algo",
        algo,
        "--model",
        str(model_path),
        "--normalize",
        str(normalize_path),
        "--stage",
        "pretrain",
        "--obstacle-type",
        obstacle_type,
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--no-safety-shield",
        "--json-out",
        str(report_path),
    ]
    run_command(root, command, dry_run=dry_run)
    if dry_run:
        return 0.0

    with report_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    success_rate = float(payload["summary"]["success_rate"])
    print(f"[EVAL] {obstacle_type}: success={success_rate:.1%} report={report_path}", flush=True)
    return success_rate


def staged_seed(base_seed: int, stage_idx: int, round_idx: int, stride: int) -> int:
    return int(base_seed + stage_idx * stride + round_idx)


def train_chunk(
    root: Path,
    args: argparse.Namespace,
    obstacle_type: str,
    stage_idx: int,
    round_idx: int,
    init_model: Path | None,
    init_normalize: Path | None,
    reset_best: bool,
) -> None:
    train_seed = staged_seed(args.train_seed, stage_idx, round_idx, args.seed_stride)
    train_eval_seed = staged_seed(args.eval_seed, stage_idx, round_idx, args.seed_stride)
    command = [
        sys.executable,
        "training/train.py",
        "--algo",
        args.algo,
        "--stage",
        "pretrain",
        "--obstacle-type",
        obstacle_type,
        "--timesteps",
        str(args.chunk_timesteps),
        "--seed",
        str(train_seed),
        "--eval",
        "--eval-freq",
        str(args.eval_freq),
        "--eval-episodes",
        str(args.train_eval_episodes),
        "--eval-seed",
        str(train_eval_seed),
        "--promote-best-to-main",
        "--no-save-final-to-main",
        "--no-safety-shield",
    ]

    if init_model is not None and init_normalize is not None:
        command.extend([
            "--continue",
            "--init-model",
            str(init_model),
            "--init-normalize",
            str(init_normalize),
        ])
    elif not args.allow_scratch:
        raise RuntimeError(
            "No start checkpoint is available. Pass --start-model/--start-normalize "
            "or use --allow-scratch."
        )

    if reset_best:
        command.append("--reset-best")
    if args.n_envs is not None:
        command.extend(["--n-envs", str(args.n_envs)])
    if args.learning_rate is not None:
        command.extend(["--learning-rate", str(args.learning_rate)])
    if args.no_diag:
        command.append("--no-diag")
    if args.run_tag:
        command.extend(["--run-tag", args.run_tag])
    if args.artifact_tag:
        command.extend(["--artifact-tag", args.artifact_tag])

    run_command(root, command, dry_run=args.dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged overnight pretrain curriculum")
    parser.add_argument("--algo", choices=ALGO_CHOICES, default="ppo")
    parser.add_argument("--stages", default=",".join(DEFAULT_STAGES),
                        help="Comma-separated obstacle types to train in order")
    parser.add_argument("--skip-stages", default="empty",
                        help="Comma-separated obstacle types to remove from --stages")
    parser.add_argument("--start-from", default="cylinders,empty",
                        help="Auto-detect start checkpoint from these obstacle types")
    parser.add_argument("--start-model", default=None,
                        help="Explicit checkpoint .zip to start from")
    parser.add_argument("--start-normalize", default=None,
                        help="Explicit VecNormalize .pkl to start from")
    parser.add_argument("--chunk-timesteps", type=int, default=1_000_000,
                        help="Training budget per attempt on each obstacle type")
    parser.add_argument("--max-rounds-per-stage", type=int, default=3,
                        help="Maximum chunks before giving up on a stage")
    parser.add_argument("--target-success", type=float, default=0.70,
                        help="Fallback target if an obstacle has no specific target")
    parser.add_argument("--target-overrides", default="",
                        help="Comma-separated overrides, e.g. beams=0.70,city_blocks=0.75")
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Episodes for pass/fail checks between stages")
    parser.add_argument("--train-eval-episodes", type=int, default=20,
                        help="Episodes for periodic eval inside training chunks")
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--train-seed", type=int, default=42,
                        help="Base training seed; each stage/round receives a deterministic offset")
    parser.add_argument("--seed-stride", type=int, default=1000,
                        help="Seed offset between stages; round number is added inside a stage")
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--reports-dir", default="reports/curriculum")
    parser.add_argument("--make-plots", action="store_true",
                        help="Generate thesis-ready plots after the curriculum finishes")
    parser.add_argument("--plots-dir", default="reports/training_plots",
                        help="Output directory for generated training plots")
    parser.add_argument("--plots-formats", default="png,svg",
                        help="Comma-separated plot formats, e.g. png,svg")
    parser.add_argument("--plots-run-filter", default="",
                        help="Optional substring filter for logs included in generated plots")
    parser.add_argument("--run-tag", default="",
                        help="Shared suffix for train.py TensorBoard/eval/diagnostics logs")
    parser.add_argument("--artifact-tag", default="",
                        help="Suffix for model/checkpoint artifacts to avoid overwriting existing runs")
    parser.add_argument("--continue-on-fail", action="store_true",
                        help="Continue to the next stage even if target was not reached")
    parser.add_argument("--allow-scratch", action="store_true",
                        help="Start from scratch if no start checkpoint is found")
    parser.add_argument("--no-diag", action="store_true",
                        help="Disable structured training diagnostics")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    args = parser.parse_args()

    if args.chunk_timesteps < 1:
        parser.error("--chunk-timesteps must be >= 1")
    if args.max_rounds_per_stage < 1:
        parser.error("--max-rounds-per-stage must be >= 1")
    if args.eval_episodes < 1 or args.train_eval_episodes < 1:
        parser.error("eval episode counts must be >= 1")
    if not 0.0 <= args.target_success <= 1.0:
        parser.error("--target-success must be in [0, 1]")
    if args.train_seed < 0 or args.eval_seed < 0:
        parser.error("seeds must be >= 0")
    if args.seed_stride < 1:
        parser.error("--seed-stride must be >= 1")

    root = repo_root()
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    reports_dir = root / args.reports_dir / run_stamp
    args.run_tag = safe_run_tag(args.run_tag)
    args.artifact_tag = safe_run_tag(args.artifact_tag)
    if args.make_plots and not args.run_tag:
        args.run_tag = args.artifact_tag or f"pipeline_{run_stamp}"
    targets = parse_target_overrides(args.target_overrides)
    stages = [stage for stage in split_csv(args.stages) if stage not in set(split_csv(args.skip_stages))]
    plots_state = {"generated": False}

    def maybe_generate_plots() -> None:
        if plots_state["generated"] or not args.make_plots or args.dry_run:
            return
        plots_state["generated"] = True
        plot_tag = args.run_tag or time.strftime("%Y%m%d_%H%M%S")
        command = [
            sys.executable,
            "training/generate_training_plots.py",
            "--reports-dir",
            args.plots_dir,
            "--tag",
            plot_tag,
            "--formats",
            args.plots_formats,
            "--curriculum-reports-dir",
            str(reports_dir),
        ]
        plots_filter = args.plots_run_filter or args.run_tag
        if plots_filter:
            command.extend(["--run-filter", plots_filter])
        try:
            run_command(root, command, dry_run=False)
        except Exception as exc:
            print(f"[PLOTS][WARN] Could not generate plots: {exc}", flush=True)

    atexit.register(maybe_generate_plots)

    current_model: Path | None = Path(args.start_model).resolve() if args.start_model else None
    current_normalize: Path | None = Path(args.start_normalize).resolve() if args.start_normalize else None
    if (current_model is None) != (current_normalize is None):
        parser.error("--start-model and --start-normalize must be provided together")

    if current_model is None:
        for start_obstacle in split_csv(args.start_from):
            found = find_checkpoint(root, args.algo, start_obstacle, args.artifact_tag)
            if found is not None:
                current_model, current_normalize, label = found
                print(
                    f"[START] Using {label} {args.algo} checkpoint from {start_obstacle}: "
                    f"{current_model}",
                    flush=True,
                )
                break

    if current_model is None and not args.allow_scratch:
        raise SystemExit(
            "No start checkpoint found. Finish/point to your static checkpoint with "
            "--start-model and --start-normalize, or pass --allow-scratch."
        )

    print("[CURRICULUM] Stages: " + ", ".join(stages), flush=True)
    print(f"[CURRICULUM] Reports: {reports_dir}", flush=True)
    if args.artifact_tag:
        print(f"[CURRICULUM] Artifact tag: {args.artifact_tag}", flush=True)
    if args.run_tag:
        print(f"[CURRICULUM] Run tag: {args.run_tag}", flush=True)

    for stage_idx, obstacle_type in enumerate(stages):
        target = targets.get(obstacle_type, args.target_success)
        print("\n" + "=" * 72, flush=True)
        print(f"[STAGE] {obstacle_type} target={target:.1%}", flush=True)
        print("=" * 72, flush=True)

        current_success = 0.0
        if current_model is not None and current_normalize is not None:
            current_success = evaluate_policy(
                root=root,
                algo=args.algo,
                model_path=current_model,
                normalize_path=current_normalize,
                obstacle_type=obstacle_type,
                episodes=args.eval_episodes,
                seed=staged_seed(args.eval_seed, stage_idx, 0, args.seed_stride),
                reports_dir=reports_dir,
                tag="precheck",
                dry_run=args.dry_run,
            )
            if current_success >= target:
                print(f"[SKIP] {obstacle_type} already passed: {current_success:.1%}", flush=True)
                continue

        for round_idx in range(1, args.max_rounds_per_stage + 1):
            print(
                f"[TRAIN] {obstacle_type} round {round_idx}/{args.max_rounds_per_stage} "
                f"from success={current_success:.1%}",
                flush=True,
            )
            train_chunk(
                root=root,
                args=args,
                obstacle_type=obstacle_type,
                stage_idx=stage_idx,
                round_idx=round_idx,
                init_model=current_model,
                init_normalize=current_normalize,
                reset_best=(round_idx == 1),
            )
            if args.dry_run:
                print(f"[DRY] Would evaluate produced checkpoint for {obstacle_type}", flush=True)
                break

            found = find_checkpoint(root, args.algo, obstacle_type, args.artifact_tag)
            if found is None:
                raise RuntimeError(f"Training did not produce a usable checkpoint for {obstacle_type}")
            current_model, current_normalize, label = found
            print(f"[CHECKPOINT] Using {label}: {current_model}", flush=True)

            current_success = evaluate_policy(
                root=root,
                algo=args.algo,
                model_path=current_model,
                normalize_path=current_normalize,
                obstacle_type=obstacle_type,
                episodes=args.eval_episodes,
                seed=staged_seed(args.eval_seed, stage_idx, round_idx, args.seed_stride),
                reports_dir=reports_dir,
                tag=f"round{round_idx}",
                dry_run=args.dry_run,
            )
            if current_success >= target:
                print(f"[PASS] {obstacle_type}: {current_success:.1%} >= {target:.1%}", flush=True)
                break
        else:
            message = (
                f"[FAIL] {obstacle_type}: target {target:.1%} was not reached "
                f"after {args.max_rounds_per_stage} rounds; last={current_success:.1%}"
            )
            if args.continue_on_fail:
                print(message, flush=True)
            else:
                raise SystemExit(message)

    print("\n[CURRICULUM] Finished.", flush=True)
    if current_model is not None:
        print(f"[CURRICULUM] Final model: {current_model}", flush=True)
    if current_normalize is not None:
        print(f"[CURRICULUM] Final normalize: {current_normalize}", flush=True)

    maybe_generate_plots()


if __name__ == "__main__":
    main()
