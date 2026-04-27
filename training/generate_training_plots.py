"""
Generate publication-friendly training plots from TensorBoard, eval history,
and structured diagnostics logs.

The script is intentionally read-only: it does not touch checkpoints or logs.
It writes figures and CSV snapshots under reports/training_plots/<timestamp>.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

try:
    from tensorboard.backend.event_processing import event_accumulator
except Exception:  # pragma: no cover - optional runtime dependency
    event_accumulator = None


FIGURES: list[str] = []


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def flatten_dict(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}" if not prefix else f"{prefix}_{key}"
        if isinstance(value, dict):
            flat.update(flatten_dict(value, name))
        else:
            flat[name] = value
    return flat


def read_jsonl(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def maybe_filter_run(df: pd.DataFrame, run_filter: str) -> pd.DataFrame:
    if not run_filter or df.empty:
        return df

    mask = pd.Series(False, index=df.index)
    for column in ("run_id", "log_name", "source", "event_file"):
        if column in df.columns:
            mask = mask | df[column].astype(str).str.contains(run_filter, case=False, regex=False, na=False)
    return df[mask].copy()


def setup_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (11, 6),
        "figure.dpi": 130,
        "savefig.dpi": 180,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
    })


def sci_tick(value: float, _position: int | None = None) -> str:
    """Compact tick labels for training steps, e.g. 1E5 instead of 100000."""
    if not np.isfinite(value):
        return ""
    if abs(value) < 1e-9:
        return "0"

    sign = "-" if value < 0 else ""
    value = abs(float(value))
    exponent = int(math.floor(math.log10(value)))
    mantissa = value / (10 ** exponent)
    rounded_mantissa = round(mantissa, 1)
    if abs(rounded_mantissa - round(rounded_mantissa)) < 1e-9:
        mantissa_text = str(int(round(rounded_mantissa)))
    else:
        mantissa_text = f"{rounded_mantissa:.1f}".rstrip("0").rstrip(".")
    return f"{sign}{mantissa_text}E{exponent}"


def format_training_step_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(FuncFormatter(sci_tick))


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str]) -> None:
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        FIGURES.append(str(path.relative_to(out_dir)))
    plt.close(fig)


def smooth_series(values: pd.Series, window: int) -> pd.Series:
    if window <= 1 or len(values) < 3:
        return values
    return values.rolling(window=min(window, len(values)), min_periods=1).mean()


def plot_grouped_lines(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    smooth: int = 1,
    label_prefix: str = "",
) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        return

    for group_name, part in df.dropna(subset=[x, y]).groupby(group, dropna=False):
        part = part.sort_values(x)
        y_values = smooth_series(part[y].astype(float), smooth)
        label = str(group_name)
        if label_prefix:
            label = f"{label_prefix}{label}"
        ax.plot(part[x], y_values, marker="o" if len(part) < 30 else None, linewidth=1.8, label=label)


def load_eval_history(eval_dir: Path, run_filter: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(eval_dir.glob("**/eval_history.jsonl")):
        for row in read_jsonl(path):
            row = dict(row)
            row["log_name"] = path.parent.name
            row["source"] = str(path)
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = numeric_columns(
        df,
        [
            "timesteps",
            "mean_reward",
            "mean_ep_length",
            "success_rate",
            "crash_rate",
            "timeout_rate",
            "crash_contact",
            "crash_out_of_bounds",
            "episodes",
        ],
    )
    if "run_id" not in df.columns:
        df["run_id"] = df["log_name"]
    return maybe_filter_run(df, run_filter)


def load_diagnostics(diagnostics_dir: Path, run_filter: str, max_episode_rows: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    milestone_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for run_dir in sorted(path for path in diagnostics_dir.glob("*") if path.is_dir()):
        if run_filter and run_filter.lower() not in run_dir.name.lower():
            continue

        for row in read_jsonl(run_dir / "milestones.jsonl"):
            flat = flatten_dict(row)
            flat["run_id"] = run_dir.name
            milestone_rows.append(flat)

        remaining = max(0, max_episode_rows - len(episode_rows))
        if remaining > 0:
            for row in read_jsonl(run_dir / "episodes_compact.jsonl", max_rows=remaining):
                flat = flatten_dict(row)
                flat["run_id"] = run_dir.name
                episode_rows.append(flat)

        summary_path = run_dir / "summary_blocks.json"
        if summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                flat = flatten_dict(payload)
                flat["run_id"] = run_dir.name
                summary_rows.append(flat)
            except (json.JSONDecodeError, OSError):
                pass

    milestones = pd.DataFrame(milestone_rows)
    episodes = pd.DataFrame(episode_rows)
    summaries = pd.DataFrame(summary_rows)

    milestones = numeric_columns(
        milestones,
        [
            "episode",
            "timesteps",
            "rates_success",
            "rates_crash",
            "rates_timeout",
            "means_reward",
            "means_dist_to_goal",
            "means_progress_ratio",
            "alignment_positive_crash_rate",
        ],
    )
    episodes = numeric_columns(
        episodes,
        [
            "episode",
            "timesteps",
            "episode_reward",
            "episode_length",
            "dist_to_goal",
            "min_goal_distance",
            "progress_ratio",
            "closest_obstacle",
            "avg_speed",
            "avg_heading_error",
            "path_efficiency",
            "hovering_time",
            "spinning_time",
            "goal_seeking_ratio",
        ],
    )
    summaries = numeric_columns(summaries, list(summaries.columns))

    return milestones, episodes, summaries


def load_curriculum_reports(reports_dir: Path, run_filter: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(reports_dir.glob("**/eval_*.json")):
        if run_filter and run_filter.lower() not in str(path).lower():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        summary = payload.get("summary", {})
        stem = path.stem
        obstacle = stem
        tag = ""
        if stem.startswith("eval_"):
            obstacle_and_tag = stem[len("eval_"):]
            obstacle, _, tag = obstacle_and_tag.rpartition("_")
            if not obstacle:
                obstacle = obstacle_and_tag
        row = flatten_dict(summary)
        row.update({
            "obstacle_type": obstacle,
            "eval_tag": tag,
            "source": str(path),
            "report_run": path.parent.name,
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    return numeric_columns(
        df,
        ["success_rate", "crash_rate", "timeout_rate", "mean_reward", "mean_steps", "mean_final_dist"],
    )


def load_tensorboard_scalars(logs_dir: Path, run_filter: str) -> pd.DataFrame:
    if event_accumulator is None:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for event_file in sorted(logs_dir.glob("**/events.out.tfevents.*")):
        if run_filter and run_filter.lower() not in str(event_file).lower():
            continue
        try:
            accumulator = event_accumulator.EventAccumulator(
                str(event_file),
                size_guidance={event_accumulator.SCALARS: 0},
            )
            accumulator.Reload()
            tags = accumulator.Tags().get("scalars", [])
        except Exception:
            continue

        log_name = event_file.parent.name
        for tag in tags:
            try:
                scalars = accumulator.Scalars(tag)
            except Exception:
                continue
            for scalar in scalars:
                rows.append({
                    "log_name": log_name,
                    "event_file": str(event_file),
                    "tag": tag,
                    "step": int(scalar.step),
                    "wall_time": float(scalar.wall_time),
                    "value": float(scalar.value),
                })

    return pd.DataFrame(rows)


def plot_eval_history(df: pd.DataFrame, out_dir: Path, formats: list[str], smooth: int) -> None:
    if df.empty:
        return

    group = "run_id" if "run_id" in df.columns else "log_name"
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    plot_grouped_lines(axes[0, 0], df, "timesteps", "success_rate", group, smooth)
    plot_grouped_lines(axes[0, 0], df, "timesteps", "crash_rate", group, smooth, "crash ")
    plot_grouped_lines(axes[0, 0], df, "timesteps", "timeout_rate", group, smooth, "timeout ")
    axes[0, 0].set_title("Evaluation outcomes")
    axes[0, 0].set_ylabel("rate")
    axes[0, 0].set_ylim(-0.03, 1.03)

    plot_grouped_lines(axes[0, 1], df, "timesteps", "mean_reward", group, smooth)
    axes[0, 1].set_title("Evaluation mean reward")
    axes[0, 1].set_ylabel("reward")

    plot_grouped_lines(axes[1, 0], df, "timesteps", "mean_ep_length", group, smooth)
    axes[1, 0].set_title("Evaluation episode length")
    axes[1, 0].set_ylabel("steps")
    axes[1, 0].set_xlabel("training steps")

    if "crash_contact" in df.columns:
        plot_grouped_lines(axes[1, 1], df, "timesteps", "crash_contact", group, smooth, "contact ")
    if "crash_out_of_bounds" in df.columns:
        plot_grouped_lines(axes[1, 1], df, "timesteps", "crash_out_of_bounds", group, smooth, "oob ")
    axes[1, 1].set_title("Evaluation crash breakdown")
    axes[1, 1].set_ylabel("episodes")
    axes[1, 1].set_xlabel("training steps")

    for ax in axes.ravel():
        format_training_step_axis(ax)
        ax.legend(loc="best")
    fig.suptitle("Periodic evaluation metrics", y=1.02)
    save_figure(fig, out_dir, "01_eval_history", formats)


def plot_milestones(df: pd.DataFrame, out_dir: Path, formats: list[str], smooth: int) -> None:
    if df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    plot_grouped_lines(axes[0, 0], df, "timesteps", "rates_success", "run_id", smooth, "success ")
    plot_grouped_lines(axes[0, 0], df, "timesteps", "rates_crash", "run_id", smooth, "crash ")
    plot_grouped_lines(axes[0, 0], df, "timesteps", "rates_timeout", "run_id", smooth, "timeout ")
    axes[0, 0].set_title("Rolling training outcomes")
    axes[0, 0].set_ylabel("rate")
    axes[0, 0].set_ylim(-0.03, 1.03)

    plot_grouped_lines(axes[0, 1], df, "timesteps", "means_reward", "run_id", smooth)
    axes[0, 1].set_title("Rolling episode reward")
    axes[0, 1].set_ylabel("reward")

    plot_grouped_lines(axes[1, 0], df, "timesteps", "means_progress_ratio", "run_id", smooth)
    axes[1, 0].set_title("Rolling path progress")
    axes[1, 0].set_ylabel("progress ratio")
    axes[1, 0].set_ylim(-0.03, 1.03)
    axes[1, 0].set_xlabel("training steps")

    plot_grouped_lines(axes[1, 1], df, "timesteps", "means_dist_to_goal", "run_id", smooth)
    axes[1, 1].set_title("Rolling final distance to goal")
    axes[1, 1].set_ylabel("meters")
    axes[1, 1].set_xlabel("training steps")

    for ax in axes.ravel():
        format_training_step_axis(ax)
        ax.legend(loc="best")
    fig.suptitle("Structured diagnostics rolling windows", y=1.02)
    save_figure(fig, out_dir, "02_training_milestones", formats)


def plot_tensorboard(df: pd.DataFrame, out_dir: Path, formats: list[str], smooth: int) -> None:
    if df.empty:
        return

    tag_sets = [
        (
            "03_policy_losses",
            "Policy optimization losses",
            [
                "train/policy_gradient_loss",
                "train/value_loss",
                "train/entropy_loss",
                "train/loss",
                "train/approx_kl",
                "train/clip_fraction",
            ],
        ),
        (
            "04_optimization_diagnostics",
            "Optimization diagnostics",
            [
                "train/explained_variance",
                "train/std",
                "train/learning_rate",
                "rollout/ep_rew_mean",
                "rollout/ep_len_mean",
                "time/fps",
            ],
        ),
    ]

    labels = {
        "train/policy_gradient_loss": "policy gradient loss",
        "train/value_loss": "value loss",
        "train/entropy_loss": "entropy loss",
        "train/loss": "total loss",
        "train/approx_kl": "approx KL",
        "train/clip_fraction": "clip fraction",
        "train/explained_variance": "explained variance",
        "train/std": "policy std",
        "train/learning_rate": "learning rate",
        "rollout/ep_rew_mean": "rollout reward mean",
        "rollout/ep_len_mean": "rollout length mean",
        "time/fps": "fps",
    }

    for stem, title, tags in tag_sets:
        present = [tag for tag in tags if tag in set(df["tag"])]
        if not present:
            continue

        cols = 2
        rows = math.ceil(len(present) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(13, 3.6 * rows), squeeze=False)
        for ax, tag in zip(axes.ravel(), present):
            part = df[df["tag"] == tag].copy()
            plot_grouped_lines(ax, part, "step", "value", "log_name", smooth)
            ax.set_title(labels.get(tag, tag))
            ax.set_xlabel("training steps")
            ax.set_ylabel("value")
            format_training_step_axis(ax)
            ax.legend(loc="best")
        for ax in axes.ravel()[len(present):]:
            ax.axis("off")
        fig.suptitle(title, y=1.01)
        save_figure(fig, out_dir, stem, formats)


def plot_reward_components(summary_df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    if summary_df.empty:
        return

    prefix = "reward_component_means_"
    component_cols = [col for col in summary_df.columns if col.startswith(prefix)]
    component_cols = [col for col in component_cols if not col.startswith("reward_component_means_bad_only_")]
    if not component_cols:
        return

    values = summary_df[component_cols].mean(numeric_only=True).sort_values(key=lambda s: s.abs(), ascending=False)
    values = values.head(18)
    labels = [col[len(prefix):] for col in values.index]

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ["#2b8a3e" if value >= 0 else "#c92a2a" for value in values]
    ax.barh(labels[::-1], values.iloc[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Mean reward component contribution")
    ax.set_xlabel("mean reward per step / terminal component")
    save_figure(fig, out_dir, "05_reward_components", formats)


def plot_failure_breakdown(summary_df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    if summary_df.empty:
        return

    prefix = "failure_reason_breakdown_"
    cols = [col for col in summary_df.columns if col.startswith(prefix)]
    if not cols:
        return

    data = summary_df.set_index("run_id")[cols].copy()
    data.columns = [col[len(prefix):] for col in cols]
    data = data.fillna(0)
    keep = [col for col in data.columns if data[col].sum() > 0]
    if not keep:
        return
    data = data[keep]

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(data))
    x = np.arange(len(data.index))
    for col in data.columns:
        values = data[col].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=col)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(data.index, rotation=30, ha="right")
    ax.set_title("Failure reason breakdown")
    ax.set_ylabel("episodes")
    ax.legend(loc="best")
    save_figure(fig, out_dir, "06_failure_breakdown", formats)


def plot_episode_cloud(episodes_df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    if episodes_df.empty or "timesteps" not in episodes_df.columns:
        return

    colors = {
        "success": "#2b8a3e",
        "crash": "#c92a2a",
        "timeout": "#f08c00",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    metrics = [
        ("progress_ratio", "Progress ratio"),
        ("dist_to_goal", "Final distance to goal"),
        ("closest_obstacle", "Closest obstacle distance"),
        ("avg_heading_error", "Average heading error"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        if metric not in episodes_df.columns:
            ax.axis("off")
            continue
        for outcome, part in episodes_df.groupby("outcome"):
            ax.scatter(
                part["timesteps"],
                part[metric],
                s=14,
                alpha=0.55,
                label=str(outcome),
                color=colors.get(str(outcome), "#495057"),
            )
        ax.set_title(title)
        ax.set_xlabel("training steps")
        ax.set_ylabel(metric)
        format_training_step_axis(ax)
        ax.legend(loc="best")
    fig.suptitle("Episode-level diagnostics", y=1.02)
    save_figure(fig, out_dir, "07_episode_diagnostics", formats)


def plot_behavior_by_outcome(summary_df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    if summary_df.empty:
        return

    metrics = [
        "avg_speed",
        "avg_heading_error",
        "closest_obstacle",
        "progress_ratio",
        "hovering_time",
        "spinning_time",
        "goal_seeking_ratio",
        "path_efficiency",
    ]
    outcomes = ["success", "crash", "timeout"]
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        for metric in metrics:
            col = f"outcome_means_{outcome}_{metric}"
            if col in summary_df.columns:
                rows.append({
                    "outcome": outcome,
                    "metric": metric,
                    "value": pd.to_numeric(summary_df[col], errors="coerce").mean(),
                })

    data = pd.DataFrame(rows).dropna()
    if data.empty:
        return

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, metric in zip(axes.ravel(), metrics):
        part = data[data["metric"] == metric]
        if part.empty:
            ax.axis("off")
            continue
        ax.bar(part["outcome"], part["value"], color=["#2b8a3e", "#c92a2a", "#f08c00"][:len(part)])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Behavior metrics grouped by outcome", y=1.02)
    save_figure(fig, out_dir, "08_behavior_by_outcome", formats)


def plot_curriculum_reports(reports_df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    if reports_df.empty or "obstacle_type" not in reports_df.columns:
        return

    reports_df = reports_df.copy()
    if "success_rate" not in reports_df.columns:
        return
    reports_df["order"] = np.arange(len(reports_df))
    last = reports_df.sort_values("order").groupby("obstacle_type", as_index=False).tail(1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(last["obstacle_type"], last["success_rate"], color="#1c7ed6")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Final curriculum success by obstacle")
    axes[0].set_ylabel("success rate")
    axes[0].tick_params(axis="x", rotation=35)

    for metric, label in [("crash_rate", "crash"), ("timeout_rate", "timeout")]:
        if metric in last.columns:
            axes[1].plot(last["obstacle_type"], last[metric], marker="o", label=label)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Final curriculum failure rates")
    axes[1].set_ylabel("rate")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].legend(loc="best")
    save_figure(fig, out_dir, "09_curriculum_obstacle_summary", formats)


def write_overview(
    out_dir: Path,
    eval_df: pd.DataFrame,
    milestones_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    summaries_df: pd.DataFrame,
    tb_df: pd.DataFrame,
    reports_df: pd.DataFrame,
) -> None:
    lines = [
        "# Training Plot Report",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Input rows",
        f"- eval history rows: {len(eval_df)}",
        f"- milestone rows: {len(milestones_df)}",
        f"- compact episode rows: {len(episodes_df)}",
        f"- summary rows: {len(summaries_df)}",
        f"- tensorboard scalar rows: {len(tb_df)}",
        f"- curriculum report rows: {len(reports_df)}",
        "",
        "## Generated figures",
    ]
    for figure in FIGURES:
        lines.append(f"- {figure}")

    if not eval_df.empty and "success_rate" in eval_df.columns:
        best = eval_df.sort_values(["success_rate", "mean_reward"], ascending=[False, False]).head(1)
        if not best.empty:
            row = best.iloc[0]
            lines.extend([
                "",
                "## Best eval row",
                f"- run_id: {row.get('run_id', row.get('log_name', ''))}",
                f"- timesteps: {row.get('timesteps', '')}",
                f"- success_rate: {row.get('success_rate', '')}",
                f"- mean_reward: {row.get('mean_reward', '')}",
            ])

    lines.extend([
        "",
        "## Notes for thesis use",
        "- 01_eval_history: success/crash/timeout and reward during periodic evaluation.",
        "- 03_policy_losses and 04_optimization_diagnostics: PPO optimization signals from TensorBoard.",
        "- 05_reward_components: reward shaping contribution sanity check.",
        "- 07_episode_diagnostics and 08_behavior_by_outcome: navigation behavior and failure analysis.",
    ])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_csvs(
    out_dir: Path,
    eval_df: pd.DataFrame,
    milestones_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    summaries_df: pd.DataFrame,
    tb_df: pd.DataFrame,
    reports_df: pd.DataFrame,
) -> None:
    datasets = {
        "eval_history.csv": eval_df,
        "diagnostics_milestones.csv": milestones_df,
        "diagnostics_episodes_compact.csv": episodes_df,
        "diagnostics_summary_blocks.csv": summaries_df,
        "tensorboard_scalars.csv": tb_df,
        "curriculum_eval_reports.csv": reports_df,
    }
    for filename, df in datasets.items():
        if not df.empty:
            df.to_csv(out_dir / filename, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots from training logs")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--eval-dir", default="logs/eval")
    parser.add_argument("--diagnostics-dir", default="logs/training_diagnostics")
    parser.add_argument("--curriculum-reports-dir", default="reports/curriculum")
    parser.add_argument("--reports-dir", default="reports/training_plots")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--run-filter", default="")
    parser.add_argument("--formats", default="png,svg")
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--max-episode-rows", type=int, default=200_000)
    args = parser.parse_args()

    root = repo_root()
    setup_style()

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (root / args.reports_dir / tag).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = parse_csv(args.formats) or ["png"]

    eval_df = load_eval_history(root / args.eval_dir, args.run_filter)
    milestones_df, episodes_df, summaries_df = load_diagnostics(
        root / args.diagnostics_dir,
        args.run_filter,
        max_episode_rows=args.max_episode_rows,
    )
    tb_df = load_tensorboard_scalars(root / args.logs_dir, args.run_filter)
    reports_df = load_curriculum_reports(root / args.curriculum_reports_dir, args.run_filter)

    save_csvs(out_dir, eval_df, milestones_df, episodes_df, summaries_df, tb_df, reports_df)
    plot_eval_history(eval_df, out_dir, formats, args.smooth_window)
    plot_milestones(milestones_df, out_dir, formats, args.smooth_window)
    plot_tensorboard(tb_df, out_dir, formats, args.smooth_window)
    plot_reward_components(summaries_df, out_dir, formats)
    plot_failure_breakdown(summaries_df, out_dir, formats)
    plot_episode_cloud(episodes_df, out_dir, formats)
    plot_behavior_by_outcome(summaries_df, out_dir, formats)
    plot_curriculum_reports(reports_df, out_dir, formats)
    write_overview(out_dir, eval_df, milestones_df, episodes_df, summaries_df, tb_df, reports_df)

    print(f"[PLOTS] Saved report to: {out_dir}")
    for figure in FIGURES:
        print(f"[PLOTS]   {figure}")


if __name__ == "__main__":
    main()
