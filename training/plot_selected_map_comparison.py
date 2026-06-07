
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MapSpec:
    key: str
    label: str
    color: str
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()


MAP_SPECS: tuple[MapSpec, ...] = (
    MapSpec(
        key="cylinders",
        label="Цилиндры (Cylinders)",
        color="#1f77b4",
        include=("RecurrentPPO_pretrain_cylinders_enhanced_obs_20260426",),
        exclude=("_rl3", "_rl_cf2x"),
    ),
    MapSpec(
        key="beams",
        label="Балки (Beams)",
        color="#ff7f0e",
        include=("RecurrentPPO_pretrain_beams_enhanced_obs_20260427",),
    ),
    MapSpec(
        key="swinging_sticks",
        label="Качающиеся палки (Swinging sticks)",
        color="#2ca02c",
        include=("RecurrentPPO_pretrain_swinging_sticks_enhanced_obs_20260427",),
    ),
    MapSpec(
        key="city_dynamic",
        label="Динамический город (Dynamic city)",
        color="#d62728",
        include=("RecurrentPPO_pretrain_city_dynamic_enhanced_obs_citydyn_v1",),
        exclude=("construction_site", "consite"),
    ),
)


METRICS: tuple[dict[str, str], ...] = (
    {
        "column": "success_rate",
        "title": "Доля успешных эпизодов",
        "ylabel": "Доля эпизодов",
        "filename": "01_success_rate.png",
        "ylim": "rate",
    },
    {
        "column": "crash_rate",
        "title": "Доля аварий",
        "ylabel": "Доля эпизодов",
        "filename": "02_crash_rate.png",
        "ylim": "rate",
    },
    {
        "column": "timeout_rate",
        "title": "Доля таймаутов",
        "ylabel": "Доля эпизодов",
        "filename": "03_timeout_rate.png",
        "ylim": "rate",
    },
    {
        "column": "reward",
        "title": "Средняя награда за эпизод",
        "ylabel": "Награда",
        "filename": "04_mean_reward.png",
        "ylim": "",
    },
    {
        "column": "dist_to_goal",
        "title": "Финальное расстояние до цели",
        "ylabel": "Метры",
        "filename": "05_final_distance.png",
        "ylim": "positive",
    },
    {
        "column": "progress_ratio",
        "title": "Прогресс к цели",
        "ylabel": "Доля пути",
        "filename": "06_goal_progress.png",
        "ylim": "rate",
    },
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create six large combined plots for selected maps."
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=repo_root() / "logs" / "training_diagnostics",
        help="Directory with structured diagnostics runs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root() / "reports" / "training_plots" / "presentation_map_comparison_20260512",
        help="Output directory for PNG plots and CSV data.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=3,
        help="Rolling smoothing window over milestone points.",
    )
    parser.add_argument(
        "--min-points-per-run",
        type=int,
        default=2,
        help="Skip diagnostic runs with fewer milestone points.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
    return rows


def extract_timestamp(run_name: str) -> str:
    parts = run_name.rsplit("_", 2)
    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
        return f"{parts[-2]}_{parts[-1]}"
    return run_name


def match_runs(diagnostics_dir: Path, spec: MapSpec) -> list[Path]:
    candidates = [path for path in diagnostics_dir.iterdir() if path.is_dir()]
    matched: list[Path] = []
    for path in candidates:
        name = path.name
        if not any(term in name for term in spec.include):
            continue
        if any(term in name for term in spec.exclude):
            continue
        if not (path / "milestones.jsonl").exists():
            continue
        matched.append(path)
    return sorted(matched, key=lambda p: extract_timestamp(p.name))


def flatten_milestone(row: dict[str, Any]) -> dict[str, Any]:
    rates = row.get("rates") or {}
    means = row.get("means") or {}
    return {
        "episode": row.get("episode"),
        "raw_timesteps": row.get("timesteps"),
        "success_rate": rates.get("success"),
        "crash_rate": rates.get("crash"),
        "timeout_rate": rates.get("timeout"),
        "reward": means.get("reward"),
        "dist_to_goal": means.get("dist_to_goal"),
        "progress_ratio": means.get("progress_ratio"),
        "window_size": row.get("window_size"),
    }


def build_map_frame(
    diagnostics_dir: Path,
    spec: MapSpec,
    min_points_per_run: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    cumulative_offset = 0.0

    for run in match_runs(diagnostics_dir, spec):
        raw_rows = read_jsonl(run / "milestones.jsonl")
        if len(raw_rows) < min_points_per_run:
            summary.append({
                "map": spec.key,
                "run": run.name,
                "points": len(raw_rows),
                "used": False,
                "reason": "too_few_points",
            })
            continue

        run_rows = [flatten_milestone(row) for row in raw_rows]
        run_df = pd.DataFrame(run_rows)
        run_df["raw_timesteps"] = pd.to_numeric(run_df["raw_timesteps"], errors="coerce")
        run_df = run_df.dropna(subset=["raw_timesteps"]).sort_values("raw_timesteps")
        if len(run_df) < min_points_per_run:
            summary.append({
                "map": spec.key,
                "run": run.name,
                "points": len(run_df),
                "used": False,
                "reason": "not_enough_numeric_steps",
            })
            continue

        start_step = float(run_df["raw_timesteps"].iloc[0])
        relative_steps = run_df["raw_timesteps"] - start_step


        relative_steps = relative_steps.cummax()
        run_df["map"] = spec.key
        run_df["label"] = spec.label
        run_df["run"] = run.name
        run_df["step"] = cumulative_offset + relative_steps
        rows.extend(run_df.to_dict("records"))

        duration = float(relative_steps.iloc[-1])
        cumulative_offset += max(duration, 1.0)
        summary.append({
            "map": spec.key,
            "run": run.name,
            "points": len(run_df),
            "used": True,
            "start_raw_timesteps": int(start_step),
            "end_raw_timesteps": int(run_df["raw_timesteps"].iloc[-1]),
            "stitched_duration": int(max(duration, 0.0)),
        })

    return pd.DataFrame(rows), summary


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 240,
        "font.family": "DejaVu Sans",
        "font.size": 18,
        "axes.titlesize": 24,
        "axes.labelsize": 21,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 3.6,
    })


def million_tick(value: float, _position: int | None = None) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value) < 1e-9:
        return "0"
    return f"{value / 1_000_000:.1f}M".replace(".0M", "M")


def smooth(values: pd.Series, window: int) -> pd.Series:
    if window <= 1 or len(values) < 3:
        return values
    return values.rolling(window=min(window, len(values)), min_periods=1).mean()


def plot_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: dict[str, str],
    smooth_window: int,
    include_legend: bool = True,
) -> tuple[list[Any], list[str]]:
    handles: list[Any] = []
    labels: list[str] = []
    column = metric["column"]

    for spec in MAP_SPECS:
        part = df[df["map"] == spec.key].copy()
        if part.empty or column not in part:
            continue
        part[column] = pd.to_numeric(part[column], errors="coerce")
        part = part.dropna(subset=["step", column]).sort_values("step")
        if part.empty:
            continue
        values = smooth(part[column], smooth_window)
        (line,) = ax.plot(
            part["step"],
            values,
            color=spec.color,
            label=spec.label,
            solid_capstyle="round",
        )
        handles.append(line)
        labels.append(spec.label)

    ax.set_title(metric["title"], pad=14)
    ax.set_xlabel("Шаги обучения на карте")
    ax.set_ylabel(metric["ylabel"])
    ax.xaxis.set_major_formatter(FuncFormatter(million_tick))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.tick_params(axis="both", which="major", length=7, width=1.2)

    if metric.get("ylim") == "rate":
        ax.set_ylim(-0.03, 1.03)
    elif metric.get("ylim") == "positive":
        ymin = max(0.0, float(pd.to_numeric(df[column], errors="coerce").min(skipna=True)) * 0.85)
        ymax = float(pd.to_numeric(df[column], errors="coerce").max(skipna=True)) * 1.08
        if np.isfinite(ymax) and ymax > ymin:
            ax.set_ylim(ymin, ymax)

    if include_legend and handles:
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=2,
            frameon=False,
            handlelength=3.0,
            columnspacing=1.8,
        )
    return handles, labels


def save_individual_plots(df: pd.DataFrame, out_dir: Path, smooth_window: int) -> None:
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(15.5, 9.2))
        plot_metric(ax, df, metric, smooth_window=smooth_window, include_legend=True)
        fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.28)
        fig.savefig(out_dir / metric["filename"], bbox_inches="tight", pad_inches=0.35)
        plt.close(fig)


def save_combined_sheet(df: pd.DataFrame, out_dir: Path, smooth_window: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(26, 15.5))
    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    for ax, metric in zip(axes.ravel(), METRICS):
        handles, labels = plot_metric(
            ax,
            df,
            metric,
            smooth_window=smooth_window,
            include_legend=False,
        )
        if handles and not legend_handles:
            legend_handles, legend_labels = handles, labels

    fig.suptitle("Сравнение обучения по картам", fontsize=30, y=0.98)
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=2,
            frameon=False,
            fontsize=21,
            handlelength=3.2,
            columnspacing=2.2,
        )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.91, bottom=0.14, wspace=0.28, hspace=0.40)
    fig.savefig(out_dir / "00_all_six_metrics.png", bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for spec in MAP_SPECS:
        frame, summary = build_map_frame(
            args.diagnostics_dir,
            spec,
            min_points_per_run=args.min_points_per_run,
        )
        if not frame.empty:
            frames.append(frame)
        summaries.extend(summary)

    if not frames:
        raise SystemExit("No milestone data found for selected maps.")

    df = pd.concat(frames, ignore_index=True)
    for metric in METRICS:
        column = metric["column"]
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df.to_csv(args.out_dir / "selected_map_curves.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.out_dir / "run_summary.csv", index=False)

    save_individual_plots(df, args.out_dir, smooth_window=max(1, args.smooth_window))
    save_combined_sheet(df, args.out_dir, smooth_window=max(1, args.smooth_window))

    print(f"Saved plots to: {args.out_dir}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()