
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
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

try:
    from tensorboard.backend.event_processing import event_accumulator
except Exception:
    event_accumulator = None

FIGURES: list[str] = []
RAW_LABELS: dict[str, str] = {}
SHORT_LABELS: dict[str, str] = {}
LEGEND_LABELS: dict[str, str] = {}
STEP_AXIS_LABEL = "Шаги обучения"
LINE_ONLY = False
PRESENTATION_STYLE = False
LINE_WIDTH = 1.8
LEGEND_FONTSIZE = 7

OBSTACLE_LABELS = {
    "crossing_spheres": "Пересекающиеся сферы",
    "swinging_sticks": "Качающиеся палки",
    "dynamic_mix": "Динамический микс",
    "city_blocks": "Город",
    "city_dynamic": "Дин. город",
    "construction_site_dynamic": "Стройка",
    "cylinders": "Цилиндры",
    "spheres": "Сферы",
    "beams": "Балки",
    "walls": "Стены",
    "gates": "Ворота",
    "boxes": "Блоки",
    "slalom": "slalom",
    "empty": "Пустая карта",
}

PREFIX_LABELS = {
    "success": "Успех",
    "crash": "Авария",
    "timeout": "Таймаут",
    "contact": "Контакт",
    "oob": "Вылет",
}

OUTCOME_LABELS = {
    "success": "Успех",
    "crash": "Авария",
    "timeout": "Таймаут",
}

METRIC_LABELS = {
    "action_smoothness": "Плавность действий",
    "avg_heading_error": "Средняя ошибка направления",
    "avg_speed": "Средняя скорость",
    "boundary_dist": "Расстояние до границы",
    "closest_obstacle": "Расстояние до препятствия",
    "dist_to_goal": "Расстояние до цели",
    "episode_length": "Длина эпизода",
    "episode_reward": "Награда за эпизод",
    "goal_seeking_ratio": "Доля движения к цели",
    "hovering_time": "Доля зависания",
    "min_goal_distance": "Минимальное расстояние до цели",
    "path_efficiency": "Эффективность пути",
    "progress_ratio": "Прогресс к цели",
    "spinning_time": "Доля вращения",
}

REWARD_COMPONENT_LABELS = {
    "boundary": "Граница арены",
    "boundary_outward": "Движение наружу",
    "efficiency_bonus": "Бонус эффективности",
    "exploration": "Исследование",
    "heading": "Направление на цель",
    "near_goal_away": "Удаление рядом с целью",
    "near_goal_boundary": "Граница рядом с целью",
    "near_goal_capture": "Захват цели",
    "near_goal_progress": "Прогресс рядом с целью",
    "near_goal_speed": "Скорость рядом с целью",
    "near_goal_stall": "Застревание рядом с целью",
    "obstacle": "Близость препятствий",
    "obstacle_approach": "Сближение с препятствием",
    "obstacle_hard": "Опасная близость",
    "progress": "Прогресс",
    "proximity": "Близость к цели",
    "smoothness": "Плавность",
    "step_penalty": "Штраф за шаг",
    "terminal": "Финальный исход",
    "velocity": "Скорость",
    "yaw_penalty": "Штраф поворота",
}

FAILURE_LABELS = {
    "collision": "Столкновения",
    "out_of_bounds": "Вылеты за границу",
    "success": "Успехи",
    "timeout": "Таймауты",
}

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

def filter_terms(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]

def text_matches_filters(text: Any, include_filter: str, exclude_filter: str = "") -> bool:
    haystack = str(text).lower()
    include_terms = filter_terms(include_filter)
    exclude_terms = filter_terms(exclude_filter)
    if include_terms and not any(term in haystack for term in include_terms):
        return False
    return not any(term in haystack for term in exclude_terms)

def maybe_filter_run(df: pd.DataFrame, run_filter: str, exclude_run_filter: str = "") -> pd.DataFrame:
    if (not run_filter and not exclude_run_filter) or df.empty:
        return df

    include_terms = filter_terms(run_filter)
    exclude_terms = filter_terms(exclude_run_filter)
    include_mask = pd.Series(not include_terms, index=df.index)
    exclude_mask = pd.Series(False, index=df.index)
    for column in ("run_id", "log_name", "source", "event_file"):
        if column in df.columns:
            values = df[column].astype(str).str.lower()
            for term in include_terms:
                include_mask = include_mask | values.str.contains(term, regex=False, na=False)
            for term in exclude_terms:
                exclude_mask = exclude_mask | values.str.contains(term, regex=False, na=False)
    return df[include_mask & ~exclude_mask].copy()

def setup_style(presentation: bool = False) -> None:
    global PRESENTATION_STYLE, LINE_WIDTH, LEGEND_FONTSIZE

    PRESENTATION_STYLE = presentation
    if presentation:
        LINE_WIDTH = 3.2
        LEGEND_FONTSIZE = 18
        plt.rcParams.update({
            "figure.figsize": (16, 10),
            "figure.dpi": 150,
            "savefig.dpi": 260,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 18,
            "axes.titlesize": 23,
            "axes.labelsize": 21,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": LEGEND_FONTSIZE,
            "lines.linewidth": LINE_WIDTH,
        })
        return

    LINE_WIDTH = 1.8
    LEGEND_FONTSIZE = 7
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

def readable_prefix(value: str) -> str:
    key = value.strip().lower()
    return PREFIX_LABELS.get(key, value.strip())

def with_readable_prefix(prefix: str, label: str) -> str:
    prefix = prefix.strip()
    if not prefix:
        return label
    return f"{readable_prefix(prefix)} — {label}"

def compact_group_label(value: Any) -> str:
    raw = str(value)
    if raw in RAW_LABELS:
        return RAW_LABELS[raw]

    lowered = raw.lower()
    obstacle = next((short for key, short in OBSTACLE_LABELS.items() if key in lowered), None)
    if obstacle is None:
        obstacle = raw[:18]

    tag = ""
    tag_match = re.search(r"(rl\d*(?:_[A-Za-z0-9]+){1,2})", raw)
    if tag_match:
        tag = tag_match.group(1)
    else:
        stamps = re.findall(r"(\d{8})_(\d{6})", raw)
        if stamps:
            date, clock = stamps[-1]
            tag = f"{date[4:8]}-{clock[:4]}"

    base = f"{obstacle} ({tag})" if tag else obstacle
    label = base
    suffix = 2
    while label in SHORT_LABELS and SHORT_LABELS[label] != raw:
        label = f"{base}#{suffix}"
        suffix += 1

    RAW_LABELS[raw] = label
    SHORT_LABELS[label] = raw
    return label

def run_sort_key(value: Any) -> tuple[int, str]:
    raw = str(value)
    stamps = re.findall(r"(\d{8})_(\d{6})", raw)
    if stamps:
        date, clock = stamps[-1]
        return int(f"{date}{clock}"), raw
    return 0, raw

def stitch_run_steps(
    df: pd.DataFrame,
    x_col: str,
    group_col: str,
    gap_steps: float = 0.0,
) -> pd.DataFrame:
    if df.empty or x_col not in df.columns or group_col not in df.columns:
        return df

    stitched_parts: list[pd.DataFrame] = []
    offset = 0.0
    gap = max(0.0, float(gap_steps))
    groups = sorted(df.groupby(group_col, dropna=False), key=lambda item: run_sort_key(item[0]))

    for order, (group_name, part) in enumerate(groups):
        part = part.copy()
        numeric_steps = pd.to_numeric(part[x_col], errors="coerce")
        if numeric_steps.dropna().empty:
            stitched_parts.append(part)
            continue

        start_step = float(numeric_steps.min())
        raw_col = f"raw_{x_col}"
        part[raw_col] = part[x_col]
        part[x_col] = numeric_steps - start_step + offset
        part["stitched_offset"] = offset
        part["stitched_run_order"] = order

        run_width = float(pd.to_numeric(part[x_col], errors="coerce").max() - offset)
        offset += max(0.0, run_width) + gap
        stitched_parts.append(part)

    if not stitched_parts:
        return df
    return pd.concat(stitched_parts, ignore_index=True)

def add_tensorboard_run_ids(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "event_file" not in df.columns:
        return df

    df = df.copy()
    run_ids: dict[str, str] = {}
    for event_file, part in df.groupby("event_file", dropna=False):
        log_name = str(part["log_name"].iloc[0]) if "log_name" in part.columns and not part.empty else "tensorboard"
        wall_time = pd.to_numeric(part["wall_time"], errors="coerce").min() if "wall_time" in part.columns else pd.NA
        if pd.notna(wall_time):
            stamp = datetime.fromtimestamp(float(wall_time)).strftime("%Y%m%d_%H%M%S")
            run_ids[str(event_file)] = f"{log_name}_{stamp}"
        else:
            run_ids[str(event_file)] = f"{log_name}_{len(run_ids) + 1}"

    df["run_id"] = df["event_file"].map(lambda item: run_ids.get(str(item), str(item)))
    return df

def collect_figure_legend(fig: plt.Figure) -> tuple[list[Any], list[str]]:
    deduped: dict[str, Any] = {}
    for ax in fig.axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            deduped.setdefault(label, handle)
        existing = ax.get_legend()
        if existing is not None:
            existing.remove()

    labels = list(deduped.keys())
    handles = [deduped[label] for label in labels]
    return handles, labels

def legend_bottom_space(label_count: int, max_entries: int) -> float:
    if label_count <= 0 or label_count > max_entries:
        return 0.07 if PRESENTATION_STYLE else 0.05
    ncols = min(4, label_count)
    nrows = math.ceil(label_count / ncols)
    if PRESENTATION_STYLE:
        return min(0.42, 0.14 + nrows * 0.07)
    return min(0.34, 0.08 + nrows * 0.04)

def place_figure_legend(fig: plt.Figure, handles: list[Any], labels: list[str], max_entries: int = 24) -> None:
    if not handles or len(labels) > max_entries:
        return

    ncols = min(3 if PRESENTATION_STYLE else 4, len(labels))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02 if PRESENTATION_STYLE else 0.012),
        ncol=ncols,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.6 if PRESENTATION_STYLE else 2.0,
        columnspacing=1.4 if PRESENTATION_STYLE else 1.0,
        labelspacing=0.8 if PRESENTATION_STYLE else 0.5,
    )

def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str]) -> None:
    handles, labels = collect_figure_legend(fig)
    bottom = legend_bottom_space(len(labels), max_entries=24)
    try:
        rect = (0.035, bottom, 0.98, 0.92) if PRESENTATION_STYLE else (0.02, bottom, 0.98, 0.94)
        fig.tight_layout(rect=rect, h_pad=3.8 if PRESENTATION_STYLE else 2.8, w_pad=2.6 if PRESENTATION_STYLE else 1.8)
    except ValueError:
        fig.subplots_adjust(
            left=0.09,
            right=0.98,
            top=0.88 if PRESENTATION_STYLE else 0.90,
            bottom=bottom,
            hspace=0.65 if PRESENTATION_STYLE else 0.55,
            wspace=0.35 if PRESENTATION_STYLE else 0.30,
        )
    place_figure_legend(fig, handles, labels)
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.35 if PRESENTATION_STYLE else 0.1)
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
        label = compact_group_label(group_name)
        if label_prefix:
            label = with_readable_prefix(label_prefix, label)
        LEGEND_LABELS[label] = with_readable_prefix(label_prefix, str(group_name)) if label_prefix else str(group_name)
        marker = None if LINE_ONLY else ("o" if len(part) < 30 else None)
        ax.plot(part[x], y_values, marker=marker, linewidth=LINE_WIDTH, label=label)

def merge_runs_for_plots(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty or not label:
        return df
    df = df.copy()
    for column in ("run_id", "log_name"):
        if column in df.columns:
            df[column] = label
    return df

def load_eval_history(eval_dir: Path, run_filter: str, exclude_run_filter: str = "") -> pd.DataFrame:
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
    return maybe_filter_run(df, run_filter, exclude_run_filter)

def load_diagnostics(
    diagnostics_dir: Path,
    run_filter: str,
    exclude_run_filter: str,
    max_episode_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    milestone_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for run_dir in sorted(path for path in diagnostics_dir.glob("*") if path.is_dir()):
        if not text_matches_filters(run_dir.name, run_filter, exclude_run_filter):
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

def load_curriculum_reports(reports_dir: Path, run_filter: str, exclude_run_filter: str = "") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    apply_run_filter = (
        bool(run_filter)
        and reports_dir.resolve() == (repo_root() / "reports" / "curriculum").resolve()
    )
    for path in sorted(reports_dir.glob("**/eval_*.json")):
        if apply_run_filter and not text_matches_filters(path, run_filter, exclude_run_filter):
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

def load_tensorboard_scalars(logs_dir: Path, run_filter: str, exclude_run_filter: str = "") -> pd.DataFrame:
    if event_accumulator is None:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for event_file in sorted(logs_dir.glob("**/events.out.tfevents.*")):
        if not text_matches_filters(event_file, run_filter, exclude_run_filter):
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
    fig_size = (18, 12) if PRESENTATION_STYLE else (13, 8)
    fig, axes = plt.subplots(2, 2, figsize=fig_size, sharex=True)
    plot_grouped_lines(axes[0, 0], df, "timesteps", "success_rate", group, smooth, "success ")
    plot_grouped_lines(axes[0, 0], df, "timesteps", "crash_rate", group, smooth, "crash ")
    plot_grouped_lines(axes[0, 0], df, "timesteps", "timeout_rate", group, smooth, "timeout ")
    axes[0, 0].set_title("Результаты контрольной оценки")
    axes[0, 0].set_ylabel("Доля эпизодов")
    axes[0, 0].set_ylim(-0.03, 1.03)

    plot_grouped_lines(axes[0, 1], df, "timesteps", "mean_reward", group, smooth)
    axes[0, 1].set_title("Средняя награда на оценке")
    axes[0, 1].set_ylabel("Награда")

    plot_grouped_lines(axes[1, 0], df, "timesteps", "mean_ep_length", group, smooth)
    axes[1, 0].set_title("Средняя длина эпизода на оценке")
    axes[1, 0].set_ylabel("Шаги среды")
    axes[1, 0].set_xlabel(STEP_AXIS_LABEL)

    if "crash_contact" in df.columns:
        plot_grouped_lines(axes[1, 1], df, "timesteps", "crash_contact", group, smooth, "contact ")
    if "crash_out_of_bounds" in df.columns:
        plot_grouped_lines(axes[1, 1], df, "timesteps", "crash_out_of_bounds", group, smooth, "oob ")
    axes[1, 1].set_title("Причины аварий на оценке")
    axes[1, 1].set_ylabel("Количество эпизодов")
    axes[1, 1].set_xlabel(STEP_AXIS_LABEL)

    for ax in axes.ravel():
        format_training_step_axis(ax)
    fig.suptitle("Периодическая оценка модели", y=1.02)
    save_figure(fig, out_dir, "01_eval_history", formats)

def plot_milestones(df: pd.DataFrame, out_dir: Path, formats: list[str], smooth: int) -> None:
    if df.empty:
        return

    fig_size = (18, 12) if PRESENTATION_STYLE else (13, 8)
    fig, axes = plt.subplots(2, 2, figsize=fig_size, sharex=True)
    plot_grouped_lines(axes[0, 0], df, "timesteps", "rates_success", "run_id", smooth, "success ")
    plot_grouped_lines(axes[0, 0], df, "timesteps", "rates_crash", "run_id", smooth, "crash ")
    plot_grouped_lines(axes[0, 0], df, "timesteps", "rates_timeout", "run_id", smooth, "timeout ")
    axes[0, 0].set_title("Исходы эпизодов в обучении")
    axes[0, 0].set_ylabel("Доля эпизодов")
    axes[0, 0].set_ylim(-0.03, 1.03)

    plot_grouped_lines(axes[0, 1], df, "timesteps", "means_reward", "run_id", smooth)
    axes[0, 1].set_title("Средняя награда за эпизод")
    axes[0, 1].set_ylabel("Награда")

    plot_grouped_lines(axes[1, 0], df, "timesteps", "means_progress_ratio", "run_id", smooth)
    axes[1, 0].set_title("Прогресс к цели")
    axes[1, 0].set_ylabel("Доля пройденного пути")
    axes[1, 0].set_ylim(-0.03, 1.03)
    axes[1, 0].set_xlabel(STEP_AXIS_LABEL)

    plot_grouped_lines(axes[1, 1], df, "timesteps", "means_dist_to_goal", "run_id", smooth)
    axes[1, 1].set_title("Финальное расстояние до цели")
    axes[1, 1].set_ylabel("Метры")
    axes[1, 1].set_xlabel(STEP_AXIS_LABEL)

    for ax in axes.ravel():
        format_training_step_axis(ax)
    fig.suptitle("Диагностика обучения по скользящему окну", y=1.02)
    save_figure(fig, out_dir, "02_training_milestones", formats)

def plot_tensorboard(df: pd.DataFrame, out_dir: Path, formats: list[str], smooth: int, plot_set: str) -> None:
    if df.empty:
        return

    full_only_tag_sets = [
        (
            "03_policy_losses",
            "Потери и ограничения PPO",
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
            "Диагностика оптимизации PPO",
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
    core_tag_sets = [
        (
            "10_rl_reward_success_curves",
            "Награда и исходы обучения",
            [
                "rollout/ep_rew_mean",
                "eval/mean_reward",
                "rollout/success_rate",
                "eval/success_rate",
                "eval/crash_rate",
                "eval/timeout_rate",
            ],
        ),
        (
            "11_rl_loss_kl_entropy_curves",
            "Потери, KL-дивергенция и качество критика",
            [
                "train/loss",
                "train/policy_gradient_loss",
                "train/value_loss",
                "train/entropy_loss",
                "train/approx_kl",
                "train/explained_variance",
            ],
        ),
        (
            "12_rl_policy_update_runtime_curves",
            "Обновление политики и скорость обучения",
            [
                "train/std",
                "train/clip_fraction",
                "train/clip_range",
                "train/learning_rate",
                "rollout/ep_len_mean",
                "eval/mean_ep_length",
                "time/fps",
            ],
        ),
    ]
    tag_sets = full_only_tag_sets + core_tag_sets if plot_set == "full" else core_tag_sets

    labels = {
        "train/policy_gradient_loss": "Потеря политики",
        "train/value_loss": "Потеря критика",
        "train/entropy_loss": "Потеря энтропии",
        "train/loss": "Общая потеря",
        "train/approx_kl": "Приближенная KL-дивергенция",
        "train/clip_fraction": "Доля обрезанных обновлений",
        "train/clip_range": "Порог обрезки PPO",
        "train/explained_variance": "Объясненная дисперсия критика",
        "train/std": "Стандартное отклонение политики",
        "train/learning_rate": "Скорость обучения",
        "rollout/ep_rew_mean": "Средняя награда в обучении",
        "rollout/ep_len_mean": "Средняя длина эпизода в обучении",
        "rollout/success_rate": "Доля успехов в обучении",
        "eval/mean_reward": "Средняя награда на оценке",
        "eval/success_rate": "Доля успехов на оценке",
        "eval/crash_rate": "Доля аварий на оценке",
        "eval/timeout_rate": "Доля таймаутов на оценке",
        "eval/mean_ep_length": "Средняя длина эпизода на оценке",
        "time/fps": "Скорость симуляции, FPS",
    }

    for stem, title, tags in tag_sets:
        present = [tag for tag in tags if tag in set(df["tag"])]
        if not present:
            continue

        cols = 2
        rows = math.ceil(len(present) / cols)
        fig_size = (18, 5.4 * rows) if PRESENTATION_STYLE else (13, 3.6 * rows)
        fig, axes = plt.subplots(rows, cols, figsize=fig_size, squeeze=False)
        for ax, tag in zip(axes.ravel(), present):
            part = df[df["tag"] == tag].copy()
            group_col = "run_id" if "run_id" in part.columns else "log_name"
            plot_grouped_lines(ax, part, "step", "value", group_col, smooth)
            ax.set_title(labels.get(tag, tag))
            ax.set_xlabel(STEP_AXIS_LABEL)
            ax.set_ylabel("Значение")
            format_training_step_axis(ax)
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
    labels = [REWARD_COMPONENT_LABELS.get(col[len(prefix):], col[len(prefix):]) for col in values.index]

    fig_size = (16, 10) if PRESENTATION_STYLE else (11, 7)
    fig, ax = plt.subplots(figsize=fig_size)
    colors = ["#2b8a3e" if value >= 0 else "#c92a2a" for value in values]
    ax.barh(labels[::-1], values.iloc[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.legend(
        handles=[
            Patch(facecolor="#2b8a3e", label="Положительный вклад"),
            Patch(facecolor="#c92a2a", label="Отрицательный вклад"),
        ]
    )
    ax.set_title("Средний вклад компонентов награды")
    ax.set_xlabel("Средний вклад компонента")
    save_figure(fig, out_dir, "05_reward_components", formats)

def plot_failure_breakdown(summary_df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    if summary_df.empty:
        return

    prefix = "failure_reason_breakdown_"
    cols = [col for col in summary_df.columns if col.startswith(prefix)]
    if not cols:
        return

    data = summary_df.set_index("run_id")[cols].copy()
    data.columns = [FAILURE_LABELS.get(col[len(prefix):], col[len(prefix):]) for col in cols]
    data = data.fillna(0)
    keep = [col for col in data.columns if data[col].sum() > 0]
    if not keep:
        return
    data = data[keep]

    fig_size = (18, 9) if PRESENTATION_STYLE else (12, 6)
    fig, ax = plt.subplots(figsize=fig_size)
    bottom = np.zeros(len(data))
    x = np.arange(len(data.index))
    for col in data.columns:
        values = data[col].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=col)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels([compact_group_label(item) for item in data.index], rotation=20, ha="right")
    ax.set_title("Распределение причин завершения эпизода")
    ax.set_ylabel("Количество эпизодов")
    save_figure(fig, out_dir, "06_failure_breakdown", formats)

def plot_episode_cloud(episodes_df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    if episodes_df.empty or "timesteps" not in episodes_df.columns:
        return

    colors = {
        "success": "#2b8a3e",
        "crash": "#c92a2a",
        "timeout": "#f08c00",
    }
    fig_size = (18, 12) if PRESENTATION_STYLE else (13, 8)
    fig, axes = plt.subplots(2, 2, figsize=fig_size, sharex=True)
    metrics = [
        ("progress_ratio", "Прогресс к цели"),
        ("dist_to_goal", "Финальное расстояние до цели"),
        ("closest_obstacle", "Минимальная дистанция до препятствия"),
        ("avg_heading_error", "Средняя ошибка направления"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        if metric not in episodes_df.columns:
            ax.axis("off")
            continue
        for outcome, part in episodes_df.groupby("outcome"):
            label = OUTCOME_LABELS.get(str(outcome), str(outcome))
            LEGEND_LABELS[label] = str(outcome)
            ax.scatter(
                part["timesteps"],
                part[metric],
                s=38 if PRESENTATION_STYLE else 14,
                alpha=0.62 if PRESENTATION_STYLE else 0.55,
                label=label,
                color=colors.get(str(outcome), "#495057"),
            )
        ax.set_title(title)
        ax.set_xlabel(STEP_AXIS_LABEL)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        format_training_step_axis(ax)
    fig.suptitle("Диагностика отдельных эпизодов", y=1.02)
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

    fig_size = (22, 11) if PRESENTATION_STYLE else (15, 7)
    fig, axes = plt.subplots(2, 4, figsize=fig_size)
    outcome_colors = {
        "success": "#2b8a3e",
        "crash": "#c92a2a",
        "timeout": "#f08c00",
    }
    for ax, metric in zip(axes.ravel(), metrics):
        part = data[data["metric"] == metric]
        if part.empty:
            ax.axis("off")
            continue
        labels = [OUTCOME_LABELS.get(str(outcome), str(outcome)) for outcome in part["outcome"]]
        colors = [outcome_colors.get(str(outcome), "#495057") for outcome in part["outcome"]]
        ax.bar(labels, part["value"], color=colors)
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.tick_params(axis="x", rotation=25)
    axes.ravel()[0].legend(
        handles=[
            Patch(facecolor="#2b8a3e", label="Успех"),
            Patch(facecolor="#c92a2a", label="Авария"),
            Patch(facecolor="#f08c00", label="Таймаут"),
        ]
    )
    fig.suptitle("Поведение агента по исходам эпизода", y=1.02)
    save_figure(fig, out_dir, "08_behavior_by_outcome", formats)

def plot_curriculum_reports(reports_df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    if reports_df.empty or "obstacle_type" not in reports_df.columns:
        return

    reports_df = reports_df.copy()
    if "success_rate" not in reports_df.columns:
        return
    reports_df["order"] = np.arange(len(reports_df))
    last = reports_df.sort_values("order").groupby("obstacle_type", as_index=False).tail(1)

    fig_size = (18, 7) if PRESENTATION_STYLE else (14, 5)
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    obstacle_labels = [OBSTACLE_LABELS.get(str(item), str(item)) for item in last["obstacle_type"]]
    axes[0].bar(obstacle_labels, last["success_rate"], color="#1c7ed6", label="Успехи")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Итоговая успешность по картам")
    axes[0].set_ylabel("Доля успешных эпизодов")
    axes[0].tick_params(axis="x", rotation=35)

    for metric, label in [("crash_rate", "Аварии"), ("timeout_rate", "Таймауты")]:
        if metric in last.columns:
            axes[1].plot(obstacle_labels, last[metric], marker=None if LINE_ONLY else "o", linewidth=LINE_WIDTH, label=label)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Итоговые неуспешные исходы")
    axes[1].set_ylabel("Доля эпизодов")
    axes[1].tick_params(axis="x", rotation=35)
    save_figure(fig, out_dir, "09_curriculum_obstacle_summary", formats)

def write_overview(
    out_dir: Path,
    eval_df: pd.DataFrame,
    milestones_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    summaries_df: pd.DataFrame,
    tb_df: pd.DataFrame,
    reports_df: pd.DataFrame,
    plot_set: str,
    stitch_runs: bool,
) -> None:
    if LEGEND_LABELS:
        pd.DataFrame(
            [
                {"short_label": short, "full_label": full}
                for short, full in sorted(LEGEND_LABELS.items())
            ]
        ).to_csv(out_dir / "legend_labels.csv", index=False)

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
        "- Plot legends are kept outside the plotting area. If a figure has too many series, the plot is left clean and labels are written to legend_labels.csv.",
        f"- Plot set: {plot_set}. Full mode writes the complete diagnostics set; compact mode is available only for temporary quick previews.",
        f"- Retry X-axis stitching: {'enabled' if stitch_runs else 'disabled'}.",
        "- 01_eval_history: success/crash/timeout and reward during periodic evaluation.",
        "- 02_training_milestones: rolling success/crash/timeout, reward, progress, and final distance during training.",
        "- 10_rl_reward_success_curves: rollout/eval reward and success/failure dynamics.",
        "- 11_rl_loss_kl_entropy_curves: total loss, policy loss, value loss, entropy loss, KL, and explained variance.",
        "- 12_rl_policy_update_runtime_curves: policy std, clipping, learning rate, episode length, and FPS.",
    ])
    if plot_set == "full":
        lines.extend([
            "- 03_policy_losses and 04_optimization_diagnostics: extra PPO optimization signals from TensorBoard.",
            "- 05_reward_components: reward shaping contribution sanity check.",
            "- 06_failure_breakdown: final failure reason breakdown.",
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
    global STEP_AXIS_LABEL, LINE_ONLY

    parser = argparse.ArgumentParser(description="Generate plots from training logs")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--eval-dir", default="logs/eval")
    parser.add_argument("--diagnostics-dir", default="logs/training_diagnostics")
    parser.add_argument("--curriculum-reports-dir", default="reports/curriculum")
    parser.add_argument("--reports-dir", default="reports/training_plots")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--run-filter", default="")
    parser.add_argument("--exclude-run-filter", default="")
    parser.add_argument("--formats", default="png,svg")
    parser.add_argument("--plot-set", choices=["compact", "full"], default="full",
                        help="full: complete diagnostics set; compact: only core learning curves for quick previews")
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--max-episode-rows", type=int, default=200_000)
    parser.add_argument("--stitch-runs", action="store_true",
                        help="Shift retry/continue runs so their timesteps are shown consecutively instead of overlapping")
    parser.add_argument("--stitch-gap-steps", type=float, default=0.0,
                        help="Visual gap between stitched runs on the X axis")
    parser.add_argument("--merge-runs-label", default="",
                        help="After stitching, draw all selected runs as one continuous series with this label")
    parser.add_argument("--line-only", action="store_true",
                        help="Disable point markers on line charts")
    parser.add_argument("--presentation", action="store_true",
                        help="Use larger fonts, thicker lines, and an enlarged legend outside the plot area")
    args = parser.parse_args()

    root = repo_root()
    setup_style(args.presentation)
    LINE_ONLY = args.line_only

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (root / args.reports_dir / tag).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = parse_csv(args.formats) or ["png"]

    eval_df = load_eval_history(root / args.eval_dir, args.run_filter, args.exclude_run_filter)
    milestones_df, episodes_df, summaries_df = load_diagnostics(
        root / args.diagnostics_dir,
        args.run_filter,
        args.exclude_run_filter,
        max_episode_rows=args.max_episode_rows,
    )
    tb_df = load_tensorboard_scalars(root / args.logs_dir, args.run_filter, args.exclude_run_filter)
    reports_df = load_curriculum_reports(root / args.curriculum_reports_dir, args.run_filter, args.exclude_run_filter)

    if args.stitch_runs:
        STEP_AXIS_LABEL = "Шаги обучения (ретраи склеены)"
        eval_df = stitch_run_steps(eval_df, "timesteps", "run_id", args.stitch_gap_steps)
        milestones_df = stitch_run_steps(milestones_df, "timesteps", "run_id", args.stitch_gap_steps)
        episodes_df = stitch_run_steps(episodes_df, "timesteps", "run_id", args.stitch_gap_steps)
        tb_df = add_tensorboard_run_ids(tb_df)
        tb_df = stitch_run_steps(tb_df, "step", "run_id", args.stitch_gap_steps)

    if args.merge_runs_label:
        eval_df = merge_runs_for_plots(eval_df, args.merge_runs_label)
        milestones_df = merge_runs_for_plots(milestones_df, args.merge_runs_label)
        episodes_df = merge_runs_for_plots(episodes_df, args.merge_runs_label)
        tb_df = merge_runs_for_plots(tb_df, args.merge_runs_label)

    save_csvs(out_dir, eval_df, milestones_df, episodes_df, summaries_df, tb_df, reports_df)
    plot_eval_history(eval_df, out_dir, formats, args.smooth_window)
    plot_milestones(milestones_df, out_dir, formats, args.smooth_window)
    plot_tensorboard(tb_df, out_dir, formats, args.smooth_window, args.plot_set)
    if args.plot_set == "full":
        plot_reward_components(summaries_df, out_dir, formats)
        plot_failure_breakdown(summaries_df, out_dir, formats)
        plot_episode_cloud(episodes_df, out_dir, formats)
        plot_behavior_by_outcome(summaries_df, out_dir, formats)
    plot_curriculum_reports(reports_df, out_dir, formats)
    write_overview(out_dir, eval_df, milestones_df, episodes_df, summaries_df, tb_df, reports_df, args.plot_set, args.stitch_runs)

    print(f"[PLOTS] Saved report to: {out_dir}")
    for figure in FIGURES:
        print(f"[PLOTS]   {figure}")

if __name__ == "__main__":
    main()
