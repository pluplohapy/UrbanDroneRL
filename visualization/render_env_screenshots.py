
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pybullet as p

from config import apply_pretrain_obstacle_overrides, load_config
from config.runtime_sync import sync_runtime_config
from envs.nav_aviary import NavAviary
from scenarios.stage_pretrain import StagePretrainScenario
from visualization.render_map_images import MAP_LABELS, MAP_SEEDS, _make_contact_sheet


DEFAULT_MAPS = (
    "cylinders",
    "beams",
    "swinging_sticks",
    "city_dynamic",
)


def _create_marker_sphere(client_id, position, radius, color):
    visual_shape = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color,
        physicsClientId=client_id,
    )
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        physicsClientId=client_id,
    )


def _add_visualization_markers(client_id, start_pos, goal_pos, cfg):
    marker_radius = max(0.16, min(0.32, min(float(cfg.ARENA_SIZE_X), float(cfg.ARENA_SIZE_Y)) * 0.04))
    _create_marker_sphere(client_id, start_pos, marker_radius, [0.1, 0.65, 1.0, 1.0])
    _create_marker_sphere(client_id, goal_pos, marker_radius, [0.0, 0.95, 0.25, 1.0])


def _camera_matrices(cfg, view_name, width, height):
    arena_x = float(cfg.ARENA_SIZE_X)
    arena_y = float(cfg.ARENA_SIZE_Y)
    arena_h = float(cfg.ARENA_HEIGHT)
    aspect = width / float(height)

    if view_name == "top":
        distance = max(arena_x * 1.7, arena_y * 1.35, arena_h * 3.8)
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0.0],
            distance=distance,
            yaw=0,
            pitch=-89.5,
            roll=0,
            upAxisIndex=2,
        )
        projection = p.computeProjectionMatrixFOV(
            fov=48,
            aspect=aspect,
            nearVal=0.1,
            farVal=140.0,
        )
        return view, projection

    distance = max(arena_y * 1.08, arena_x * 1.85, arena_h * 3.35)
    target_z = min(arena_h * 0.45, 2.6)
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, target_z],
        distance=distance,
        yaw=-38,
        pitch=-48,
        roll=0,
        upAxisIndex=2,
    )
    projection = p.computeProjectionMatrixFOV(
        fov=42,
        aspect=aspect,
        nearVal=0.1,
        farVal=140.0,
    )
    return view, projection


def _render_camera(client_id, cfg, output_path, width, height, view_name):
    view_matrix, projection_matrix = _camera_matrices(cfg, view_name, width, height)
    _, _, rgba, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        shadow=1,
        lightDirection=[-0.35, -0.6, -0.85],
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client_id,
    )
    frame = np.reshape(np.asarray(rgba, dtype=np.uint8), (height, width, 4))
    Image.fromarray(frame[:, :, :3], mode="RGB").save(output_path, quality=95)


def render_environment_map(map_name, output_dir, width, height, seed, dynamic_seconds, views):
    cfg = apply_pretrain_obstacle_overrides(load_config("pretrain"), map_name)
    cfg.SAFETY_SHIELD_ENABLED = False
    sync_runtime_config(cfg)

    scenario = StagePretrainScenario(obstacle_type=map_name, seed=seed)
    env = NavAviary(
        scenario=scenario,
        gui=False,
        fixed_map=False,
        show_trajectory=False,
    )

    try:
        env.reset(seed=seed)
        client_id = env.CLIENT

        steps = max(1, int(math.ceil(dynamic_seconds / 0.02)))
        for _ in range(steps):
            env.scenario.update_dynamic_obstacles(0.02)

        _add_visualization_markers(client_id, env.start_pos, env.goal_pos, cfg)

        result_paths = []
        for view_name in views:
            output_path = output_dir / f"{map_name}_{view_name}.png"
            _render_camera(client_id, cfg, output_path, width, height, view_name)
            result_paths.append(output_path)
        return result_paths
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Render NavAviary environment screenshots")
    parser.add_argument(
        "--maps",
        nargs="+",
        default=list(DEFAULT_MAPS),
        help="Pretrain obstacle types to render",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/presentation_env_screenshots_ru_hq_no_construction"),
        help="Directory for PNG outputs",
    )
    parser.add_argument("--width", type=int, default=3840, help="Image width")
    parser.add_argument("--height", type=int, default=2160, help="Image height")
    parser.add_argument("--seed", type=int, default=None, help="Override seed for every map")
    parser.add_argument(
        "--dynamic-seconds",
        type=float,
        default=2.2,
        help="Advance dynamic obstacles before rendering",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        choices=("overview", "top"),
        default=["overview", "top"],
        help="Camera views to render",
    )
    parser.add_argument(
        "--no-contact-sheet",
        action="store_true",
        help="Skip combined preview sheets",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_paths = []
    for map_name in args.maps:
        seed = args.seed if args.seed is not None else MAP_SEEDS.get(map_name, 12345)
        paths = render_environment_map(
            map_name=map_name,
            output_dir=output_dir,
            width=args.width,
            height=args.height,
            seed=seed,
            dynamic_seconds=args.dynamic_seconds,
            views=args.views,
        )
        all_paths.extend(paths)
        print(f"Rendered {MAP_LABELS.get(map_name, map_name)}: " + ", ".join(str(path) for path in paths))

    if not args.no_contact_sheet:
        for view_name in args.views:
            sheet_path = _make_contact_sheet(output_dir, args.maps, view_name)
            if sheet_path:
                all_paths.append(sheet_path)
                print(f"Rendered contact sheet: {sheet_path}")

    manifest = output_dir / "manifest.txt"
    with manifest.open("w", encoding="utf-8") as handle:
        handle.write("NavAviary environment screenshots\n")
        handle.write("==================================\n\n")
        for path in all_paths:
            handle.write(f"{path.name}\n")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()