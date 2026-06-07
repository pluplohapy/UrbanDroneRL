
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pybullet as p
import pybullet_data

from config import apply_pretrain_obstacle_overrides, load_config
from scenarios.stage_pretrain import StagePretrainScenario

DEFAULT_MAPS = (
    "cylinders",
    "beams",
    "swinging_sticks",
    "city_dynamic",
    "construction_site_dynamic",
)

MAP_LABELS = {
    "cylinders": "Цилиндры (Cylinders)",
    "beams": "Подвижные балки (Moving Beams)",
    "swinging_sticks": "Качающиеся палки (Swinging Sticks)",
    "city_dynamic": "Динамический город (Dynamic City)",
    "construction_site_dynamic": "Стройплощадка (Construction Site)",
    "spheres": "Сферы (Spheres)",
    "crossing_spheres": "Пересекающиеся сферы (Crossing Spheres)",
    "walls": "Стены (Walls)",
    "gates": "Ворота (Gates)",
    "slalom": "Слалом (Slalom)",
    "city_blocks": "Городские блоки (City Blocks)",
}

MAP_SEEDS = {
    "cylinders": 2778,
    "beams": 3778,
    "swinging_sticks": 4778,
    "city_dynamic": 5778,
    "construction_site_dynamic": 6778,
    "spheres": 3378,
    "crossing_spheres": 3478,
    "walls": 3578,
    "gates": 3678,
    "slalom": 3878,
    "city_blocks": 3978,
}

def _create_visual_box(client_id, position, half_extents, color):
    visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
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

def _create_visual_cylinder(client_id, position, radius, length, color, orientation=None):
    visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=length,
        rgbaColor=color,
        physicsClientId=client_id,
    )
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        baseOrientation=orientation or [0, 0, 0, 1],
        physicsClientId=client_id,
    )

def _create_visual_sphere(client_id, position, radius, color):
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

def _add_presentation_floor_and_frame(client_id, cfg):
    arena_x = float(cfg.ARENA_SIZE_X)
    arena_y = float(cfg.ARENA_SIZE_Y)
    arena_h = float(cfg.ARENA_HEIGHT)
    rail = max(0.035, min(arena_x, arena_y) * 0.008)
    floor_z = -0.018
    grid_z = 0.004

    _create_visual_box(
        client_id,
        [0, 0, floor_z],
        [arena_x / 2.0, arena_y / 2.0, 0.012],
        [0.90, 0.90, 0.86, 1.0],
    )

    grid_color = [0.72, 0.74, 0.70, 1.0]
    grid_half_thickness = max(0.004, min(arena_x, arena_y) * 0.0015)
    for x in np.arange(-arena_x / 2.0 + 1.0, arena_x / 2.0, 1.0):
        _create_visual_box(
            client_id,
            [float(x), 0, grid_z],
            [grid_half_thickness, arena_y / 2.0, 0.003],
            grid_color,
        )
    for y in np.arange(-arena_y / 2.0 + 1.0, arena_y / 2.0, 1.0):
        _create_visual_box(
            client_id,
            [0, float(y), grid_z],
            [arena_x / 2.0, grid_half_thickness, 0.003],
            grid_color,
        )

    _create_visual_box(
        client_id,
        [0, -arena_y / 2.0, rail],
        [arena_x / 2.0, rail, rail],
        [0.15, 0.16, 0.18, 1.0],
    )
    _create_visual_box(
        client_id,
        [0, arena_y / 2.0, rail],
        [arena_x / 2.0, rail, rail],
        [0.15, 0.16, 0.18, 1.0],
    )
    _create_visual_box(
        client_id,
        [-arena_x / 2.0, 0, rail],
        [rail, arena_y / 2.0, rail],
        [0.15, 0.16, 0.18, 1.0],
    )
    _create_visual_box(
        client_id,
        [arena_x / 2.0, 0, rail],
        [rail, arena_y / 2.0, rail],
        [0.15, 0.16, 0.18, 1.0],
    )

    post_half = [rail, rail, arena_h / 2.0]
    for x in (-arena_x / 2.0, arena_x / 2.0):
        for y in (-arena_y / 2.0, arena_y / 2.0):
            _create_visual_box(
                client_id,
                [x, y, arena_h / 2.0],
                post_half,
                [0.18, 0.19, 0.22, 1.0],
            )

def _add_start_goal_markers(client_id, start_pos, goal_pos, cfg):
    marker_radius = max(0.18, min(0.35, min(float(cfg.ARENA_SIZE_X), float(cfg.ARENA_SIZE_Y)) * 0.045))

    start_color = [0.10, 0.76, 0.36, 1.0]
    goal_color = [0.10, 0.35, 0.95, 1.0]

    _create_visual_sphere(client_id, start_pos, marker_radius, start_color)
    _create_visual_sphere(client_id, goal_pos, marker_radius, goal_color)

def _style_existing_bodies(client_id):
    for body_index in range(p.getNumBodies(physicsClientId=client_id)):
        body_id = p.getBodyUniqueId(body_index, physicsClientId=client_id)
        try:
            p.changeVisualShape(
                body_id,
                -1,
                specularColor=[0.12, 0.12, 0.12],
                physicsClientId=client_id,
            )
        except Exception:
            pass

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
            farVal=120.0,
        )
        return view, projection

    distance = max(arena_y * 1.12, arena_x * 1.85, arena_h * 3.4)
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
        farVal=120.0,
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

def render_map(map_name, output_dir, width, height, seed, dynamic_seconds, views, show_floor_frame=True):
    cfg = apply_pretrain_obstacle_overrides(load_config("pretrain"), map_name)
    client_id = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    p.resetSimulation(physicsClientId=client_id)
    p.setGravity(0, 0, -9.81, physicsClientId=client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client_id)

    scenario = StagePretrainScenario(obstacle_type=map_name, seed=seed)
    start_pos, goal_pos = scenario.reset(client_id)

    steps = max(1, int(math.ceil(dynamic_seconds / 0.02)))
    for _ in range(steps):
        scenario.update_dynamic_obstacles(0.02)
        p.stepSimulation(physicsClientId=client_id)

    _style_existing_bodies(client_id)
    if show_floor_frame:
        _add_presentation_floor_and_frame(client_id, cfg)
    _add_start_goal_markers(client_id, np.asarray(start_pos), np.asarray(goal_pos), cfg)

    result_paths = []
    for view_name in views:
        output_path = output_dir / f"{map_name}_{view_name}.png"
        _render_camera(client_id, cfg, output_path, width, height, view_name)
        result_paths.append(output_path)

    scenario.cleanup()
    p.disconnect(physicsClientId=client_id)
    return result_paths

def _load_font(size):
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/Library/Fonts/Arial.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()

def _make_contact_sheet(output_dir, map_names, view_name):
    images = []
    for map_name in map_names:
        path = output_dir / f"{map_name}_{view_name}.png"
        if path.exists():
            images.append((map_name, Image.open(path).convert("RGB")))
    if not images:
        return None

    thumb_w, thumb_h = 1180, 664
    label_h = 88
    margin = 42
    columns = 2 if len(images) <= 6 else 3
    rows = math.ceil(len(images) / columns)
    sheet_w = columns * thumb_w + (columns + 1) * margin
    sheet_h = rows * (thumb_h + label_h) + (rows + 1) * margin
    sheet = Image.new("RGB", (sheet_w, sheet_h), (248, 248, 246))
    draw = ImageDraw.Draw(sheet)
    font = _load_font(38)

    for idx, (map_name, image) in enumerate(images):
        row = idx // columns
        col = idx % columns
        x = margin + col * (thumb_w + margin)
        y = margin + row * (thumb_h + label_h + margin)
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        frame = Image.new("RGB", (thumb_w, thumb_h), (232, 232, 228))
        paste_x = (thumb_w - image.width) // 2
        paste_y = (thumb_h - image.height) // 2
        frame.paste(image, (paste_x, paste_y))
        sheet.paste(frame, (x, y))

        label = MAP_LABELS.get(map_name, map_name.replace("_", " ").title())
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_x = x + (thumb_w - (text_bbox[2] - text_bbox[0])) // 2
        draw.text((text_x, y + thumb_h + 12), label, fill=(32, 34, 36), font=font)

    output_path = output_dir / f"00_contact_sheet_{view_name}.png"
    sheet.save(output_path, quality=95)
    return output_path

def parse_args():
    parser = argparse.ArgumentParser(description="Render presentation map screenshots")
    parser.add_argument(
        "--maps",
        nargs="+",
        default=list(DEFAULT_MAPS),
        help="Pretrain obstacle types to render",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/presentation_map_images"),
        help="Directory for PNG outputs",
    )
    parser.add_argument("--width", type=int, default=1920, help="Image width")
    parser.add_argument("--height", type=int, default=1080, help="Image height")
    parser.add_argument("--seed", type=int, default=None, help="Override seed for every map")
    parser.add_argument(
        "--dynamic-seconds",
        type=float,
        default=2.2,
        help="Advance dynamic maps before rendering",
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
    parser.add_argument(
        "--no-floor-frame",
        action="store_true",
        help="Render only obstacles and start/goal markers on a plain background",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_paths = []
    for map_name in args.maps:
        seed = args.seed if args.seed is not None else MAP_SEEDS.get(map_name, 12345)
        paths = render_map(
            map_name=map_name,
            output_dir=output_dir,
            width=args.width,
            height=args.height,
            seed=seed,
            dynamic_seconds=args.dynamic_seconds,
            views=args.views,
            show_floor_frame=not args.no_floor_frame,
        )
        all_paths.extend(paths)
        print(f"Rendered {map_name}: " + ", ".join(str(path) for path in paths))

    if not args.no_contact_sheet:
        for view_name in args.views:
            sheet_path = _make_contact_sheet(output_dir, args.maps, view_name)
            if sheet_path:
                all_paths.append(sheet_path)
                print(f"Rendered contact sheet: {sheet_path}")

    manifest = output_dir / "manifest.txt"
    with manifest.open("w", encoding="utf-8") as handle:
        handle.write("Presentation map images\n")
        handle.write("=======================\n\n")
        for path in all_paths:
            handle.write(f"{path.name}\n")
    print(f"Manifest: {manifest}")

if __name__ == "__main__":
    main()
