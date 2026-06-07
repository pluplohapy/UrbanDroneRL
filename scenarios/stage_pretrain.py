
from collections import deque
from types import SimpleNamespace
import numpy as np
from typing import List, Tuple
from scenarios.base_scenario import BaseScenario
from scenarios.pretrain_obstacles import (
    CylinderObstacle, SphereObstacle, WallObstacle,
    BeamObstacle, BoxObstacle, MovingBoxObstacle, SwingingStickObstacle
)
from config import load_config


class StagePretrainScenario(BaseScenario):

    def __init__(self, obstacle_type='random', seed=None):
        super().__init__(seed)
        self.obstacle_type = obstacle_type
        self.current_obstacle_type = None
        self.config = self._config_for_obstacle_type(load_config('pretrain'), obstacle_type)
        self.client_id = None

    @staticmethod
    def _config_for_obstacle_type(base_config, obstacle_type: str):
        params = getattr(base_config, "OBSTACLE_TYPES", {}).get(obstacle_type, {})
        overrides = dict(params.get("config_overrides", {}))
        if not overrides:
            return base_config

        values = {
            name: getattr(base_config, name)
            for name in dir(base_config)
            if not name.startswith("_")
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _resolve_obstacle_type(self) -> str:
        if self.obstacle_type == 'random':
            choices = [
                name for name, params in self.config.OBSTACLE_TYPES.items()
                if not bool(params.get("requires_explicit_obstacle_type", False))
            ]
            return str(self.rng.choice(choices))

        if self.obstacle_type == 'dynamic_mix':
            dynamic_types = [
                name for name, params in self.config.OBSTACLE_TYPES.items()
                if bool(params.get('dynamic', False))
                and not bool(params.get("requires_explicit_obstacle_type", False))
            ]
            if not dynamic_types:
                raise ValueError("No dynamic obstacle types configured for dynamic_mix mode")

            weights_config = getattr(self.config, "DYNAMIC_MIX_WEIGHTS", None)
            if weights_config:
                weights = np.array([float(weights_config.get(name, 0.0)) for name in dynamic_types], dtype=float)
                if np.sum(weights) > 0.0:
                    weights = weights / np.sum(weights)
                    return str(self.rng.choice(dynamic_types, p=weights))

            return str(self.rng.choice(dynamic_types))

        return self.obstacle_type

    def generate(self, client_id):
        return self.reset(client_id)

    def reset(self, client_id):
        self.client_id = client_id

        max_attempts = int(getattr(self.config, "SPAWN_VALIDATION_ATTEMPTS", 1))
        explicit_params = self.config.OBSTACLE_TYPES.get(self.obstacle_type, {})
        max_attempts = int(explicit_params.get('spawn_validation_attempts', max_attempts))

        for attempt in range(max_attempts):
            self._cleanup_generated_obstacles()


            start_pos, goal_pos = self._generate_start_goal_zones()


            chosen_type = self._resolve_obstacle_type()
            self.current_obstacle_type = chosen_type


            self._generate_obstacles(chosen_type, start_pos, goal_pos, client_id)

            chosen_params = self.config.OBSTACLE_TYPES.get(chosen_type, {})
            validate_spawn = (
                'spawn_validation_attempts' in chosen_params
                or bool(chosen_params.get('validate_spawn_clearance', False))
            )
            if not validate_spawn or self.points_clear_of_obstacles([start_pos, goal_pos]):
                return start_pos, goal_pos

        raise RuntimeError(f"Could not generate {self.obstacle_type} with clear start/goal points")

    def _cleanup_generated_obstacles(self):
        try:
            import pybullet as p
            valid_body_ids = {
                p.getBodyUniqueId(i, physicsClientId=self.client_id)
                for i in range(p.getNumBodies(physicsClientId=self.client_id))
            }
        except Exception:
            valid_body_ids = None

        for obstacle in self.obstacles:
            try:
                body_id = getattr(obstacle, "body_id", None)
                if valid_body_ids is not None and body_id not in valid_body_ids:
                    obstacle.body_id = None
                    continue
                if hasattr(obstacle, "cleanup"):
                    obstacle.cleanup()
            except Exception:
                pass
        self.obstacles = []

    def _generate_start_goal_zones(self) -> Tuple[np.ndarray, np.ndarray]:
        bidirectional = bool(getattr(self.config, "PRETRAIN_BIDIRECTIONAL_GOALS", True))
        if not bidirectional:
            start_y_range = self.config.START_ZONE_Y
            goal_y_range = self.config.GOAL_ZONE_Y
        elif self.rng.random() < 0.5:
            start_y_range = self.config.START_ZONE_Y
            goal_y_range = self.config.GOAL_ZONE_Y
        else:
            start_y_range = self.config.GOAL_ZONE_Y
            goal_y_range = self.config.START_ZONE_Y

        start_x = self.rng.uniform(*self.config.START_ZONE_X)
        goal_x = self._sample_goal_x()
        start_y = self.rng.uniform(*start_y_range)
        goal_y = self._sample_goal_y(goal_y_range)
        start_z = self.rng.uniform(*self.config.START_ZONE_Z)
        goal_z = self.rng.uniform(*self.config.GOAL_ZONE_Z)

        start_pos = np.array([start_x, start_y, start_z])
        goal_pos = np.array([goal_x, goal_y, goal_z])

        return start_pos, goal_pos

    def _sample_goal_x(self) -> float:
        bias = float(getattr(self.config, "GOAL_CORNER_BIAS", 0.0))
        bands = getattr(self.config, "GOAL_CORNER_X_BANDS", None)
        if bands and self.rng.random() < bias:
            clipped_bands = [
                clipped for band in bands
                if (clipped := self._clip_goal_range_to_arena(band, axis="x"))[0] <= clipped[1]
            ]
            if clipped_bands:
                band = clipped_bands[self.rng.randint(0, len(clipped_bands))]
                return float(self.rng.uniform(*band))

        return float(self.rng.uniform(*self._clip_goal_range_to_arena(self.config.GOAL_ZONE_X, axis="x")))

    def _goal_wall_clearance(self) -> float:
        return max(0.0, float(getattr(self.config, "GOAL_WALL_CLEARANCE", 0.0)))

    def _clip_goal_range_to_arena(self, value_range, axis: str) -> Tuple[float, float]:
        low, high = float(value_range[0]), float(value_range[1])
        if low > high:
            low, high = high, low

        half_extent = (
            float(self.config.ARENA_SIZE_X) / 2.0
            if axis == "x"
            else float(self.config.ARENA_SIZE_Y) / 2.0
        )
        clearance = min(self._goal_wall_clearance(), max(half_extent - 1e-6, 0.0))
        if clearance <= 0.0:
            return low, high

        safe_low = -half_extent + clearance
        safe_high = half_extent - clearance
        clipped_low = max(low, safe_low)
        clipped_high = min(high, safe_high)
        if clipped_low <= clipped_high:
            return clipped_low, clipped_high

        center = float(np.clip((low + high) / 2.0, safe_low, safe_high))
        return center, center

    def _sample_goal_y(self, goal_y_range) -> float:
        bias = float(getattr(self.config, "GOAL_END_Y_BIAS", 0.0))
        goal_y_range = self._clip_goal_range_to_arena(goal_y_range, axis="y")
        if self.rng.random() >= bias:
            return float(self.rng.uniform(*goal_y_range))

        low, high = float(goal_y_range[0]), float(goal_y_range[1])
        span = high - low
        if span <= 0.0:
            return low

        edge_fraction = float(np.clip(getattr(self.config, "GOAL_END_Y_EDGE_FRACTION", 1.0), 0.0, 1.0))
        edge_width = max(span * edge_fraction, 1e-6)
        if (low + high) >= 0.0:
            return float(self.rng.uniform(high - edge_width, high))
        return float(self.rng.uniform(low, low + edge_width))

    def _generate_obstacles(self, obstacle_type: str, start_pos: np.ndarray,
                           goal_pos: np.ndarray, client_id: int):
        if obstacle_type not in self.config.OBSTACLE_TYPES:
            raise ValueError(f"Unknown obstacle type for pretrain scenario: {obstacle_type}")

        self.current_obstacle_type = obstacle_type
        params = self.config.OBSTACLE_TYPES[obstacle_type]
        n_obstacles = self.rng.randint(params['count'][0], params['count'][1] + 1)

        if obstacle_type == 'empty':

            return
        elif obstacle_type == 'cylinders':
            self._generate_cylinders(n_obstacles, params, start_pos, goal_pos, client_id)
        elif obstacle_type == 'spheres':
            self._generate_spheres(n_obstacles, params, start_pos, goal_pos, client_id)
        elif obstacle_type == 'crossing_spheres':
            self._generate_crossing_spheres(n_obstacles, params, start_pos, goal_pos, client_id)
        elif obstacle_type == 'walls':
            self._generate_walls(n_obstacles, params, start_pos, goal_pos, client_id)
        elif obstacle_type == 'beams':
            self._generate_beams(n_obstacles, params, start_pos, goal_pos, client_id)
        elif obstacle_type == 'boxes':
            self._generate_boxes(n_obstacles, params, start_pos, goal_pos, client_id)
        elif obstacle_type == 'gates':
            self._generate_gates(n_obstacles, params, start_pos, goal_pos, client_id)
        elif obstacle_type == 'slalom':
            self._generate_slalom(n_obstacles, params, start_pos, goal_pos, client_id)
        elif obstacle_type == 'city_blocks':
            self._generate_city_blocks(n_obstacles, params, start_pos, goal_pos, client_id)
        elif obstacle_type == 'city_dynamic':
            self._generate_city_dynamic(n_obstacles, params, start_pos, goal_pos, client_id)
        elif obstacle_type == 'construction_site_dynamic':
            self._generate_construction_site_dynamic(n_obstacles, params, start_pos, goal_pos, client_id)
        elif obstacle_type == 'swinging_sticks':
            self._generate_swinging_sticks(n_obstacles, params, start_pos, goal_pos, client_id)

    def _generate_cylinders(self, n_obstacles: int, params: dict,
                           start_pos: np.ndarray, goal_pos: np.ndarray, client_id: int):
        wall_margin = float(getattr(self.config, "CYLINDER_WALL_MARGIN", 0.35))
        pair_clearance = float(getattr(self.config, "CYLINDER_PAIR_CLEARANCE", 0.45))
        path_clearance = float(getattr(self.config, "CYLINDER_PATH_CLEARANCE", 0.45))
        grid_resolution = float(getattr(self.config, "CYLINDER_PATH_GRID_RESOLUTION", 0.20))


        for target_count in range(n_obstacles, 0, -1):
            for _ in range(40):
                layout = self._sample_cylinder_layout(
                    target_count=target_count,
                    params=params,
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    wall_margin=wall_margin,
                    pair_clearance=pair_clearance,
                    attempts_per_obstacle=120,
                )
                if not layout:
                    continue

                if not self._has_feasible_path_2d(
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    cylinders=layout,
                    clearance=path_clearance,
                    grid_resolution=grid_resolution,
                ):
                    continue

                for pos, radius, height in layout:
                    obstacle = CylinderObstacle(pos, radius, height, client_id)
                    self.obstacles.append(obstacle)
                if target_count < n_obstacles:
                    print(
                        f"[SCENARIO] cylinders reduced from {n_obstacles} "
                        f"to {target_count} to keep map feasible"
                    )
                return

        print("[SCENARIO][WARN] Could not sample feasible cylinder map, using empty layout")

    def _sample_cylinder_layout(
        self,
        target_count: int,
        params: dict,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        wall_margin: float,
        pair_clearance: float,
        attempts_per_obstacle: int = 120,
    ) -> List[Tuple[np.ndarray, float, float]]:
        half_x = self.config.ARENA_SIZE_X / 2.0
        half_y = self.config.ARENA_SIZE_Y / 2.0
        layout: List[Tuple[np.ndarray, float, float]] = []

        for _ in range(target_count):
            placed = False
            for _ in range(attempts_per_obstacle):
                radius = self.rng.uniform(*params["radius"])
                height = self.rng.uniform(*params["height"])

                x_min = -half_x + radius + wall_margin
                x_max = half_x - radius - wall_margin
                y_min = -half_y + radius + wall_margin
                y_max = half_y - radius - wall_margin
                if x_min >= x_max or y_min >= y_max:
                    return []

                x = self.rng.uniform(x_min, x_max)
                y = self.rng.uniform(y_min, y_max)
                pos = np.array([x, y, 0.0])

                placed_xy = [(p, r) for p, r, _ in layout]
                if self._is_valid_position(
                    pos=pos,
                    radius=radius,
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    placed=placed_xy,
                    min_pair_clearance=pair_clearance,
                    wall_margin=wall_margin,
                ):
                    layout.append((pos, radius, height))
                    placed = True
                    break

            if not placed:
                return []

        return layout

    def _has_feasible_path_2d(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        cylinders: List[Tuple[np.ndarray, float, float]],
        clearance: float,
        grid_resolution: float,
    ) -> bool:
        half_x = self.config.ARENA_SIZE_X / 2.0
        half_y = self.config.ARENA_SIZE_Y / 2.0
        x_min, x_max = -half_x, half_x
        y_min, y_max = -half_y, half_y

        x_vals = np.arange(x_min, x_max + grid_resolution * 0.5, grid_resolution)
        y_vals = np.arange(y_min, y_max + grid_resolution * 0.5, grid_resolution)
        xx, yy = np.meshgrid(x_vals, y_vals)


        blocked = (np.abs(xx) >= (half_x - clearance)) | (np.abs(yy) >= (half_y - clearance))

        for pos, radius, _ in cylinders:
            inflated = radius + clearance
            blocked |= (xx - pos[0]) ** 2 + (yy - pos[1]) ** 2 <= inflated ** 2

        ny, nx = blocked.shape

        def _to_idx(p: np.ndarray) -> Tuple[int, int]:
            ix = int(np.clip(np.rint((p[0] - x_min) / grid_resolution), 0, nx - 1))
            iy = int(np.clip(np.rint((p[1] - y_min) / grid_resolution), 0, ny - 1))
            return iy, ix

        start_idx = _to_idx(start_pos)
        goal_idx = _to_idx(goal_pos)
        if blocked[start_idx] or blocked[goal_idx]:
            return False

        queue = deque([start_idx])
        visited = np.zeros_like(blocked, dtype=bool)
        visited[start_idx] = True
        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]

        while queue:
            iy, ix = queue.popleft()
            if (iy, ix) == goal_idx:
                return True

            for dy, dx in neighbors:
                ny_i = iy + dy
                nx_i = ix + dx
                if ny_i < 0 or ny_i >= ny or nx_i < 0 or nx_i >= nx:
                    continue
                if visited[ny_i, nx_i] or blocked[ny_i, nx_i]:
                    continue
                visited[ny_i, nx_i] = True
                queue.append((ny_i, nx_i))

        return False

    def _generate_spheres(self, n_obstacles: int, params: dict,
                         start_pos: np.ndarray, goal_pos: np.ndarray, client_id: int):
        half_x = self.config.ARENA_SIZE_X / 2.0
        half_y = self.config.ARENA_SIZE_Y / 2.0
        start_goal_clearance = float(getattr(self.config, "DYNAMIC_START_GOAL_CLEARANCE", 1.1))

        for _ in range(n_obstacles):
            radius = self.rng.uniform(*params['radius'])
            speed = self.rng.uniform(*params['speed'])
            amplitude = self.rng.uniform(*self.config.SPHERE_MOVEMENT_AMPLITUDE)
            frequency = self.rng.uniform(*self.config.SPHERE_MOVEMENT_FREQUENCY)
            phase = self.rng.uniform(0.0, 2.0 * np.pi)
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle), 0.0])
            sweep_radius = radius + amplitude


            for _ in range(80):
                x_min = -half_x + sweep_radius + 0.15
                x_max = half_x - sweep_radius - 0.15
                y_min = -half_y + start_goal_clearance + sweep_radius
                y_max = half_y - start_goal_clearance - sweep_radius
                if x_min > x_max:
                    x_min = x_max = 0.0
                if y_min > y_max:
                    continue

                x = self.rng.uniform(x_min, x_max)
                y = self.rng.uniform(y_min, y_max)
                z = self.rng.uniform(0.9, self.config.ARENA_HEIGHT - 0.8)
                pos = np.array([x, y, z])

                if self._is_valid_position(pos, sweep_radius, start_pos, goal_pos):
                    obstacle = SphereObstacle(
                        pos,
                        radius,
                        speed,
                        amplitude,
                        frequency,
                        client_id,
                        direction=direction,
                        phase=phase,
                    )
                    self.obstacles.append(obstacle)
                    break

    def _generate_crossing_spheres(self, n_obstacles: int, params: dict,
                                   start_pos: np.ndarray, goal_pos: np.ndarray, client_id: int):
        half_y = self.config.ARENA_SIZE_Y / 2.0
        start_goal_clearance = float(getattr(self.config, "DYNAMIC_START_GOAL_CLEARANCE", 1.1))
        y_min = -half_y + start_goal_clearance + 0.6
        y_max = half_y - start_goal_clearance - 0.6
        if y_min >= y_max:
            return

        y_bands = np.linspace(y_min, y_max, n_obstacles + 2)[1:-1]
        self.rng.shuffle(y_bands)
        placed = []

        for band_y in y_bands:
            radius = self.rng.uniform(*params['radius'])
            speed = self.rng.uniform(*params['speed'])
            amplitude_range = params.get('amplitude', self.config.SPHERE_MOVEMENT_AMPLITUDE)
            frequency_range = params.get('frequency', self.config.SPHERE_MOVEMENT_FREQUENCY)
            amplitude = self.rng.uniform(*amplitude_range)
            frequency = self.rng.uniform(*frequency_range)
            phase = self.rng.uniform(0.0, 2.0 * np.pi)
            direction = np.array([1.0, 0.0, 0.0])

            for _ in range(60):
                y = float(band_y + self.rng.uniform(-0.25, 0.25))
                z = self.rng.uniform(0.95, self.config.ARENA_HEIGHT - 0.75)
                pos = np.array([0.0, y, z])
                sweep_radius = radius + 0.15

                if self._is_valid_position(
                    pos,
                    sweep_radius,
                    start_pos,
                    goal_pos,
                    placed=placed,
                    min_pair_clearance=0.45,
                    wall_margin=0.0,
                ):
                    obstacle = SphereObstacle(
                        pos,
                        radius,
                        speed,
                        amplitude,
                        frequency,
                        client_id,
                        direction=direction,
                        phase=phase,
                    )
                    self.obstacles.append(obstacle)
                    placed.append((pos, sweep_radius))
                    break

    def _generate_walls(self, n_obstacles: int, params: dict,
                       start_pos: np.ndarray, goal_pos: np.ndarray, client_id: int):
        for _ in range(n_obstacles):
            width = self.rng.uniform(*params['width'])
            height = self.rng.uniform(*params['height'])
            thickness = params['thickness']


            for _ in range(50):
                x = self.rng.uniform(-self.config.ARENA_SIZE_X/2 + width/2,
                                    self.config.ARENA_SIZE_X/2 - width/2)
                y = self.rng.uniform(-self.config.ARENA_SIZE_Y/2 + thickness/2,
                                    self.config.ARENA_SIZE_Y/2 - thickness/2)
                pos = np.array([x, y, 0.0])


                if self._is_valid_position(pos, width/2, start_pos, goal_pos):
                    obstacle = WallObstacle(pos, width, height, thickness, client_id)
                    self.obstacles.append(obstacle)
                    break

    def _generate_beams(self, n_obstacles: int, params: dict,
                       start_pos: np.ndarray, goal_pos: np.ndarray, client_id: int):
        placed = []
        half_x = self.config.ARENA_SIZE_X / 2.0
        half_y = self.config.ARENA_SIZE_Y / 2.0
        start_goal_clearance = float(getattr(self.config, "DYNAMIC_START_GOAL_CLEARANCE", 1.1))
        pair_y_clearance = float(getattr(self.config, "BEAM_PAIR_Y_CLEARANCE", 0.9))
        bottom_gap = float(getattr(self.config, "BEAM_BOTTOM_GAP", 0.45))
        ceiling_gap = float(getattr(self.config, "BEAM_CEILING_GAP", 0.25))
        swing_axis = params.get("swing_axis", "yaw")

        for _ in range(n_obstacles):
            for _ in range(120):
                length = self.rng.uniform(*params['length'])
                thickness = params['thickness']
                swing_angle = self._sample_scalar(params.get('swing_angle', 0))
                swing_period = self.rng.uniform(*params.get('swing_period', (3.0, 5.0)))
                phase = self.rng.uniform(0.0, 2.0 * np.pi)

                height_min, height_max = params['height']
                if swing_axis in ("pitch", "roll"):
                    angle_rad = np.radians(abs(swing_angle))
                    vertical_reach = length * 0.5 * np.sin(angle_rad) + thickness * 0.5
                    height_min = max(height_min, bottom_gap + vertical_reach)
                    height_max = min(height_max, self.config.ARENA_HEIGHT - ceiling_gap - vertical_reach)
                    if height_min >= height_max:
                        continue

                height = self.rng.uniform(height_min, height_max)

                x_min = -half_x + length / 2.0
                x_max = half_x - length / 2.0
                y_min = -half_y + start_goal_clearance
                y_max = half_y - start_goal_clearance
                if x_min > x_max or y_min > y_max:
                    continue

                x = self.rng.uniform(x_min, x_max)
                y = self.rng.uniform(y_min, y_max)
                pos = np.array([x, y, 0.0])

                if self._is_valid_bar_position(
                    pos=pos,
                    length=length,
                    thickness=thickness,
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    placed=placed,
                    min_start_goal_clearance=start_goal_clearance,
                    min_pair_y_clearance=pair_y_clearance,
                ):
                    obstacle = BeamObstacle(pos, length, height, thickness,
                                            swing_angle, swing_period, client_id,
                                            swing_axis=swing_axis, phase=phase)
                    self.obstacles.append(obstacle)
                    placed.append((pos, length, thickness))
                    break

    def _generate_boxes(self, n_obstacles: int, params: dict,
                       start_pos: np.ndarray, goal_pos: np.ndarray, client_id: int):
        for _ in range(n_obstacles):
            size = self.rng.uniform(*params['size'])
            height = self.rng.uniform(*params['height'])


            for _ in range(50):
                x = self.rng.uniform(-self.config.ARENA_SIZE_X/2 + 1, self.config.ARENA_SIZE_X/2 - 1)
                y = self.rng.uniform(-self.config.ARENA_SIZE_Y/2 + 1, self.config.ARENA_SIZE_Y/2 - 1)
                pos = np.array([x, y, 0.0])

                if self._is_valid_position(pos, size/2, start_pos, goal_pos):
                    obstacle = BoxObstacle(pos, size, height, client_id)
                    self.obstacles.append(obstacle)
                    break

    def _generate_gates(self, n_obstacles: int, params: dict,
                        start_pos: np.ndarray, goal_pos: np.ndarray, client_id: int):
        half_x = self.config.ARENA_SIZE_X / 2.0
        half_y = self.config.ARENA_SIZE_Y / 2.0
        start_goal_clearance = float(getattr(self.config, "DYNAMIC_START_GOAL_CLEARANCE", 1.1))
        y_min = -half_y + start_goal_clearance + 0.7
        y_max = half_y - start_goal_clearance - 0.7
        if y_min >= y_max:
            return

        gate_ys = np.linspace(y_min, y_max, n_obstacles)
        prev_gap_center = 0.0

        for gate_idx, y in enumerate(gate_ys):
            gap_min, gap_max = params.get('gap_center', (-0.4, 0.4))
            raw_gap_center = self.rng.uniform(gap_min, gap_max)
            if gate_idx > 0 and abs(raw_gap_center - prev_gap_center) < 0.25:
                raw_gap_center = -prev_gap_center
            gap_center = float(np.clip(raw_gap_center, gap_min, gap_max))
            prev_gap_center = gap_center
            self._add_gate_at_y(float(y), params, gap_center, client_id)

    def _add_gate_at_y(self, y: float, params: dict, gap_center: float, client_id: int):
        half_x = self.config.ARENA_SIZE_X / 2.0
        gap_width = self.rng.uniform(*params['gap_width'])
        height = self.rng.uniform(*params['height'])
        thickness = params['thickness']
        gap_left = gap_center - gap_width / 2.0
        gap_right = gap_center + gap_width / 2.0
        x_min, x_max = -half_x, half_x

        left_width = max(0.0, gap_left - x_min)
        right_width = max(0.0, x_max - gap_right)
        if left_width > 0.25:
            left_x = x_min + left_width / 2.0
            self.obstacles.append(WallObstacle(
                np.array([left_x, y, 0.0]),
                left_width,
                height,
                thickness,
                client_id
            ))
        if right_width > 0.25:
            right_x = gap_right + right_width / 2.0
            self.obstacles.append(WallObstacle(
                np.array([right_x, y, 0.0]),
                right_width,
                height,
                thickness,
                client_id
            ))

    def _generate_slalom(self, n_obstacles: int, params: dict,
                         start_pos: np.ndarray, goal_pos: np.ndarray, client_id: int):
        half_y = self.config.ARENA_SIZE_Y / 2.0
        start_goal_clearance = float(getattr(self.config, "DYNAMIC_START_GOAL_CLEARANCE", 1.1))
        y_min = -half_y + start_goal_clearance + 0.5
        y_max = half_y - start_goal_clearance - 0.5
        if y_min >= y_max:
            return

        y_values = np.linspace(y_min, y_max, n_obstacles)
        side = -1.0 if self.rng.random() < 0.5 else 1.0

        for y in y_values:
            size = self.rng.uniform(*params['size'])
            height = self.rng.uniform(*params['height'])
            offset = self.rng.uniform(*params['lateral_offset'])
            x = float(side * offset)
            side *= -1.0
            pos = np.array([x, y + self.rng.uniform(-0.15, 0.15), 0.0])

            if self._is_valid_position(pos, size / 2.0, start_pos, goal_pos, wall_margin=0.05):
                self.obstacles.append(BoxObstacle(pos, size, height, client_id))

    def _generate_city_blocks(self, n_obstacles: int, params: dict,
                              start_pos: np.ndarray, goal_pos: np.ndarray, client_id: int):
        half_x = self.config.ARENA_SIZE_X / 2.0
        half_y = self.config.ARENA_SIZE_Y / 2.0
        y_min = -half_y + 1.55
        y_max = half_y - 1.55
        if y_min >= y_max:
            return

        rows = np.linspace(y_min, y_max, n_obstacles)
        center_shift = float(params.get('center_shift', 0.45))
        center_x = self.rng.uniform(-center_shift, center_shift)

        for row_idx, y in enumerate(rows):
            center_x = float(np.clip(
                center_x + self.rng.uniform(-0.25, 0.25),
                -center_shift,
                center_shift
            ))
            avenue_width = self.rng.uniform(*params['avenue_width'])
            block_size = self.rng.uniform(*params['block_size'])
            height = self.rng.uniform(*params['height'])
            gap_left = center_x - avenue_width / 2.0
            gap_right = center_x + avenue_width / 2.0

            left_space = gap_left - (-half_x)
            right_space = half_x - gap_right
            if left_space > block_size + 0.1:
                x = -half_x + min(left_space / 2.0, block_size)
                pos = np.array([x, y + self.rng.uniform(-0.12, 0.12), 0.0])
                if self._is_valid_position(pos, block_size / 2.0, start_pos, goal_pos, wall_margin=0.02):
                    self.obstacles.append(BoxObstacle(pos, block_size, height, client_id))

            if right_space > block_size + 0.1:
                x = half_x - min(right_space / 2.0, block_size)
                pos = np.array([x, y + self.rng.uniform(-0.12, 0.12), 0.0])
                if self._is_valid_position(pos, block_size / 2.0, start_pos, goal_pos, wall_margin=0.02):
                    self.obstacles.append(BoxObstacle(pos, block_size, height, client_id))

            if row_idx % 2 == 1:
                gate_params = {
                    'gap_width': params['gate_gap_width'],
                    'gap_center': (center_x - 0.1, center_x + 0.1),
                    'height': (height, height),
                    'thickness': 0.12,
                }
                self._add_gate_at_y(float(y), gate_params, center_x, client_id)

    def _generate_city_dynamic(self, n_obstacles: int, params: dict,
                               start_pos: np.ndarray, goal_pos: np.ndarray, client_id: int):
        half_x = self.config.ARENA_SIZE_X / 2.0
        half_y = self.config.ARENA_SIZE_Y / 2.0
        y_min = -half_y + 2.35
        y_max = half_y - 2.35
        if y_min >= y_max:
            return

        rows = np.linspace(y_min, y_max, n_obstacles)
        bird_lanes = []
        cross_count = int(self.rng.randint(params['cross_streets'][0], params['cross_streets'][1] + 1))
        cross_y_values = np.linspace(y_min + 2.2, y_max - 2.2, max(cross_count, 1))
        cross_y_values = [
            float(y + self.rng.uniform(-0.45, 0.45))
            for y in cross_y_values[:cross_count]
        ]

        center_shift = float(params.get('center_shift', 0.65))
        center_x = self.rng.uniform(-center_shift, center_shift)

        building_color = [0.34, 0.38, 0.44, 1.0]
        roof_color = [0.22, 0.24, 0.28, 1.0]
        sign_color = [0.10, 0.45, 0.90, 1.0]

        for row_idx, y in enumerate(rows):
            center_x = float(np.clip(
                center_x + self.rng.uniform(-0.35, 0.35),
                -center_shift,
                center_shift
            ))
            avenue_width = self.rng.uniform(*params['avenue_width'])
            block_depth_y = self.rng.uniform(*params['building_depth_y'])
            height = self.rng.uniform(*params['height'])
            gap_left = center_x - avenue_width / 2.0
            gap_right = center_x + avenue_width / 2.0
            near_cross = any(abs(y - cross_y) < params['cross_street_width'] / 2.0 for cross_y in cross_y_values)
            y_jitter = self.rng.uniform(-0.18, 0.18)

            left_space = gap_left - (-half_x)
            right_space = half_x - gap_right
            if not near_cross and left_space > 0.6:
                width_x = max(0.4, left_space - self.rng.uniform(0.05, 0.20))
                x = -half_x + width_x / 2.0
                pos = np.array([x, y + y_jitter, 0.0])
                self.obstacles.append(BoxObstacle(pos, width_x, height, client_id,
                                                  depth=block_depth_y, rgba_color=building_color))
                if self.rng.random() < 0.45:
                    self.obstacles.append(BoxObstacle(
                        np.array([x, y + y_jitter, height]),
                        width_x * 0.55,
                        0.22,
                        client_id,
                        depth=block_depth_y * 0.45,
                        rgba_color=roof_color,
                    ))
            if not near_cross and right_space > 0.6:
                width_x = max(0.4, right_space - self.rng.uniform(0.05, 0.20))
                x = half_x - width_x / 2.0
                pos = np.array([x, y + y_jitter, 0.0])
                self.obstacles.append(BoxObstacle(pos, width_x, height, client_id,
                                                  depth=block_depth_y, rgba_color=building_color))
                if self.rng.random() < 0.45:
                    self.obstacles.append(BoxObstacle(
                        np.array([x, y + y_jitter, height]),
                        width_x * 0.55,
                        0.22,
                        client_id,
                        depth=block_depth_y * 0.45,
                        rgba_color=roof_color,
                    ))
            if not near_cross:
                bird_lanes.append((center_x, y + y_jitter, avenue_width, height))

            if row_idx % 2 == 0:
                for side in (-1.0, 1.0):
                    curb_x = center_x + side * (avenue_width / 2.0 + 0.20)
                    if abs(curb_x) < half_x - 0.15:
                        self.obstacles.append(CylinderObstacle(
                            np.array([curb_x, y + self.rng.uniform(-0.25, 0.25), 0.0]),
                            radius=self.rng.uniform(*params['lamp_radius']),
                            height=self.rng.uniform(*params['lamp_height']),
                            physics_client=client_id,
                        ))
                        if self.rng.random() < 0.5:
                            self.obstacles.append(BoxObstacle(
                                np.array([curb_x - side * 0.18, y, 2.0]),
                                0.55,
                                0.35,
                                client_id,
                                depth=0.06,
                                rgba_color=sign_color,
                            ))



        wire_count = int(self.rng.randint(params['wire_count'][0], params['wire_count'][1] + 1))
        for y in np.linspace(y_min + 1.0, y_max - 1.0, wire_count):
            length = self.rng.uniform(*params['wire_length'])
            z = self.rng.uniform(*params['wire_height'])
            self.obstacles.append(BeamObstacle(
                np.array([0.0, float(y + self.rng.uniform(-0.4, 0.4)), z]),
                length=length,
                height=z,
                thickness=self.rng.uniform(*params['wire_thickness']),
                swing_angle=0.0,
                swing_period=6.0,
                physics_client=client_id,
            ))


        for idx in range(int(self.rng.randint(params['vehicle_count'][0], params['vehicle_count'][1] + 1))):
            lane_x = self.rng.uniform(-0.55, 0.55)
            y = self.rng.uniform(y_min + 0.8, y_max - 0.8)
            direction = np.array([0.0, 1.0, 0.0])
            width, depth = params['vehicle_width'], params['vehicle_length']
            max_amplitude = max(0.25, half_y - abs(y) - 0.8)
            if idx % 3 == 2 and cross_y_values:
                y = float(self.rng.choice(cross_y_values))
                lane_x = self.rng.uniform(-half_x + 1.2, half_x - 1.2)
                direction = np.array([1.0, 0.0, 0.0])
                width, depth = params['vehicle_length'], params['vehicle_width']
                max_amplitude = max(0.25, half_x - abs(lane_x) - 0.8)
            amplitude = min(self.rng.uniform(*params['vehicle_amplitude']), max_amplitude)
            self.obstacles.append(MovingBoxObstacle(
                np.array([lane_x, y, 0.0]),
                width=width,
                depth=depth,
                height=params['vehicle_height'],
                speed=self.rng.uniform(*params['vehicle_speed']),
                amplitude=amplitude,
                frequency=self.rng.uniform(*params['vehicle_frequency']),
                physics_client=client_id,
                direction=direction,
                phase=self.rng.uniform(0.0, 2.0 * np.pi),
            ))


        bird_count = int(self.rng.randint(params['bird_count'][0], params['bird_count'][1] + 1))
        bird_z_low = float(params['bird_height'][0])
        bird_z_high = min(float(params['bird_height'][1]), self.config.ARENA_HEIGHT - 0.65)
        for _ in range(bird_count):
            if bird_lanes:
                lane = bird_lanes[int(self.rng.randint(0, len(bird_lanes)))]
                lane_x, lane_y, avenue_width, roof_height = lane
            else:
                lane_x = self.rng.uniform(-0.4, 0.4)
                lane_y = self.rng.uniform(y_min, y_max)
                avenue_width = self.rng.uniform(*params['avenue_width'])
                roof_height = bird_z_low

            radius = self.rng.uniform(*params['bird_radius'])
            amplitude_limit = max(0.25, avenue_width / 2.0 - radius - 0.08)
            amplitude = min(self.rng.uniform(*params['bird_amplitude']), amplitude_limit)
            z = max(
                self.rng.uniform(bird_z_low, bird_z_high),
                min(bird_z_high, roof_height + float(params.get('bird_roof_clearance', 0.25)))
            )
            self.obstacles.append(SphereObstacle(
                position=np.array([lane_x, lane_y + self.rng.uniform(-0.25, 0.25), z]),
                radius=radius,
                speed=self.rng.uniform(*params['bird_speed']),
                amplitude=amplitude,
                frequency=self.rng.uniform(*params['bird_frequency']),
                physics_client=client_id,
                direction=np.array([1.0, 0.0, 0.0]),
                phase=self.rng.uniform(0.0, 2.0 * np.pi),
            ))

    def _generate_construction_site_dynamic(self, n_obstacles: int, params: dict,
                                            start_pos: np.ndarray, goal_pos: np.ndarray, client_id: int):
        half_x = self.config.ARENA_SIZE_X / 2.0
        half_y = self.config.ARENA_SIZE_Y / 2.0
        y_min = -half_y + 2.8
        y_max = half_y - 2.8
        if y_min >= y_max:
            return

        rows = np.linspace(y_min, y_max, n_obstacles)
        lane_width = self.rng.uniform(*params['lane_width'])
        center_x = self.rng.uniform(-0.45, 0.45)
        frame_color = [0.48, 0.50, 0.54, 1.0]
        slab_color = [0.42, 0.42, 0.40, 1.0]
        scaffold_color = [0.82, 0.68, 0.22, 1.0]
        barrier_color = [0.95, 0.38, 0.12, 1.0]
        cargo_color = [0.78, 0.52, 0.20, 1.0]

        frame_sites = []
        for row_idx, y in enumerate(rows):
            center_x = float(np.clip(center_x + self.rng.uniform(-0.25, 0.25), -0.55, 0.55))
            lane_left = center_x - lane_width / 2.0
            lane_right = center_x + lane_width / 2.0
            depth_y = self.rng.uniform(*params['frame_depth_y'])
            height = self.rng.uniform(*params['frame_height'])
            levels = int(self.rng.randint(params['floor_levels'][0], params['floor_levels'][1] + 1))
            y_center = float(y + self.rng.uniform(-0.22, 0.22))

            side_specs = [
                (-1.0, -half_x, lane_left),
                (1.0, lane_right, half_x),
            ]
            for side, side_min, side_max in side_specs:
                side_space = side_max - side_min
                if side_space < 0.95:
                    continue

                width_x = max(0.75, side_space - self.rng.uniform(0.15, 0.35))
                x_center = side_min + width_x / 2.0 if side < 0 else side_max - width_x / 2.0
                x_center = float(np.clip(x_center, -half_x + width_x / 2.0, half_x - width_x / 2.0))
                frame_sites.append((x_center, y_center, width_x, depth_y, height, side))


                slab_count = max(1, levels)
                level_values = np.linspace(0.85, max(1.1, height - 0.35), slab_count)
                for level in level_values:
                    slab_thickness = self.rng.uniform(*params['floor_slab_thickness'])
                    self.obstacles.append(BoxObstacle(
                        np.array([x_center, y_center, float(level)]),
                        width_x,
                        slab_thickness,
                        client_id,
                        depth=depth_y,
                        rgba_color=slab_color,
                    ))


                column_radius = self.rng.uniform(*params['column_radius'])
                for x_offset, y_offset in (
                    (-width_x / 2.0 + 0.18, -depth_y / 2.0 + 0.18),
                    (width_x / 2.0 - 0.18, depth_y / 2.0 - 0.18),
                ):
                    self.obstacles.append(CylinderObstacle(
                        np.array([x_center + x_offset, y_center + y_offset, 0.0]),
                        radius=column_radius,
                        height=height,
                        physics_client=client_id,
                    ))


                facade_x = lane_left - 0.18 if side < 0 else lane_right + 0.18
                facade_x = float(np.clip(facade_x, -half_x + 0.12, half_x - 0.12))
                scaffold_height = min(height, self.rng.uniform(*params['scaffold_height']))
                scaffold_radius = self.rng.uniform(*params['scaffold_radius'])
                for y_offset in (-depth_y * 0.35, depth_y * 0.35):
                    self.obstacles.append(CylinderObstacle(
                        np.array([facade_x, y_center + y_offset, 0.0]),
                        radius=scaffold_radius,
                        height=scaffold_height,
                        physics_client=client_id,
                    ))
                z = self.rng.uniform(1.1, max(1.25, scaffold_height - 0.25))
                self.obstacles.append(BeamObstacle(
                    np.array([facade_x - side * 0.28, y_center, float(z)]),
                    length=min(width_x, 1.25),
                    height=float(z),
                    thickness=self.rng.uniform(*params['scaffold_beam_thickness']),
                    swing_angle=0.0,
                    swing_period=6.0,
                    physics_client=client_id,
                ))


                if row_idx % 2 == 0:
                    self.obstacles.append(BoxObstacle(
                        np.array([facade_x - side * 0.10, y_center + self.rng.uniform(-0.25, 0.25), 0.0]),
                        0.18,
                        self.rng.uniform(*params['barrier_height']),
                        client_id,
                        depth=min(depth_y * 0.75, 1.2),
                        rgba_color=barrier_color,
                    ))
                if self.rng.random() < 0.35:
                    self.obstacles.append(BoxObstacle(
                        np.array([facade_x - side * 0.38, y_center + self.rng.uniform(-0.35, 0.35), 0.0]),
                        self.rng.uniform(0.35, 0.65),
                        self.rng.uniform(0.18, 0.45),
                        client_id,
                        depth=self.rng.uniform(0.35, 0.75),
                        rgba_color=frame_color,
                    ))


        crane_count = int(self.rng.randint(params['crane_count'][0], params['crane_count'][1] + 1))
        crane_y_values = np.linspace(y_min + 2.0, y_max - 2.0, max(crane_count, 1))
        for idx in range(crane_count):
            side = -1.0 if idx % 2 == 0 else 1.0
            mast_x = side * self.rng.uniform(half_x - 1.25, half_x - 0.75)
            mast_y = float(crane_y_values[idx] + self.rng.uniform(-0.6, 0.6))
            crane_height = self.rng.uniform(*params['crane_height'])
            mast_radius = self.rng.uniform(*params['crane_mast_radius'])
            self.obstacles.append(CylinderObstacle(
                np.array([mast_x, mast_y, 0.0]),
                radius=mast_radius,
                height=crane_height,
                physics_client=client_id,
            ))
            self.obstacles.append(BeamObstacle(
                np.array([mast_x - side * 0.65, mast_y, crane_height]),
                length=self.rng.uniform(*params['crane_arm_length']),
                height=crane_height,
                thickness=self.rng.uniform(*params['crane_arm_thickness']),
                swing_angle=float(params['crane_swing_angle']),
                swing_period=self.rng.uniform(*params['crane_swing_period']),
                physics_client=client_id,
                swing_axis="yaw",
                phase=self.rng.uniform(0.0, 2.0 * np.pi),
            ))


        for _ in range(int(self.rng.randint(params['load_count'][0], params['load_count'][1] + 1))):
            size = self.rng.uniform(*params['load_size'])
            y = self.rng.uniform(y_min + 1.2, y_max - 1.2)
            z = self.rng.uniform(2.4, min(4.0, self.config.ARENA_HEIGHT - 1.2))
            direction = np.array([1.0, 0.0, 0.0]) if self.rng.random() < 0.5 else np.array([0.0, 1.0, 0.0])
            amplitude_limit = half_x - lane_width / 2.0 - size - 0.25 if abs(direction[0]) > 0 else half_y - abs(y) - 1.0
            amplitude = min(self.rng.uniform(*params['load_swing_amplitude']), max(0.25, amplitude_limit))
            self.obstacles.append(MovingBoxObstacle(
                np.array([self.rng.uniform(-0.35, 0.35), y, z]),
                width=size,
                depth=size,
                height=self.rng.uniform(*params['load_height']),
                speed=0.45,
                amplitude=amplitude,
                frequency=self.rng.uniform(*params['load_frequency']),
                physics_client=client_id,
                direction=direction,
                phase=self.rng.uniform(0.0, 2.0 * np.pi),
                rgba_color=cargo_color,
            ))


        for idx in range(int(self.rng.randint(params['vehicle_count'][0], params['vehicle_count'][1] + 1))):
            x = self.rng.uniform(-lane_width * 0.25, lane_width * 0.25)
            y = self.rng.uniform(y_min + 1.0, y_max - 1.0)
            max_amplitude = max(0.35, half_y - abs(y) - 1.1)
            self.obstacles.append(MovingBoxObstacle(
                np.array([x, y, 0.0]),
                width=params['vehicle_width'],
                depth=params['vehicle_length'],
                height=params['vehicle_height'],
                speed=self.rng.uniform(*params['vehicle_speed']),
                amplitude=min(self.rng.uniform(*params['vehicle_amplitude']), max_amplitude),
                frequency=self.rng.uniform(*params['vehicle_frequency']),
                physics_client=client_id,
                direction=np.array([0.0, 1.0, 0.0]),
                phase=self.rng.uniform(0.0, 2.0 * np.pi),
                rgba_color=[0.95, 0.72, 0.10, 1.0] if idx % 2 == 0 else [0.18, 0.42, 0.82, 1.0],
            ))


        for _ in range(int(self.rng.randint(params['lift_count'][0], params['lift_count'][1] + 1))):
            if frame_sites:
                x_center, y_center, width_x, depth_y, height, side = frame_sites[int(self.rng.randint(0, len(frame_sites)))]
                x = x_center - side * (width_x / 2.0 + 0.12)
                y = y_center + self.rng.uniform(-depth_y * 0.35, depth_y * 0.35)
                max_amp = max(0.3, min(height - 0.8, self.config.ARENA_HEIGHT - 1.1))
            else:
                x = self.rng.choice([-1.0, 1.0]) * (half_x - 0.8)
                y = self.rng.uniform(y_min, y_max)
                max_amp = 1.2
            lift_size = self.rng.uniform(*params['lift_size'])
            self.obstacles.append(MovingBoxObstacle(
                np.array([float(np.clip(x, -half_x + 0.4, half_x - 0.4)), y, 0.55]),
                width=lift_size,
                depth=lift_size,
                height=params['lift_height'],
                speed=0.35,
                amplitude=min(self.rng.uniform(*params['lift_amplitude']), max_amp),
                frequency=self.rng.uniform(*params['lift_frequency']),
                physics_client=client_id,
                direction=np.array([0.0, 0.0, 1.0]),
                phase=self.rng.uniform(0.0, 2.0 * np.pi),
                rgba_color=[0.30, 0.62, 0.86, 1.0],
            ))


        pipe_count = int(self.rng.randint(params['swinging_pipe_count'][0], params['swinging_pipe_count'][1] + 1))
        for y in np.linspace(y_min + 1.5, y_max - 1.5, max(pipe_count, 1))[:pipe_count]:
            length = self.rng.uniform(*params['swinging_pipe_length'])
            x = self.rng.uniform(-0.35, 0.35)
            z = self.rng.uniform(2.5, min(4.4, self.config.ARENA_HEIGHT - 1.0))
            self.obstacles.append(SwingingStickObstacle(
                np.array([x, float(y + self.rng.uniform(-0.55, 0.55)), z]),
                length=length,
                thickness=self.rng.uniform(*params['swinging_pipe_thickness']),
                swing_angle=float(params['swinging_pipe_angle']),
                swing_period=self.rng.uniform(*params['swinging_pipe_period']),
                physics_client=client_id,
                vertical_swing=False,
                phase=self.rng.uniform(0.0, 2.0 * np.pi),
            ))


        debris_count = int(self.rng.randint(params['debris_count'][0], params['debris_count'][1] + 1))
        z_low, z_high = params['debris_height']
        z_high = min(float(z_high), self.config.ARENA_HEIGHT - 0.65)
        for _ in range(debris_count):
            radius = self.rng.uniform(*params['debris_radius'])
            y = self.rng.uniform(y_min, y_max)
            self.obstacles.append(SphereObstacle(
                position=np.array([
                    self.rng.uniform(-lane_width * 0.35, lane_width * 0.35),
                    y,
                    self.rng.uniform(float(z_low), z_high),
                ]),
                radius=radius,
                speed=self.rng.uniform(*params['debris_speed']),
                amplitude=self.rng.uniform(*params['debris_amplitude']),
                frequency=self.rng.uniform(*params['debris_frequency']),
                physics_client=client_id,
                direction=np.array([1.0, 0.0, 0.0]),
                phase=self.rng.uniform(0.0, 2.0 * np.pi),
            ))

    def _generate_swinging_sticks(self, n_obstacles: int, params: dict,
                                  start_pos: np.ndarray, goal_pos: np.ndarray, client_id: int):
        placed = []
        half_x = self.config.ARENA_SIZE_X / 2.0
        half_y = self.config.ARENA_SIZE_Y / 2.0
        start_goal_clearance = float(getattr(self.config, "DYNAMIC_START_GOAL_CLEARANCE", 1.1))
        pair_y_clearance = float(getattr(self.config, "STICK_PAIR_Y_CLEARANCE", 0.75))
        bottom_gap = self.config.ARENA_HEIGHT * float(getattr(self.config, "STICK_BOTTOM_GAP_FRACTION", 1.0 / 3.0))
        ceiling_gap = float(getattr(self.config, "STICK_CEILING_GAP", 0.25))
        side_margin = float(getattr(self.config, "STICK_SIDE_MARGIN", 0.15))

        for _ in range(n_obstacles):
            for _ in range(160):
                length = self.rng.uniform(*params['length'])
                thickness = params['thickness']
                swing_angle = params['swing_angle']
                swing_period = self.rng.uniform(*params['swing_period'])
                vertical_swing = params.get('vertical_swing', False)
                vertical_amplitude = (
                    self.rng.uniform(*params['vertical_amplitude'])
                    if 'vertical_amplitude' in params
                    else 0.15
                )
                if not vertical_swing:
                    vertical_amplitude = 0.0

                z_min = bottom_gap + thickness + vertical_amplitude
                z_max = self.config.ARENA_HEIGHT - ceiling_gap - thickness - vertical_amplitude
                if z_min >= z_max:
                    continue

                x_min = -half_x + length / 2.0 + side_margin
                x_max = half_x - length / 2.0 - side_margin
                y_min = -half_y + start_goal_clearance
                y_max = half_y - start_goal_clearance
                if x_min > x_max or y_min > y_max:
                    continue

                x = self.rng.uniform(x_min, x_max)
                y = self.rng.uniform(y_min, y_max)
                z = self.rng.uniform(z_min, z_max)
                phase = self.rng.uniform(0.0, 2.0 * np.pi)
                pos = np.array([x, y, z])

                if self._is_valid_bar_position(
                    pos=pos,
                    length=length,
                    thickness=thickness,
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    placed=placed,
                    min_start_goal_clearance=start_goal_clearance,
                    min_pair_y_clearance=pair_y_clearance,
                ):
                    obstacle = SwingingStickObstacle(pos, length, thickness,
                                                    swing_angle, swing_period, client_id,
                                                    vertical_swing, vertical_amplitude,
                                                    phase=phase)
                    self.obstacles.append(obstacle)
                    placed.append((pos, length, thickness))
                    break

    def _sample_scalar(self, value):
        if isinstance(value, (tuple, list)):
            return self.rng.uniform(*value)
        return value

    @staticmethod
    def _distance_point_to_x_bar(point: np.ndarray, bar_pos: np.ndarray, length: float) -> float:
        dx = max(abs(point[0] - bar_pos[0]) - length / 2.0, 0.0)
        dy = abs(point[1] - bar_pos[1])
        return float(np.hypot(dx, dy))

    def _is_valid_bar_position(
        self,
        pos: np.ndarray,
        length: float,
        thickness: float,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        placed: List[Tuple[np.ndarray, float, float]],
        min_start_goal_clearance: float,
        min_pair_y_clearance: float,
    ) -> bool:
        half_x = self.config.ARENA_SIZE_X / 2.0
        half_y = self.config.ARENA_SIZE_Y / 2.0
        if pos[0] - length / 2.0 < -half_x or pos[0] + length / 2.0 > half_x:
            return False
        if abs(pos[1]) + thickness > half_y:
            return False

        required_clearance = min_start_goal_clearance + thickness
        if self._distance_point_to_x_bar(start_pos, pos, length) < required_clearance:
            return False
        if self._distance_point_to_x_bar(goal_pos, pos, length) < required_clearance:
            return False

        for other_pos, other_length, other_thickness in placed:
            left = pos[0] - length / 2.0
            right = pos[0] + length / 2.0
            other_left = other_pos[0] - other_length / 2.0
            other_right = other_pos[0] + other_length / 2.0
            expanded_overlap = min(right, other_right) - max(left, other_left)
            y_clearance = min_pair_y_clearance + thickness + other_thickness
            if expanded_overlap > -0.25 and abs(pos[1] - other_pos[1]) < y_clearance:
                return False

        return True

    def _is_valid_position(
        self,
        pos: np.ndarray,
        radius: float,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        placed: List[Tuple[np.ndarray, float]] = None,
        min_pair_clearance: float = 0.0,
        wall_margin: float = 0.0,
    ) -> bool:

        half_x = self.config.ARENA_SIZE_X / 2.0
        half_y = self.config.ARENA_SIZE_Y / 2.0
        if abs(pos[0]) + radius + wall_margin > half_x:
            return False
        if abs(pos[1]) + radius + wall_margin > half_y:
            return False


        dist_to_start = np.linalg.norm(pos[:2] - start_pos[:2])
        dist_to_goal = np.linalg.norm(pos[:2] - goal_pos[:2])

        min_clearance = 1.0
        if dist_to_start < min_clearance + radius or dist_to_goal < min_clearance + radius:
            return False


        if placed:
            for other_pos, other_radius in placed:
                dist_to_other = np.linalg.norm(pos[:2] - other_pos[:2])
                if dist_to_other < radius + other_radius + min_pair_clearance:
                    return False

        return True

    def points_clear_of_obstacles(self, points, clearance: float = None) -> bool:
        if clearance is None:
            params = getattr(self.config, "OBSTACLE_TYPES", {}).get(self.current_obstacle_type, {})
            clearance = float(params.get("spawn_clearance", getattr(self.config, "MIN_CLEARANCE", 0.35) * 0.5))

        for point in points:
            point = np.array(point, dtype=float)
            for obstacle in self.obstacles:
                if not self._point_clear_of_obstacle(point, obstacle, clearance):
                    return False
        return True

    def _point_clear_of_obstacle(self, point: np.ndarray, obstacle, clearance: float) -> bool:
        obstacle_type = obstacle.__class__.__name__

        if obstacle_type in ("BoxObstacle", "MovingBoxObstacle"):
            pos = np.array(
                getattr(obstacle, "initial_position", getattr(obstacle, "position", [0.0, 0.0, 0.0])),
                dtype=float
            )
            width = float(getattr(obstacle, "width", getattr(obstacle, "size", 0.0)))
            depth = float(getattr(obstacle, "depth", getattr(obstacle, "size", 0.0)))
            height = float(getattr(obstacle, "height", 0.0))
            in_xy = (
                abs(point[0] - pos[0]) <= width / 2.0 + clearance and
                abs(point[1] - pos[1]) <= depth / 2.0 + clearance
            )
            in_z = pos[2] - clearance <= point[2] <= pos[2] + height + clearance
            if in_xy and in_z:
                return False

            if obstacle_type == "MovingBoxObstacle":
                direction = np.array(getattr(obstacle, "direction", [0.0, 1.0, 0.0]), dtype=float)
                amplitude = float(getattr(obstacle, "amplitude", 0.0))
                a = pos[:2] - direction[:2] * amplitude
                b = pos[:2] + direction[:2] * amplitude
                if self._distance_point_to_segment_2d(point[:2], a, b) <= max(width, depth) / 2.0 + clearance and in_z:
                    return False
            return True

        if obstacle_type in ("CylinderObstacle", "StaticObstacle"):
            pos = np.array(getattr(obstacle, "position", [0.0, 0.0, 0.0]), dtype=float)
            radius = float(getattr(obstacle, "radius", 0.0))
            height = float(getattr(obstacle, "height", 0.0))
            if np.linalg.norm(point[:2] - pos[:2]) <= radius + clearance and pos[2] - clearance <= point[2] <= pos[2] + height + clearance:
                return False
            return True

        if obstacle_type == "SphereObstacle":
            center = np.array(getattr(obstacle, "initial_position", getattr(obstacle, "position", [0.0, 0.0, 0.0])), dtype=float)
            radius = float(getattr(obstacle, "radius", 0.0))
            return np.linalg.norm(point - center) > radius + clearance

        if obstacle_type in ("BeamObstacle", "SwingingStickObstacle"):
            pos = np.array(getattr(obstacle, "position", [0.0, 0.0, 0.0]), dtype=float)
            length = float(getattr(obstacle, "length", 0.0))
            thickness = float(getattr(obstacle, "thickness", 0.0))
            z_center = float(getattr(obstacle, "height", pos[2]))
            in_x = abs(point[0] - pos[0]) <= length / 2.0 + clearance
            in_y = abs(point[1] - pos[1]) <= thickness / 2.0 + clearance
            in_z = abs(point[2] - z_center) <= thickness / 2.0 + clearance
            if in_x and in_y and in_z:
                return False
            return True

        return True

    @staticmethod
    def _distance_point_to_segment_2d(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-9:
            return float(np.linalg.norm(point - a))
        t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
        closest = a + t * ab
        return float(np.linalg.norm(point - closest))

    def update_dynamic_obstacles(self, dt):
        for obstacle in self.obstacles:
            if hasattr(obstacle, 'dynamic') and obstacle.dynamic:
                obstacle.update(dt)

    def get_obstacles(self):
        return self.obstacles

    def cleanup(self):
        for obstacle in self.obstacles:
            obstacle.cleanup()
        self.obstacles = []