"""
Stage Pretrain: Diverse obstacles for curriculum learning.
No RRT* planner - pure RL training.
"""

from collections import deque
import numpy as np
from typing import List, Tuple
from scenarios.base_scenario import BaseScenario
from scenarios.pretrain_obstacles import (
    CylinderObstacle, SphereObstacle, WallObstacle,
    BeamObstacle, BoxObstacle, SwingingStickObstacle
)
from config import load_config


class StagePretrainScenario(BaseScenario):
    """Pretrain scenario with diverse obstacle types."""

    def __init__(self, obstacle_type='random', seed=None):
        """
        Initialize pretrain scenario.

        Args:
            obstacle_type: Type of obstacles to generate
                          'random' - random type each reset (includes empty)
                          'dynamic_mix' - random dynamic type each reset
                          'empty', 'cylinders', 'spheres', 'crossing_spheres',
                          'walls', 'beams', 'boxes', 'gates', 'slalom',
                          'city_blocks', 'swinging_sticks' - specific type
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.obstacle_type = obstacle_type
        self.current_obstacle_type = None
        self.config = load_config('pretrain')
        self.client_id = None

    def _resolve_obstacle_type(self) -> str:
        """Resolve obstacle mode into a concrete obstacle type."""
        if self.obstacle_type == 'random':
            return str(self.rng.choice(list(self.config.OBSTACLE_TYPES.keys())))

        if self.obstacle_type == 'dynamic_mix':
            dynamic_types = [
                name for name, params in self.config.OBSTACLE_TYPES.items()
                if bool(params.get('dynamic', False))
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
        """
        Generate scenario: create obstacles and return start/goal.
        Same as reset() for pretrain.

        Args:
            client_id: PyBullet client ID

        Returns:
            start_pos, goal_pos: Starting and goal positions
        """
        return self.reset(client_id)

    def reset(self, client_id):
        """
        Reset scenario: generate start/goal and create obstacles.

        Args:
            client_id: PyBullet client ID

        Returns:
            start_pos, goal_pos: Starting and goal positions
        """
        self.client_id = client_id

        # Clear obstacles list (p.resetSimulation() already removed them)
        self.obstacles = []

        # Generate start and goal in opposite zones
        start_pos, goal_pos = self._generate_start_goal_zones()

        # Choose obstacle type
        chosen_type = self._resolve_obstacle_type()
        self.current_obstacle_type = chosen_type

        # Generate obstacles of chosen type
        self._generate_obstacles(chosen_type, start_pos, goal_pos, client_id)

        return start_pos, goal_pos

    def _generate_start_goal_zones(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate start and goal in opposite zones to ensure long paths.

        Returns:
            start_pos, goal_pos: Starting and goal positions
        """
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
        """Sample goal X, optionally biased toward corridor side/corner bands."""
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
        """Configured terminal clearance from physical arena walls."""
        return max(0.0, float(getattr(self.config, "GOAL_WALL_CLEARANCE", 0.0)))

    def _clip_goal_range_to_arena(self, value_range, axis: str) -> Tuple[float, float]:
        """Clip a terminal sampling range so goals are not placed in wall-danger zones."""
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
        """Sample goal Y, optionally biased toward the end wall side of its zone."""
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
        """
        Generate obstacles of specified type.

        Args:
            obstacle_type: Type of obstacles to generate
            start_pos: Start position
            goal_pos: Goal position
            client_id: PyBullet client ID
        """
        if obstacle_type not in self.config.OBSTACLE_TYPES:
            raise ValueError(f"Unknown obstacle type for pretrain scenario: {obstacle_type}")

        self.current_obstacle_type = obstacle_type
        params = self.config.OBSTACLE_TYPES[obstacle_type]
        n_obstacles = self.rng.randint(params['count'][0], params['count'][1] + 1)

        if obstacle_type == 'empty':
            # No obstacles for empty map
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
        elif obstacle_type == 'swinging_sticks':
            self._generate_swinging_sticks(n_obstacles, params, start_pos, goal_pos, client_id)

    def _generate_cylinders(self, n_obstacles: int, params: dict,
                           start_pos: np.ndarray, goal_pos: np.ndarray, client_id: int):
        """
        Generate cylindrical obstacles with map-quality constraints.

        We reject layouts that create dead-ends/blocked passages by requiring:
        1) minimum offset from arena walls,
        2) minimum spacing between cylinders,
        3) a traversable 2D path from start to goal on an inflated occupancy grid.
        """
        wall_margin = float(getattr(self.config, "CYLINDER_WALL_MARGIN", 0.35))
        pair_clearance = float(getattr(self.config, "CYLINDER_PAIR_CLEARANCE", 0.45))
        path_clearance = float(getattr(self.config, "CYLINDER_PATH_CLEARANCE", 0.45))
        grid_resolution = float(getattr(self.config, "CYLINDER_PATH_GRID_RESOLUTION", 0.20))

        # If strict constraints are hard to satisfy, gradually reduce obstacle count.
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
        """Sample non-overlapping cylinders with wall/start/goal clearance."""
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
        """
        Fast XY reachability check on an inflated occupancy grid.

        Cylinders and arena borders are expanded by `clearance` to ensure that
        accepted maps still contain a practical corridor for the drone body.
        """
        half_x = self.config.ARENA_SIZE_X / 2.0
        half_y = self.config.ARENA_SIZE_Y / 2.0
        x_min, x_max = -half_x, half_x
        y_min, y_max = -half_y, half_y

        x_vals = np.arange(x_min, x_max + grid_resolution * 0.5, grid_resolution)
        y_vals = np.arange(y_min, y_max + grid_resolution * 0.5, grid_resolution)
        xx, yy = np.meshgrid(x_vals, y_vals)

        # Block near walls to avoid "wall-hugging" passages that immediately OOB.
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
        """Generate spherical obstacles (birds)."""
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

            # Find valid position
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
        """Generate dynamic spheres crossing the main corridor at separated Y bands."""
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
        """Generate vertical wall obstacles."""
        for _ in range(n_obstacles):
            width = self.rng.uniform(*params['width'])
            height = self.rng.uniform(*params['height'])
            thickness = params['thickness']

            # Find valid position
            for _ in range(50):
                x = self.rng.uniform(-self.config.ARENA_SIZE_X/2 + width/2,
                                    self.config.ARENA_SIZE_X/2 - width/2)
                y = self.rng.uniform(-self.config.ARENA_SIZE_Y/2 + thickness/2,
                                    self.config.ARENA_SIZE_Y/2 - thickness/2)
                pos = np.array([x, y, 0.0])

                # Check clearance (use width as effective radius)
                if self._is_valid_position(pos, width/2, start_pos, goal_pos):
                    obstacle = WallObstacle(pos, width, height, thickness, client_id)
                    self.obstacles.append(obstacle)
                    break

    def _generate_beams(self, n_obstacles: int, params: dict,
                       start_pos: np.ndarray, goal_pos: np.ndarray, client_id: int):
        """Generate swinging beams with feasible spacing and vertical clearance."""
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
        """Generate box/cube obstacles."""
        for _ in range(n_obstacles):
            size = self.rng.uniform(*params['size'])
            height = self.rng.uniform(*params['height'])

            # Find valid position
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
        """Generate wall gates with offset openings along the corridor."""
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
        """Add two wall segments at one Y position, leaving a flyable gap."""
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
        """Generate alternating side blocks that force smooth lateral corrections."""
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
        """
        Generate a small city-like avenue.

        The map keeps a continuous drivable/flyable street by construction: each
        row has two side blocks and, on alternating rows, short gate walls with a
        visible opening. This is harder than cylinders but still PPO-friendly.
        """
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

    def _generate_swinging_sticks(self, n_obstacles: int, params: dict,
                                  start_pos: np.ndarray, goal_pos: np.ndarray, client_id: int):
        """Generate swinging sticks with varied height and guaranteed low gap."""
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
        """Sample scalar config values that may be fixed or expressed as a range."""
        if isinstance(value, (tuple, list)):
            return self.rng.uniform(*value)
        return value

    @staticmethod
    def _distance_point_to_x_bar(point: np.ndarray, bar_pos: np.ndarray, length: float) -> float:
        """Distance in XY from a point to a bar segment aligned with the X axis."""
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
        """Validate an X-aligned bar without overestimating it as a large circle."""
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
        """
        Check if position is valid (not too close to start/goal).

        Args:
            pos: Position to check
            radius: Effective radius of obstacle
            start_pos: Start position
            goal_pos: Goal position
            placed: Already placed obstacle centers/radii in XY
            min_pair_clearance: Extra XY spacing between obstacles
            wall_margin: Extra spacing from arena walls

        Returns:
            True if valid position
        """
        # Keep obstacle away from arena borders.
        half_x = self.config.ARENA_SIZE_X / 2.0
        half_y = self.config.ARENA_SIZE_Y / 2.0
        if abs(pos[0]) + radius + wall_margin > half_x:
            return False
        if abs(pos[1]) + radius + wall_margin > half_y:
            return False

        # Check clearance from start and goal
        dist_to_start = np.linalg.norm(pos[:2] - start_pos[:2])
        dist_to_goal = np.linalg.norm(pos[:2] - goal_pos[:2])

        min_clearance = 1.0  # From base config
        if dist_to_start < min_clearance + radius or dist_to_goal < min_clearance + radius:
            return False

        # Check clearance from already placed obstacles in XY.
        if placed:
            for other_pos, other_radius in placed:
                dist_to_other = np.linalg.norm(pos[:2] - other_pos[:2])
                if dist_to_other < radius + other_radius + min_pair_clearance:
                    return False

        return True

    def update_dynamic_obstacles(self, dt):
        """
        Update dynamic obstacles (spheres, beams, sticks).

        Args:
            dt: Time step
        """
        for obstacle in self.obstacles:
            if hasattr(obstacle, 'dynamic') and obstacle.dynamic:
                obstacle.update(dt)

    def get_obstacles(self):
        """
        Get list of obstacles.

        Returns:
            List of obstacle objects
        """
        return self.obstacles

    def cleanup(self):
        """Remove all obstacles from simulation."""
        for obstacle in self.obstacles:
            obstacle.cleanup()
        self.obstacles = []
