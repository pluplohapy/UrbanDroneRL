"""
Stage Pretrain: Diverse obstacles for curriculum learning.
No RRT* planner - pure RL training.
"""

import numpy as np
from typing import Tuple
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
                          'random' - random type each reset
                          'cylinders', 'spheres', 'walls', 'beams', 'boxes', 'swinging_sticks' - specific type
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.obstacle_type = obstacle_type
        self.config = load_config('pretrain')
        self.client_id = None

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
        if self.obstacle_type == 'random':
            chosen_type = self.rng.choice(list(self.config.OBSTACLE_TYPES.keys()))
        else:
            chosen_type = self.obstacle_type

        # Generate obstacles of chosen type
        self._generate_obstacles(chosen_type, start_pos, goal_pos, client_id)

        return start_pos, goal_pos

    def _generate_start_goal_zones(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate start and goal in opposite zones to ensure long paths.

        Returns:
            start_pos, goal_pos: Starting and goal positions
        """
        # Randomly choose which side for start
        if self.rng.random() < 0.5:
            # Start left, Goal right
            start_x = self.rng.uniform(*self.config.START_ZONE_X)
            goal_x = self.rng.uniform(*self.config.GOAL_ZONE_X)
        else:
            # Start right, Goal left
            start_x = self.rng.uniform(*self.config.GOAL_ZONE_X)
            goal_x = self.rng.uniform(*self.config.START_ZONE_X)

        start_y = self.rng.uniform(*self.config.START_ZONE_Y)
        goal_y = self.rng.uniform(*self.config.GOAL_ZONE_Y)
        start_z = self.rng.uniform(*self.config.START_ZONE_Z)
        goal_z = self.rng.uniform(*self.config.GOAL_ZONE_Z)

        start_pos = np.array([start_x, start_y, start_z])
        goal_pos = np.array([goal_x, goal_y, goal_z])

        return start_pos, goal_pos

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
        params = self.config.OBSTACLE_TYPES[obstacle_type]
        n_obstacles = self.rng.randint(params['count'][0], params['count'][1] + 1)

        if obstacle_type == 'cylinders':
            self._generate_cylinders(n_obstacles, params, start_pos, goal_pos, client_id)
        elif obstacle_type == 'spheres':
            self._generate_spheres(n_obstacles, params, start_pos, goal_pos, client_id)
        elif obstacle_type == 'walls':
            self._generate_walls(n_obstacles, params, start_pos, goal_pos, client_id)
        elif obstacle_type == 'beams':
            self._generate_beams(n_obstacles, params, start_pos, goal_pos, client_id)
        elif obstacle_type == 'boxes':
            self._generate_boxes(n_obstacles, params, start_pos, goal_pos, client_id)
        elif obstacle_type == 'swinging_sticks':
            self._generate_swinging_sticks(n_obstacles, params, start_pos, goal_pos, client_id)

    def _generate_cylinders(self, n_obstacles: int, params: dict,
                           start_pos: np.ndarray, goal_pos: np.ndarray, client_id: int):
        """Generate cylindrical obstacles."""
        for _ in range(n_obstacles):
            radius = self.rng.uniform(*params['radius'])
            height = self.rng.uniform(*params['height'])

            # Find valid position
            for _ in range(50):
                x = self.rng.uniform(-self.config.ARENA_SIZE_X/2 + 1, self.config.ARENA_SIZE_X/2 - 1)
                y = self.rng.uniform(-self.config.ARENA_SIZE_Y/2 + 1, self.config.ARENA_SIZE_Y/2 - 1)
                pos = np.array([x, y, 0.0])

                if self._is_valid_position(pos, radius, start_pos, goal_pos):
                    obstacle = CylinderObstacle(pos, radius, height, client_id)
                    self.obstacles.append(obstacle)
                    break

    def _generate_spheres(self, n_obstacles: int, params: dict,
                         start_pos: np.ndarray, goal_pos: np.ndarray, client_id: int):
        """Generate spherical obstacles (birds)."""
        for _ in range(n_obstacles):
            radius = self.rng.uniform(*params['radius'])
            speed = self.rng.uniform(*params['speed'])
            amplitude = self.rng.uniform(*self.config.SPHERE_MOVEMENT_AMPLITUDE)
            frequency = self.rng.uniform(*self.config.SPHERE_MOVEMENT_FREQUENCY)

            # Find valid position
            for _ in range(50):
                x = self.rng.uniform(-self.config.ARENA_SIZE_X/2 + 2, self.config.ARENA_SIZE_X/2 - 2)
                y = self.rng.uniform(-self.config.ARENA_SIZE_Y/2 + 2, self.config.ARENA_SIZE_Y/2 - 2)
                z = self.rng.uniform(1.0, self.config.ARENA_HEIGHT - 1.0)
                pos = np.array([x, y, z])

                if self._is_valid_position(pos, radius + amplitude, start_pos, goal_pos):
                    obstacle = SphereObstacle(pos, radius, speed, amplitude, frequency, client_id)
                    self.obstacles.append(obstacle)
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
        """Generate horizontal beam obstacles."""
        for _ in range(n_obstacles):
            length = self.rng.uniform(*params['length'])
            height = self.rng.uniform(*params['height'])
            thickness = params['thickness']
            swing_angle = params.get('swing_angle', 0)
            swing_period = self.rng.uniform(*params.get('swing_period', (3.0, 5.0)))

            # Find valid position
            for _ in range(50):
                x = self.rng.uniform(-self.config.ARENA_SIZE_X/2 + length/2,
                                    self.config.ARENA_SIZE_X/2 - length/2)
                y = self.rng.uniform(-self.config.ARENA_SIZE_Y/2 + length/2,
                                    self.config.ARENA_SIZE_Y/2 - length/2)
                pos = np.array([x, y, 0.0])

                if self._is_valid_position(pos, length/2, start_pos, goal_pos):
                    obstacle = BeamObstacle(pos, length, height, thickness,
                                          swing_angle, swing_period, client_id)
                    self.obstacles.append(obstacle)
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

    def _generate_swinging_sticks(self, n_obstacles: int, params: dict,
                                  start_pos: np.ndarray, goal_pos: np.ndarray, client_id: int):
        """Generate swinging stick obstacles (branches)."""
        for _ in range(n_obstacles):
            length = self.rng.uniform(*params['length'])
            thickness = params['thickness']
            swing_angle = params['swing_angle']
            swing_period = self.rng.uniform(*params['swing_period'])
            vertical_swing = params.get('vertical_swing', False)

            # Find valid position (distributed across all heights)
            for _ in range(50):
                x = self.rng.uniform(-self.config.ARENA_SIZE_X/2 + 1, self.config.ARENA_SIZE_X/2 - 1)
                y = self.rng.uniform(-self.config.ARENA_SIZE_Y/2 + 1, self.config.ARENA_SIZE_Y/2 - 1)
                z = self.rng.uniform(0.8, self.config.ARENA_HEIGHT - 0.5)  # От низа до верха
                pos = np.array([x, y, z])

                if self._is_valid_position(pos, length/2, start_pos, goal_pos):
                    obstacle = SwingingStickObstacle(pos, length, thickness,
                                                    swing_angle, swing_period, client_id, vertical_swing)
                    self.obstacles.append(obstacle)
                    break

    def _is_valid_position(self, pos: np.ndarray, radius: float,
                          start_pos: np.ndarray, goal_pos: np.ndarray) -> bool:
        """
        Check if position is valid (not too close to start/goal).

        Args:
            pos: Position to check
            radius: Effective radius of obstacle
            start_pos: Start position
            goal_pos: Goal position

        Returns:
            True if valid position
        """
        # Check clearance from start and goal
        dist_to_start = np.linalg.norm(pos[:2] - start_pos[:2])
        dist_to_goal = np.linalg.norm(pos[:2] - goal_pos[:2])

        min_clearance = 1.0  # From base config
        if dist_to_start < min_clearance + radius or dist_to_goal < min_clearance + radius:
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
