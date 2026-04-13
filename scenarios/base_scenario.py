"""
Base scenario class.
All scenarios must inherit from this and implement required methods.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List
import config


class BaseScenario(ABC):
    """Abstract base class for all scenarios."""

    def __init__(self, seed: int = None):
        """
        Initialize scenario.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.obstacles = []
        self.dynamic_obstacles = []

    @abstractmethod
    def generate(self, physics_client: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate scenario: create obstacles and return start/goal positions.

        Args:
            physics_client: PyBullet physics client ID

        Returns:
            start_pos: [x, y, z] starting position
            goal_pos: [x, y, z] goal position
        """
        pass

    @abstractmethod
    def reset(self, physics_client: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset scenario: remove old obstacles, generate new ones.

        Args:
            physics_client: PyBullet physics client ID

        Returns:
            start_pos: [x, y, z] starting position
            goal_pos: [x, y, z] goal position
        """
        pass

    def update_dynamic_obstacles(self, dt: float):
        """
        Update positions of dynamic obstacles.

        Args:
            dt: Time step in seconds
        """
        for obstacle in self.dynamic_obstacles:
            obstacle.update(dt)

    def cleanup(self):
        """Remove all obstacles from simulation."""
        for obstacle in self.obstacles:
            try:
                obstacle.remove()
            except:
                pass
        for obstacle in self.dynamic_obstacles:
            try:
                obstacle.remove()
            except:
                pass
        self.obstacles = []
        self.dynamic_obstacles = []

    def _generate_start_goal(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate valid start and goal positions.

        Returns:
            start_pos: [x, y, z]
            goal_pos: [x, y, z]
        """
        max_attempts = 1000

        for _ in range(max_attempts):
            # Start in lower half, goal in upper half
            start_x = self.rng.uniform(-config.ARENA_SIZE_X / 2 + 1, config.ARENA_SIZE_X / 2 - 1)
            start_y = self.rng.uniform(-config.ARENA_SIZE_Y / 2 + 1, config.ARENA_SIZE_Y / 2 - 1)
            start_z = self.rng.uniform(0.5, config.ARENA_HEIGHT / 2)

            goal_x = self.rng.uniform(-config.ARENA_SIZE_X / 2 + 1, config.ARENA_SIZE_X / 2 - 1)
            goal_y = self.rng.uniform(-config.ARENA_SIZE_Y / 2 + 1, config.ARENA_SIZE_Y / 2 - 1)
            goal_z = self.rng.uniform(config.ARENA_HEIGHT / 2, config.ARENA_HEIGHT - 0.5)

            start_pos = np.array([start_x, start_y, start_z])
            goal_pos = np.array([goal_x, goal_y, goal_z])

            # Check minimum distance
            dist = np.linalg.norm(goal_pos - start_pos)
            if dist < config.MIN_START_GOAL_DIST:
                continue

            # Check clearance from obstacles
            if not self._check_clearance(start_pos) or not self._check_clearance(goal_pos):
                continue

            return start_pos, goal_pos

        # Fallback: return positions without obstacle clearance check
        start_pos = np.array([0.0, 0.0, 1.0])
        goal_pos = np.array([0.0, 0.0, 4.0])
        return start_pos, goal_pos

    def _check_clearance(self, position: np.ndarray) -> bool:
        """
        Check if position has minimum clearance from all obstacles.

        Args:
            position: [x, y, z] position to check

        Returns:
            True if clearance is sufficient
        """
        for obstacle in self.obstacles:
            obs_pos = obstacle.get_position()
            # For cylinders, check horizontal distance
            horizontal_dist = np.linalg.norm(position[:2] - obs_pos[:2])
            if horizontal_dist < (obstacle.radius + config.MIN_CLEARANCE):
                # Check if within height range
                if obs_pos[2] <= position[2] <= obs_pos[2] + obstacle.height:
                    return False

        for obstacle in self.dynamic_obstacles:
            obs_pos = obstacle.get_position()
            horizontal_dist = np.linalg.norm(position[:2] - obs_pos[:2])
            if horizontal_dist < (obstacle.radius + config.MIN_CLEARANCE):
                if obstacle.height > 0:
                    # Cylinder
                    if obs_pos[2] <= position[2] <= obs_pos[2] + obstacle.height:
                        return False
                else:
                    # Sphere
                    dist_3d = np.linalg.norm(position - obs_pos)
                    if dist_3d < (obstacle.radius + config.MIN_CLEARANCE):
                        return False

        return True
