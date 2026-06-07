import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List
import config


class BaseScenario(ABC):

    def __init__(self, seed: int = None):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.obstacles = []
        self.dynamic_obstacles = []

    @abstractmethod
    def generate(self, physics_client: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def reset(self, physics_client: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def update_dynamic_obstacles(self, dt: float):
        for obstacle in self.dynamic_obstacles:
            obstacle.update(dt)

    def cleanup(self):
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
        max_attempts = 1000

        for _ in range(max_attempts):
            start_x = self.rng.uniform(-config.ARENA_SIZE_X / 2 + 1, config.ARENA_SIZE_X / 2 - 1)
            start_y = self.rng.uniform(-config.ARENA_SIZE_Y / 2 + 1, config.ARENA_SIZE_Y / 2 - 1)
            start_z = self.rng.uniform(0.5, config.ARENA_HEIGHT / 2)

            goal_x = self.rng.uniform(-config.ARENA_SIZE_X / 2 + 1, config.ARENA_SIZE_X / 2 - 1)
            goal_y = self.rng.uniform(-config.ARENA_SIZE_Y / 2 + 1, config.ARENA_SIZE_Y / 2 - 1)
            goal_z = self.rng.uniform(config.ARENA_HEIGHT / 2, config.ARENA_HEIGHT - 0.5)

            start_pos = np.array([start_x, start_y, start_z])
            goal_pos = np.array([goal_x, goal_y, goal_z])

            dist = np.linalg.norm(goal_pos - start_pos)
            if dist < config.MIN_START_GOAL_DIST:
                continue

            if not self._check_clearance(start_pos) or not self._check_clearance(goal_pos):
                continue

            return start_pos, goal_pos

        start_pos = np.array([0.0, 0.0, 1.0])
        goal_pos = np.array([0.0, 0.0, 4.0])
        return start_pos, goal_pos

    def _check_clearance(self, position: np.ndarray) -> bool:
        for obstacle in self.obstacles:
            obs_pos = obstacle.get_position()
            horizontal_dist = np.linalg.norm(position[:2] - obs_pos[:2])
            if horizontal_dist < (obstacle.radius + config.MIN_CLEARANCE):
                if obs_pos[2] <= position[2] <= obs_pos[2] + obstacle.height:
                    return False

        for obstacle in self.dynamic_obstacles:
            obs_pos = obstacle.get_position()
            horizontal_dist = np.linalg.norm(position[:2] - obs_pos[:2])
            if horizontal_dist < (obstacle.radius + config.MIN_CLEARANCE):
                if obstacle.height > 0:
                    if obs_pos[2] <= position[2] <= obs_pos[2] + obstacle.height:
                        return False
                else:
                    dist_3d = np.linalg.norm(position - obs_pos)
                    if dist_3d < (obstacle.radius + config.MIN_CLEARANCE):
                        return False

        return True