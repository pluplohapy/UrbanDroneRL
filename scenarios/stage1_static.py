"""
Stage 1: Static obstacles scenario.
Adds cylindrical obstacles that drone must navigate around.
"""

import numpy as np
import pybullet as p
from scenarios.base_scenario import BaseScenario
from envs.obstacles import StaticObstacle
import config


class Stage1Scenario(BaseScenario):
    """Stage 1 with static cylindrical obstacles."""

    def __init__(self, seed=None):
        """
        Initialize Stage 1 scenario.

        Args:
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.n_obstacles = 0
        self.client_id = None

    def generate(self, client_id):
        """
        Generate scenario: create obstacles and return start/goal.
        Same as reset() for Stage 1.

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
        # Store client_id
        self.client_id = client_id

        # Clear obstacles list (p.resetSimulation() already removed them)
        self.obstacles = []

        # Generate start and goal positions
        start_pos, goal_pos = self._generate_start_goal()

        # Generate random number of obstacles
        self.n_obstacles = self.rng.randint(
            config.STAGE1_N_OBSTACLES[0],
            config.STAGE1_N_OBSTACLES[1] + 1
        )

        # Create obstacles
        # print(f"[Stage1] Creating {self.n_obstacles} obstacles...")
        for i in range(self.n_obstacles):
            # Random radius and position
            radius = self.rng.uniform(
                config.STAGE1_RADIUS[0],
                config.STAGE1_RADIUS[1]
            )

            # Try to find valid position (not blocking start/goal)
            max_attempts = 50
            for attempt in range(max_attempts):
                x = self.rng.uniform(-config.ARENA_SIZE_X/2 + 1, config.ARENA_SIZE_X/2 - 1)
                y = self.rng.uniform(-config.ARENA_SIZE_Y/2 + 1, config.ARENA_SIZE_Y/2 - 1)
                pos = np.array([x, y, 0.0])  # z=0 is bottom of cylinder

                # Check clearance from start and goal
                dist_to_start = np.linalg.norm(pos[:2] - start_pos[:2])
                dist_to_goal = np.linalg.norm(pos[:2] - goal_pos[:2])

                if (dist_to_start > config.MIN_CLEARANCE + radius and
                    dist_to_goal > config.MIN_CLEARANCE + radius):
                    # Valid position found
                    obstacle = StaticObstacle(
                        position=pos,
                        radius=radius,
                        height=config.ARENA_HEIGHT,  # Full height cylinder
                        physics_client=client_id
                    )
                    self.obstacles.append(obstacle)
                    # print(f"[Stage1]   Obstacle {i+1}: pos=[{pos[0]:.2f}, {pos[1]:.2f}], radius={radius:.2f}, body_id={obstacle.body_id}")
                    break

        return start_pos, goal_pos

    def update_dynamic_obstacles(self, dt):
        """
        Update dynamic obstacles (none in Stage 1).

        Args:
            dt: Time step
        """
        pass  # No dynamic obstacles in Stage 1

    def get_obstacles(self):
        """
        Get list of obstacles.

        Returns:
            List of obstacle objects
        """
        return self.obstacles
