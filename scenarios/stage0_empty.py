"""
Stage 0: Empty scenario.
No obstacles, only start and goal positions.
Used for baseline verification.
"""

import numpy as np
from typing import Tuple
from scenarios.base_scenario import BaseScenario
import config


class Stage0Scenario(BaseScenario):
    """Empty scenario with no obstacles."""

    def __init__(self, seed: int = None):
        """Initialize empty scenario."""
        super().__init__(seed)

    def generate(self, physics_client: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate empty scenario with start and goal.

        Args:
            physics_client: PyBullet physics client ID

        Returns:
            start_pos: [x, y, z]
            goal_pos: [x, y, z]
        """
        # No obstacles to create
        return self._generate_start_goal()

    def reset(self, physics_client: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset scenario.

        Args:
            physics_client: PyBullet physics client ID

        Returns:
            start_pos: [x, y, z]
            goal_pos: [x, y, z]
        """
        # Clean up (nothing to clean in empty scenario)
        self.cleanup()

        # Generate new start/goal
        return self.generate(physics_client)
