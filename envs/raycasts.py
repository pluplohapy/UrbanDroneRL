"""
Raycast module for obstacle detection.
Implements 16-ray configuration: 8 horizontal + 4 at +30° + 4 at -30°.
"""

import numpy as np
import pybullet as p
from typing import List, Tuple


class RaycastSensor:
    """16-ray sensor for obstacle detection."""

    def __init__(self, ray_length: float = 5.0):
        """
        Initialize raycast sensor.

        Args:
            ray_length: Maximum ray distance in meters
        """
        self.ray_length = ray_length
        self.n_rays = 16

        # Generate ray directions in body frame
        self.ray_directions = self._generate_ray_directions()

    def _generate_ray_directions(self) -> np.ndarray:
        """
        Generate 16 ray directions in body frame.

        Returns:
            Array of shape (16, 3) with normalized direction vectors
        """
        directions = []

        # 8 horizontal rays (elevation = 0°)
        for i in range(8):
            angle = i * (2 * np.pi / 8)
            dx = np.cos(angle)
            dy = np.sin(angle)
            dz = 0.0
            directions.append([dx, dy, dz])

        # 4 rays at +30° elevation
        elevation_up = np.radians(30)
        for i in range(4):
            angle = i * (2 * np.pi / 4)
            dx = np.cos(angle) * np.cos(elevation_up)
            dy = np.sin(angle) * np.cos(elevation_up)
            dz = np.sin(elevation_up)
            directions.append([dx, dy, dz])

        # 4 rays at -30° elevation
        elevation_down = np.radians(-30)
        for i in range(4):
            angle = i * (2 * np.pi / 4)
            dx = np.cos(angle) * np.cos(elevation_down)
            dy = np.sin(angle) * np.cos(elevation_down)
            dz = np.sin(elevation_down)
            directions.append([dx, dy, dz])

        return np.array(directions)

    def cast_rays(self, drone_pos: np.ndarray, drone_orn: np.ndarray,
                  physics_client: int) -> np.ndarray:
        """
        Cast rays from drone position and return normalized distances.

        Args:
            drone_pos: Drone position [x, y, z]
            drone_orn: Drone orientation quaternion [x, y, z, w]
            physics_client: PyBullet physics client ID

        Returns:
            Array of shape (16,) with normalized distances [0, 1]
            where 0 = max distance (no hit), 1 = very close
        """
        # Convert quaternion to rotation matrix
        rot_matrix = np.array(p.getMatrixFromQuaternion(drone_orn)).reshape(3, 3)

        # Transform ray directions from body frame to world frame
        world_directions = (rot_matrix @ self.ray_directions.T).T

        # Cast all rays
        ray_results = []
        for direction in world_directions:
            ray_from = drone_pos
            ray_to = drone_pos + direction * self.ray_length

            result = p.rayTest(ray_from, ray_to, physicsClientId=physics_client)

            # result is a list with one tuple: (objectUniqueId, linkIndex, hit_fraction, hit_position, hit_normal)
            hit_fraction = result[0][2]

            # hit_fraction is in [0, 1] where 1 = no hit
            # Convert to distance
            distance = hit_fraction * self.ray_length
            ray_results.append(distance)

        # Normalize distances to [0, 1] where 0 = far, 1 = close
        # This makes it easier for the network to learn
        normalized = np.array(ray_results) / self.ray_length

        return normalized
