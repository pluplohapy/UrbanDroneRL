"""
Raycast module for obstacle detection.
Implements 20-ray configuration: 8 horizontal + 4 at +30° + 4 at -30° + 1 up + 1 down + 2 diagonal up.
"""

import numpy as np
import pybullet as p
from typing import Iterable, Optional


class RaycastSensor:
    """20-ray sensor for obstacle detection."""

    def __init__(self, ray_length: float = 5.0):
        """
        Initialize raycast sensor.

        Args:
            ray_length: Maximum ray distance in meters
        """
        self.ray_length = ray_length
        self.n_rays = 20

        # Generate ray directions in body frame
        self.ray_directions = self._generate_ray_directions()

    def _generate_ray_directions(self) -> np.ndarray:
        """
        Generate 20 ray directions in body frame.

        Returns:
            Array of shape (20, 3) with normalized direction vectors
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

        # 1 ray straight up (90°)
        directions.append([0.0, 0.0, 1.0])

        # 1 ray straight down (-90°)
        directions.append([0.0, 0.0, -1.0])

        # 2 additional rays at +60° elevation (forward and backward)
        elevation_steep = np.radians(60)
        for angle in [0, np.pi]:  # Forward and backward
            dx = np.cos(angle) * np.cos(elevation_steep)
            dy = np.sin(angle) * np.cos(elevation_steep)
            dz = np.sin(elevation_steep)
            directions.append([dx, dy, dz])

        return np.array(directions)

    def cast_rays(
        self,
        drone_pos: np.ndarray,
        drone_orn: np.ndarray,
        physics_client: int,
        ignore_body_ids: Optional[Iterable[int]] = None
    ) -> np.ndarray:
        """
        Cast rays from drone position and return normalized distances.

        Args:
            drone_pos: Drone position [x, y, z]
            drone_orn: Drone orientation quaternion [x, y, z, w]
            physics_client: PyBullet physics client ID
            ignore_body_ids: Optional body ids to ignore (e.g. other drones)

        Returns:
            Array of shape (20,) with normalized distances [0, 1]
            where 0 = max distance (no hit), 1 = very close
        """
        ignored = {int(body_id) for body_id in ignore_body_ids} if ignore_body_ids is not None else set()

        # Convert quaternion to rotation matrix
        rot_matrix = np.array(p.getMatrixFromQuaternion(drone_orn)).reshape(3, 3)

        # Transform ray directions from body frame to world frame
        world_directions = (rot_matrix @ self.ray_directions.T).T

        # Cast all rays
        ray_results = []
        for direction in world_directions:
            # If ignored bodies are hit (e.g. other drones), continue the ray further.
            # This keeps obstacle sensing independent from peer drones in swarm mode.
            ray_from = np.array(drone_pos, dtype=float)
            remaining = float(self.ray_length)
            traveled = 0.0
            distance = self.ray_length

            for _ in range(8):
                ray_to = ray_from + direction * remaining
                result = p.rayTest(ray_from, ray_to, physicsClientId=physics_client)[0]
                hit_body_id = int(result[0])
                hit_fraction = float(result[2])

                if hit_body_id < 0:
                    distance = self.ray_length
                    break

                hit_distance = max(0.0, hit_fraction) * remaining
                abs_hit_distance = min(self.ray_length, traveled + hit_distance)

                if hit_body_id not in ignored:
                    distance = abs_hit_distance
                    break

                advance = min(remaining, hit_distance + 1e-3)
                traveled += advance
                remaining -= advance
                if remaining <= 1e-6:
                    distance = self.ray_length
                    break
                ray_from = np.array(drone_pos, dtype=float) + direction * traveled

            ray_results.append(distance)

        # Normalize distances to [0, 1] where 1 = far (safe), 0 = close (danger)
        # This is more intuitive: high values = safe, low values = danger
        normalized = np.array(ray_results) / self.ray_length

        return normalized
