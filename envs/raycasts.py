import numpy as np
import pybullet as p
from typing import Iterable, Optional

class RaycastSensor:

    def __init__(self, ray_length: float = 5.0):
        self.ray_length = ray_length
        self.n_rays = 20

        self.ray_directions = self._generate_ray_directions()

    def _generate_ray_directions(self) -> np.ndarray:
        directions = []

        for i in range(8):
            angle = i * (2 * np.pi / 8)
            dx = np.cos(angle)
            dy = np.sin(angle)
            dz = 0.0
            directions.append([dx, dy, dz])

        elevation_up = np.radians(30)
        for i in range(4):
            angle = i * (2 * np.pi / 4)
            dx = np.cos(angle) * np.cos(elevation_up)
            dy = np.sin(angle) * np.cos(elevation_up)
            dz = np.sin(elevation_up)
            directions.append([dx, dy, dz])

        elevation_down = np.radians(-30)
        for i in range(4):
            angle = i * (2 * np.pi / 4)
            dx = np.cos(angle) * np.cos(elevation_down)
            dy = np.sin(angle) * np.cos(elevation_down)
            dz = np.sin(elevation_down)
            directions.append([dx, dy, dz])

        directions.append([0.0, 0.0, 1.0])

        directions.append([0.0, 0.0, -1.0])

        elevation_steep = np.radians(60)
        for angle in [0, np.pi]:
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
        ignored = {int(body_id) for body_id in ignore_body_ids} if ignore_body_ids is not None else set()

        rot_matrix = np.array(p.getMatrixFromQuaternion(drone_orn)).reshape(3, 3)

        world_directions = (rot_matrix @ self.ray_directions.T).T

        ray_results = []
        for direction in world_directions:
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

        normalized = np.array(ray_results) / self.ray_length

        return normalized
