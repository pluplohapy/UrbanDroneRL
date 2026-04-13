"""
Obstacle classes for scenarios.
Includes static cylinders and dynamic objects with sinusoidal motion.
"""

import numpy as np
import pybullet as p
from typing import Tuple, Optional


class StaticObstacle:
    """Static cylindrical obstacle."""

    def __init__(self, position: np.ndarray, radius: float, height: float,
                 physics_client: int):
        """
        Create a static cylinder obstacle.

        Args:
            position: [x, y, z] position (z is bottom of cylinder)
            radius: Cylinder radius in meters
            height: Cylinder height in meters
            physics_client: PyBullet physics client ID
        """
        self.position = position
        self.radius = radius
        self.height = height
        self.physics_client = physics_client

        # Create collision shape
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            height=height,
            physicsClientId=physics_client
        )

        # Create visual shape
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=[0.7, 0.7, 0.7, 1.0],
            physicsClientId=physics_client
        )

        # Create multibody (position z is at center of cylinder)
        center_pos = position.copy()
        center_pos[2] += height / 2

        self.body_id = p.createMultiBody(
            baseMass=0,  # Static object
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=center_pos,
            physicsClientId=physics_client
        )

    def get_position(self) -> np.ndarray:
        """Get current position."""
        return self.position

    def remove(self):
        """Remove obstacle from simulation."""
        p.removeBody(self.body_id, physicsClientId=self.physics_client)


class DynamicObstacle:
    """Dynamic obstacle with sinusoidal motion."""

    def __init__(self, initial_position: np.ndarray, radius: float, height: float,
                 amplitude: float, frequency: float, phase: float,
                 axis: str, physics_client: int):
        """
        Create a dynamic obstacle with sinusoidal motion.

        Args:
            initial_position: [x, y, z] initial position
            radius: Obstacle radius in meters
            height: Obstacle height in meters (for cylinder) or 0 for sphere
            amplitude: Oscillation amplitude in meters
            frequency: Oscillation frequency in Hz
            phase: Phase offset in radians
            axis: Oscillation axis ('x', 'y', or 'z')
            physics_client: PyBullet physics client ID
        """
        self.initial_position = initial_position.copy()
        self.radius = radius
        self.height = height
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.axis = axis
        self.physics_client = physics_client
        self.time = 0.0

        # Determine shape type
        if height > 0:
            # Cylinder
            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER,
                radius=radius,
                height=height,
                physicsClientId=physics_client
            )
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=radius,
                length=height,
                rgbaColor=[0.9, 0.3, 0.3, 1.0],
                physicsClientId=physics_client
            )
        else:
            # Sphere
            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_SPHERE,
                radius=radius,
                physicsClientId=physics_client
            )
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=radius,
                rgbaColor=[0.9, 0.3, 0.3, 1.0],
                physicsClientId=physics_client
            )

        # Create multibody
        self.body_id = p.createMultiBody(
            baseMass=0,  # Kinematic object
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=initial_position,
            physicsClientId=physics_client
        )

    def update(self, dt: float):
        """
        Update obstacle position based on sinusoidal motion.

        Args:
            dt: Time step in seconds
        """
        self.time += dt

        # Calculate offset
        offset = self.amplitude * np.sin(2 * np.pi * self.frequency * self.time + self.phase)

        # Apply offset to appropriate axis
        new_position = self.initial_position.copy()
        if self.axis == 'x':
            new_position[0] += offset
        elif self.axis == 'y':
            new_position[1] += offset
        elif self.axis == 'z':
            new_position[2] += offset

        # Update position in simulation
        p.resetBasePositionAndOrientation(
            self.body_id,
            new_position,
            [0, 0, 0, 1],
            physicsClientId=self.physics_client
        )

    def get_position(self) -> np.ndarray:
        """Get current position."""
        pos, _ = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.physics_client)
        return np.array(pos)

    def remove(self):
        """Remove obstacle from simulation."""
        p.removeBody(self.body_id, physicsClientId=self.physics_client)
