import numpy as np
import pybullet as p
from typing import Tuple, Optional

class StaticObstacle:

    def __init__(self, position: np.ndarray, radius: float, height: float,
                 physics_client: int):
        self.position = position
        self.radius = radius
        self.height = height
        self.physics_client = physics_client

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
            rgbaColor=[0.7, 0.7, 0.7, 1.0],
            physicsClientId=physics_client
        )

        center_pos = position.copy()
        center_pos[2] += height / 2

        self.body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=center_pos,
            physicsClientId=physics_client
        )

    def get_position(self) -> np.ndarray:
        return self.position

    def remove(self):
        p.removeBody(self.body_id, physicsClientId=self.physics_client)

class DynamicObstacle:

    def __init__(self, initial_position: np.ndarray, radius: float, height: float,
                 amplitude: float, frequency: float, phase: float,
                 axis: str, physics_client: int):
        self.initial_position = initial_position.copy()
        self.radius = radius
        self.height = height
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.axis = axis
        self.physics_client = physics_client
        self.time = 0.0

        if height > 0:
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

        self.body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=initial_position,
            physicsClientId=physics_client
        )

    def update(self, dt: float):
        self.time += dt

        offset = self.amplitude * np.sin(2 * np.pi * self.frequency * self.time + self.phase)

        new_position = self.initial_position.copy()
        if self.axis == 'x':
            new_position[0] += offset
        elif self.axis == 'y':
            new_position[1] += offset
        elif self.axis == 'z':
            new_position[2] += offset

        p.resetBasePositionAndOrientation(
            self.body_id,
            new_position,
            [0, 0, 0, 1],
            physicsClientId=self.physics_client
        )

    def get_position(self) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.physics_client)
        return np.array(pos)

    def remove(self):
        p.removeBody(self.body_id, physicsClientId=self.physics_client)
