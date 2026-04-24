"""
Pretrain obstacles library - diverse obstacle types for curriculum learning.
"""

import numpy as np
import pybullet as p


class CylinderObstacle:
    """Cylindrical obstacle (столб)."""

    def __init__(self, position, radius, height, physics_client):
        self.position = position
        self.radius = radius
        self.height = height
        self.client = physics_client
        self.dynamic = False

        # Create collision shape
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=height,
            physicsClientId=self.client
        )

        # Create visual shape
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=[0.6, 0.3, 0.1, 1.0],
            physicsClientId=self.client
        )

        # Create body
        self.body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[position[0], position[1], position[2] + height/2],
            physicsClientId=self.client
        )

    def get_position(self):
        return self.position

    def update(self, dt):
        pass  # Static

    def cleanup(self):
        if self.body_id is not None:
            p.removeBody(self.body_id, physicsClientId=self.client)
            self.body_id = None


class SphereObstacle:
    """Spherical obstacle with sinusoidal movement (птица)."""

    def __init__(self, position, radius, speed, amplitude, frequency, physics_client):
        self.initial_position = np.array(position)
        self.position = np.array(position)
        self.radius = radius
        self.speed = speed
        self.amplitude = amplitude
        self.frequency = frequency
        self.client = physics_client
        self.dynamic = True
        self.time = 0.0

        # Random movement direction
        angle = np.random.uniform(0, 2 * np.pi)
        self.direction = np.array([np.cos(angle), np.sin(angle), 0])

        # Create collision shape
        collision_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=radius,
            physicsClientId=self.client
        )

        # Create visual shape
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[0.8, 0.2, 0.2, 1.0],
            physicsClientId=self.client
        )

        # Create body
        self.body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=self.client
        )

    def get_position(self):
        return self.position

    def update(self, dt):
        """Sinusoidal movement."""
        self.time += dt

        # Sinusoidal offset
        offset = self.amplitude * np.sin(2 * np.pi * self.frequency * self.time)

        # Update position
        self.position = self.initial_position + self.direction * offset

        # Update PyBullet body
        p.resetBasePositionAndOrientation(
            self.body_id,
            self.position,
            [0, 0, 0, 1],
            physicsClientId=self.client
        )

    def cleanup(self):
        if self.body_id is not None:
            p.removeBody(self.body_id, physicsClientId=self.client)
            self.body_id = None


class WallObstacle:
    """Vertical wall obstacle."""

    def __init__(self, position, width, height, thickness, physics_client):
        self.position = position
        self.width = width
        self.height = height
        self.thickness = thickness
        self.client = physics_client
        self.dynamic = False

        # Create collision shape (box)
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[width/2, thickness/2, height/2],
            physicsClientId=self.client
        )

        # Create visual shape
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[width/2, thickness/2, height/2],
            rgbaColor=[0.5, 0.5, 0.5, 1.0],
            physicsClientId=self.client
        )

        # Create body
        self.body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[position[0], position[1], position[2] + height/2],
            physicsClientId=self.client
        )

        # For collision checking (treat as cylinder with large radius)
        self.radius = max(width, thickness) / 2

    def get_position(self):
        return self.position

    def update(self, dt):
        pass  # Static

    def cleanup(self):
        if self.body_id is not None:
            p.removeBody(self.body_id, physicsClientId=self.client)
            self.body_id = None


class BeamObstacle:
    """Horizontal beam obstacle that can swing around a chosen axis."""

    def __init__(
        self,
        position,
        length,
        height,
        thickness,
        swing_angle,
        swing_period,
        physics_client,
        swing_axis="yaw",
        phase=0.0
    ):
        self.position = np.array(position)
        self.length = length
        self.height = height
        self.thickness = thickness
        self.swing_angle = np.radians(swing_angle)
        self.swing_period = swing_period
        self.client = physics_client
        self.dynamic = swing_angle > 0
        self.time = 0.0
        self.swing_axis = swing_axis
        self.phase = phase

        # Create collision shape (box)
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[length/2, thickness/2, thickness/2],
            physicsClientId=self.client
        )

        # Create visual shape
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[length/2, thickness/2, thickness/2],
            rgbaColor=[0.4, 0.3, 0.2, 1.0],
            physicsClientId=self.client
        )

        # Create body
        self.body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[position[0], position[1], height],
            baseOrientation=self._orientation(),
            physicsClientId=self.client
        )

        # For collision checking
        self.radius = max(length, thickness) / 2

    def get_position(self):
        return self.position

    def _angle(self):
        return self.swing_angle * np.sin(2 * np.pi * self.time / self.swing_period + self.phase)

    def _orientation(self):
        angle = self._angle()
        if self.swing_axis == "pitch":
            return p.getQuaternionFromEuler([0, angle, 0])
        if self.swing_axis == "roll":
            return p.getQuaternionFromEuler([angle, 0, 0])
        return p.getQuaternionFromEuler([0, 0, angle])

    def update(self, dt):
        """Swinging motion."""
        if not self.dynamic:
            return

        self.time += dt

        # Update orientation
        p.resetBasePositionAndOrientation(
            self.body_id,
            [self.position[0], self.position[1], self.height],
            self._orientation(),
            physicsClientId=self.client
        )

    def cleanup(self):
        if self.body_id is not None:
            p.removeBody(self.body_id, physicsClientId=self.client)
            self.body_id = None


class BoxObstacle:
    """Box/cube obstacle."""

    def __init__(self, position, size, height, physics_client):
        self.position = position
        self.size = size
        self.height = height
        self.client = physics_client
        self.dynamic = False

        # Create collision shape
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[size/2, size/2, height/2],
            physicsClientId=self.client
        )

        # Create visual shape
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[size/2, size/2, height/2],
            rgbaColor=[0.3, 0.3, 0.6, 1.0],
            physicsClientId=self.client
        )

        # Create body
        self.body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[position[0], position[1], position[2] + height/2],
            physicsClientId=self.client
        )

        # For collision checking
        self.radius = size / 2

    def get_position(self):
        return self.position

    def update(self, dt):
        pass  # Static

    def cleanup(self):
        if self.body_id is not None:
            p.removeBody(self.body_id, physicsClientId=self.client)
            self.body_id = None


class SwingingStickObstacle:
    """Swinging stick obstacle (ветка)."""

    def __init__(
        self,
        position,
        length,
        thickness,
        swing_angle,
        swing_period,
        physics_client,
        vertical_swing=False,
        vertical_amplitude=0.15,
        phase=0.0
    ):
        self.initial_position = np.array(position)
        self.position = np.array(position)
        self.length = length
        self.thickness = thickness
        self.swing_angle = np.radians(swing_angle)
        self.swing_period = swing_period
        self.client = physics_client
        self.dynamic = True
        self.time = 0.0
        self.vertical_swing = vertical_swing
        self.phase = phase

        # For collision checking
        self.radius = length / 2
        self.height = 0  # Horizontal stick, no vertical extent for clearance check

        # Vertical movement parameters (like branches in wind)
        self.vertical_amplitude = vertical_amplitude  # From config
        self.vertical_frequency = 1.0 / swing_period  # Sync with swing

        # Create collision shape (capsule)
        collision_shape = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=thickness,
            height=length,
            physicsClientId=self.client
        )

        # Create visual shape
        visual_shape = p.createVisualShape(
            p.GEOM_CAPSULE,
            radius=thickness,
            length=length,
            rgbaColor=[0.4, 0.6, 0.3, 1.0],
            physicsClientId=self.client
        )

        initial_position, initial_quat = self._body_state()
        self.position = initial_position

        # Create body
        self.body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=initial_position,
            baseOrientation=initial_quat,
            physicsClientId=self.client
        )

    def get_position(self):
        return self.position

    def _phase_angle(self):
        return 2 * np.pi * self.time / self.swing_period + self.phase

    def _body_state(self):
        phase_angle = self._phase_angle()

        if self.vertical_swing:
            vertical_offset = self.vertical_amplitude * np.sin(phase_angle)
            new_position = self.initial_position + np.array([0, 0, vertical_offset])
            quat = p.getQuaternionFromEuler([0, np.pi/2, 0])
        else:
            angle = self.swing_angle * np.sin(phase_angle)
            new_position = self.initial_position
            quat = p.getQuaternionFromEuler([0, np.pi/2, angle])

        return new_position, quat

    def update(self, dt):
        """Swinging motion around anchor point with vertical movement."""
        self.time += dt

        new_position, quat = self._body_state()
        self.position = new_position

        p.resetBasePositionAndOrientation(
            self.body_id,
            new_position,
            quat,
            physicsClientId=self.client
        )

    def cleanup(self):
        if self.body_id is not None:
            p.removeBody(self.body_id, physicsClientId=self.client)
            self.body_id = None
