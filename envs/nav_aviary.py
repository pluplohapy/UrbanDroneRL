"""
Navigation environment for RL drone training.
Inherits from BaseRLAviary and implements velocity control.
"""

import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from envs.raycasts import RaycastSensor
import config


class NavAviary(BaseRLAviary):
    """Navigation environment with velocity control."""

    def __init__(self, scenario, gui: bool = False, physics=Physics.PYB):
        """
        Initialize navigation environment.

        Args:
            scenario: Scenario object providing obstacles and start/goal
            gui: Whether to show PyBullet GUI
            physics: Physics engine (must be Physics.PYB)
        """
        self.scenario = scenario
        self.start_pos = None
        self.goal_pos = None
        self.prev_dist_to_goal = None
        self.control_step_counter = 0  # Renamed to avoid conflict with BaseRLAviary
        self.prev_action = np.zeros(4)
        self.visited_cells = set()  # For exploration bonus

        # Initialize raycast sensor
        self.raycast_sensor = RaycastSensor(ray_length=config.RAY_LENGTH)

        # Call parent constructor
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            neighbourhood_radius=np.inf,
            physics=physics,
            pyb_freq=240,
            ctrl_freq=30,
            gui=gui,
            record=False,
            obs=ObservationType.KIN,
            act=ActionType.VEL
        )

    def _preprocessAction(self, action):
        """
        Override to properly handle velocity control.
        BaseRLAviary's VEL mode has SPEED_LIMIT=0.25 m/s which is too slow.
        We implement our own velocity control here.

        Args:
            action: (1, 4) array with [vx, vy, vz, yaw_rate] normalized to [-1, 1]
                    Actions are in BODY FRAME (forward/right/up relative to drone)

        Returns:
            (1, 4) array of RPMs for the 4 motors
        """
        rpm = np.zeros((self.NUM_DRONES, 4))

        for k in range(action.shape[0]):
            state = self._getDroneStateVector(k)
            cur_quat = state[3:7]

            # Convert normalized action to actual velocities in BODY frame
            target_vel_body = np.array([
                action[k, 0] * config.VX_MAX,
                action[k, 1] * config.VY_MAX,
                action[k, 2] * config.VZ_MAX
            ])

            # Transform velocity from body frame to world frame
            rot_matrix = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
            target_vel_world = rot_matrix @ target_vel_body

            # Target yaw rate (integrate to get target yaw)
            target_yaw_rate = action[k, 3] * config.YAW_RATE_MAX
            target_yaw = state[9] + target_yaw_rate * self.CTRL_TIMESTEP

            # Compute target position in world frame: current + velocity * dt
            # Use a larger multiplier to make the target further ahead
            # This encourages the PID to track velocity better
            target_pos = state[0:3] + target_vel_world * self.CTRL_TIMESTEP * 3.0

            # Use PID controller to compute RPMs
            rpm_k, _, _ = self.ctrl[k].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=target_pos,
                target_rpy=np.array([0, 0, target_yaw])
            )
            rpm[k, :] = rpm_k

        return rpm

    def _actionSpace(self):
        """
        Define action space: velocity control.

        Returns:
            Box space for [vx, vy, vz, yaw_rate] normalized to [-1, 1]
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _observationSpace(self):
        """
        Define observation space: 28 features (вернул к 28).
        3 (goal) + 1 (dist) + 3 (vel) + 1 (height) + 4 (prev_action) + 16 (raycasts) = 28

        Returns:
            Box space for 28-dimensional observation
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(28,), dtype=np.float32)

    def _computeObs(self):
        """
        Compute observation vector (28 features).
        3 (goal) + 1 (dist) + 3 (vel) + 1 (height) + 4 (prev_action) + 16 (raycasts) = 28

        Returns:
            np.ndarray of shape (28,)
        """
        # Get drone state
        drone_pos = self._getDroneStateVector(0)[:3]
        drone_quat = self._getDroneStateVector(0)[3:7]
        drone_vel = self._getDroneStateVector(0)[10:13]

        # 1. Goal in body frame (3)
        goal_world = self.goal_pos - drone_pos
        rot_matrix = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)
        goal_body = rot_matrix.T @ goal_world

        # Normalize goal vector
        max_dist = np.sqrt(config.ARENA_SIZE_X**2 + config.ARENA_SIZE_Y**2 + config.ARENA_HEIGHT**2)
        goal_body_norm = goal_body / max_dist

        # 2. Distance to goal (1) - НОВОЕ!
        dist_to_goal = np.linalg.norm(goal_world)
        dist_to_goal_norm = np.clip(dist_to_goal / max_dist, 0, 1)

        # 3. Linear velocity in body frame, normalized (3)
        vel_body = rot_matrix.T @ drone_vel
        vel_norm = np.clip(vel_body / np.array([config.VX_MAX, config.VY_MAX, config.VZ_MAX]), -1, 1)

        # 4. Normalized height (1)
        height_norm = drone_pos[2] / config.ARENA_HEIGHT

        # 5. Previous action (4)
        prev_action = self.prev_action

        # 6. Raycasts (16)
        raycasts = self.raycast_sensor.cast_rays(drone_pos, drone_quat, self.CLIENT)

        # Concatenate all features
        obs = np.concatenate([
            goal_body_norm,        # 3
            [dist_to_goal_norm],   # 1 - НОВОЕ!
            vel_norm,              # 3
            [height_norm],         # 1
            prev_action,           # 4
            raycasts               # 16
        ])

        return obs.astype(np.float32)

    def _computeReward(self):
        """
        Compute reward based on progress and safety.

        Returns:
            float reward value
        """
        drone_pos = self._getDroneStateVector(0)[:3]
        drone_quat = self._getDroneStateVector(0)[3:7]
        drone_vel = self._getDroneStateVector(0)[10:13]
        curr_dist = np.linalg.norm(self.goal_pos - drone_pos)

        # Progress reward
        progress = self.prev_dist_to_goal - curr_dist
        reward = config.REWARD_PROGRESS_SCALE * progress

        # Velocity reward - награда за полёт в направлении цели
        goal_world = self.goal_pos - drone_pos
        goal_direction = goal_world / (np.linalg.norm(goal_world) + 1e-6)
        velocity_towards_goal = np.dot(drone_vel, goal_direction)
        velocity_reward = 0.2 * max(0, velocity_towards_goal)
        reward += velocity_reward

        # Proximity bonus - УСИЛЕННАЯ награда за близость к цели
        proximity_bonus = 15.0 * np.exp(-curr_dist)  # Увеличено с 2.0
        reward += proximity_bonus

        # Update previous distance
        self.prev_dist_to_goal = curr_dist

        # Exploration bonus - награда за посещение новых клеток
        grid_x = int((drone_pos[0] + config.ARENA_SIZE_X / 2) / config.EXPLORATION_GRID_SIZE)
        grid_y = int((drone_pos[1] + config.ARENA_SIZE_Y / 2) / config.EXPLORATION_GRID_SIZE)
        grid_z = int(drone_pos[2] / config.EXPLORATION_GRID_SIZE)
        grid_key = (grid_x, grid_y, grid_z)

        if grid_key not in self.visited_cells:
            self.visited_cells.add(grid_key)
            reward += config.REWARD_EXPLORATION_BONUS

        # Proximity penalty based on raycasts - УСИЛЕННЫЙ штраф
        raycasts = self.raycast_sensor.cast_rays(drone_pos, drone_quat, self.CLIENT)
        min_ray = np.min(raycasts)

        # Convert normalized ray to actual distance
        min_dist = min_ray * config.RAY_LENGTH

        if min_dist < config.REWARD_PROXIMITY_THRESHOLD:
            penalty = config.REWARD_PROXIMITY_SCALE * (config.REWARD_PROXIMITY_THRESHOLD - min_dist)
            reward -= penalty

        # Step penalty
        reward -= config.REWARD_STEP_PENALTY

        return reward

    def _computeTerminated(self):
        """
        Check if episode should terminate (success or crash).

        Returns:
            bool indicating termination
        """
        drone_pos = self._getDroneStateVector(0)[:3]

        # Success: reached goal
        dist_to_goal = np.linalg.norm(self.goal_pos - drone_pos)
        if dist_to_goal < config.SUCCESS_DIST:
            return True

        # Crash: collision detected
        contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[0], physicsClientId=self.CLIENT)
        if len(contact_points) > 0:
            return True

        return False

    def _computeTruncated(self):
        """
        Check if episode should be truncated (timeout or out of bounds).

        Returns:
            bool indicating truncation
        """
        # Timeout
        if self.control_step_counter >= config.MAX_STEPS:
            return True

        # Out of bounds
        drone_pos = self._getDroneStateVector(0)[:3]
        if (abs(drone_pos[0]) > config.ARENA_SIZE_X / 2 or
            abs(drone_pos[1]) > config.ARENA_SIZE_Y / 2 or
            drone_pos[2] < 0.1 or
            drone_pos[2] > config.ARENA_HEIGHT):
            return True

        return False

    def _computeInfo(self):
        """
        Compute info dictionary.

        Returns:
            dict with episode information
        """
        drone_pos = self._getDroneStateVector(0)[:3]
        drone_quat = self._getDroneStateVector(0)[3:7]
        dist_to_goal = np.linalg.norm(self.goal_pos - drone_pos)

        # Check success and crash
        is_success = dist_to_goal < config.SUCCESS_DIST
        contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[0], physicsClientId=self.CLIENT)
        is_crash = len(contact_points) > 0

        # Get min ray distance
        raycasts = self.raycast_sensor.cast_rays(drone_pos, drone_quat, self.CLIENT)
        min_ray_dist = np.min(raycasts) * config.RAY_LENGTH

        return {
            "is_success": is_success,
            "is_crash": is_crash,
            "dist_to_goal": dist_to_goal,
            "min_ray_dist": min_ray_dist,
            "step": self.control_step_counter
        }

    def reset(self, seed=None, options=None):
        """
        Reset environment.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            observation, info
        """
        # Generate start/goal BEFORE creating obstacles
        self.start_pos, self.goal_pos = self.scenario._generate_start_goal()

        # Set initial position
        self.INIT_XYZS = np.array([self.start_pos])
        self.INIT_RPYS = np.array([[0, 0, 0]])

        # Call parent reset (this calls p.resetSimulation())
        obs, info = super().reset(seed=seed, options=options)

        # NOW create obstacles AFTER resetSimulation
        self.scenario.reset(self.CLIENT)

        # Reset internal state
        self.control_step_counter = 0
        self.prev_dist_to_goal = np.linalg.norm(self.goal_pos - self.start_pos)
        self.prev_action = np.zeros(4)
        self.visited_cells = set()  # Reset exploration tracking

        return obs, info

    def step(self, action):
        """
        Execute one step.

        Args:
            action: Action vector [vx, vy, vz, yaw_rate] normalized to [-1, 1]

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Store action
        self.prev_action = action.copy()

        # Update dynamic obstacles
        dt = 1.0 / self.CTRL_FREQ
        self.scenario.update_dynamic_obstacles(dt)

        # Convert normalized action to actual velocities
        vx = action[0] * config.VX_MAX
        vy = action[1] * config.VY_MAX
        vz = action[2] * config.VZ_MAX
        yaw_rate = action[3] * config.YAW_RATE_MAX

        # Execute action through parent class
        obs, reward, terminated, truncated, info = super().step(
            np.array([[vx, vy, vz, yaw_rate]])
        )

        # Increment control step counter
        self.control_step_counter += 1

        # Add terminal rewards
        if terminated:
            if info["is_success"]:
                reward += config.REWARD_SUCCESS
            elif info["is_crash"]:
                reward += config.REWARD_CRASH

        # Add timeout penalty if episode ends without success
        if truncated and not terminated:
            reward -= 50.0  # Штраф за timeout без достижения цели

        return obs, reward, terminated, truncated, info
