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
        self.prev_prev_action = np.zeros(4)  # Для smoothness penalty
        self.visited_cells = set()  # For exploration bonus

        # Debug tracking
        self.episode_trajectory = []  # List of positions
        self.episode_actions = []  # List of actions
        self.episode_heading_errors = []  # List of heading errors
        self.episode_speeds = []  # List of speeds
        self.episode_min_obstacle_dist = float('inf')  # Closest obstacle distance
        self.episode_near_misses = 0  # Count of near misses
        self.episode_min_goal_dist = float('inf')  # Closest to goal
        self.reward_components = {}  # Dict of reward components

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
        Define observation space: 29 features.
        3 (goal) + 1 (dist) + 3 (vel) + 1 (height) + 1 (yaw) + 4 (prev_action) + 16 (raycasts) = 29

        Returns:
            Box space for 29-dimensional observation
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(29,), dtype=np.float32)

    def _computeObs(self):
        """
        Compute observation vector (29 features).
        3 (goal) + 1 (dist) + 3 (vel) + 1 (height) + 1 (yaw) + 4 (prev_action) + 16 (raycasts) = 29

        Returns:
            np.ndarray of shape (29,)
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

        # 2. Distance to goal (1)
        dist_to_goal = np.linalg.norm(goal_world)
        dist_to_goal_norm = np.clip(dist_to_goal / max_dist, 0, 1)

        # 3. Linear velocity in body frame, normalized (3)
        vel_body = rot_matrix.T @ drone_vel
        vel_norm = np.clip(vel_body / np.array([config.VX_MAX, config.VY_MAX, config.VZ_MAX]), -1, 1)

        # 4. Normalized height (1)
        height_norm = drone_pos[2] / config.ARENA_HEIGHT

        # 5. Yaw angle (1) - НОВОЕ!
        # Extract yaw from quaternion
        qx, qy, qz, qw = drone_quat
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
        # Normalize to [-1, 1] (yaw is in [-pi, pi])
        yaw_norm = yaw / np.pi

        # 6. Previous action (4)
        prev_action = self.prev_action

        # 7. Raycasts (16)
        raycasts = self.raycast_sensor.cast_rays(drone_pos, drone_quat, self.CLIENT)

        # Concatenate all features
        obs = np.concatenate([
            goal_body_norm,        # 3
            [dist_to_goal_norm],   # 1
            vel_norm,              # 3
            [height_norm],         # 1
            [yaw_norm],            # 1 - НОВОЕ!
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

        # Initialize reward components dict
        if config.DEBUG_MODE and config.LOG_REWARD_COMPONENTS:
            self.reward_components = {}

        # Progress reward
        progress = self.prev_dist_to_goal - curr_dist
        reward_progress = config.REWARD_PROGRESS_SCALE * progress
        reward = reward_progress
        if config.DEBUG_MODE and config.LOG_REWARD_COMPONENTS:
            self.reward_components['progress'] = reward_progress

        # Velocity reward - награда за полёт в направлении цели
        goal_world = self.goal_pos - drone_pos
        goal_direction = goal_world / (np.linalg.norm(goal_world) + 1e-6)
        velocity_towards_goal = np.dot(drone_vel, goal_direction)
        reward_velocity = config.REWARD_VELOCITY_SCALE * max(0, velocity_towards_goal)  # Увеличено с 1.0 до 2.0
        reward += reward_velocity
        if config.DEBUG_MODE and config.LOG_REWARD_COMPONENTS:
            self.reward_components['velocity'] = reward_velocity

        # Yaw penalty - штраф за избыточное вращение
        # Получаем текущее действие yaw_rate из prev_action (нормализованное [-1, 1])
        yaw_action = abs(self.prev_action[3]) if len(self.prev_action) > 3 else 0
        reward_yaw_penalty = -config.REWARD_YAW_PENALTY_SCALE * yaw_action
        reward += reward_yaw_penalty
        if config.DEBUG_MODE and config.LOG_REWARD_COMPONENTS:
            self.reward_components['yaw_penalty'] = reward_yaw_penalty

        # Heading reward - награда за правильное направление к цели
        speed = np.linalg.norm(drone_vel)
        if speed > 0.1:  # Только если дрон движется
            vel_direction = drone_vel / speed
            heading_alignment = np.dot(vel_direction, goal_direction)  # cos угла между скоростью и направлением к цели
            reward_heading = config.REWARD_HEADING_SCALE * max(0, heading_alignment)
            reward += reward_heading
            if config.DEBUG_MODE and config.LOG_REWARD_COMPONENTS:
                self.reward_components['heading'] = reward_heading
        elif config.DEBUG_MODE and config.LOG_REWARD_COMPONENTS:
            self.reward_components['heading'] = 0.0

        # Action smoothness penalty - штраф за резкие изменения действий
        if hasattr(self, 'prev_prev_action') and len(self.prev_prev_action) > 0:
            action_change = np.linalg.norm(self.prev_action - self.prev_prev_action)
            reward_smoothness = -config.REWARD_ACTION_SMOOTHNESS_SCALE * action_change
            reward += reward_smoothness
            if config.DEBUG_MODE and config.LOG_REWARD_COMPONENTS:
                self.reward_components['smoothness'] = reward_smoothness
        elif config.DEBUG_MODE and config.LOG_REWARD_COMPONENTS:
            self.reward_components['smoothness'] = 0.0

        # Proximity bonus - УСИЛЕННАЯ награда за близость к цели
        reward_proximity = 15.0 * np.exp(-curr_dist)
        reward += reward_proximity
        if config.DEBUG_MODE and config.LOG_REWARD_COMPONENTS:
            self.reward_components['proximity'] = reward_proximity

        # Update previous distance
        self.prev_dist_to_goal = curr_dist

        # Exploration bonus - награда за посещение новых клеток
        grid_x = int((drone_pos[0] + config.ARENA_SIZE_X / 2) / config.EXPLORATION_GRID_SIZE)
        grid_y = int((drone_pos[1] + config.ARENA_SIZE_Y / 2) / config.EXPLORATION_GRID_SIZE)
        grid_z = int(drone_pos[2] / config.EXPLORATION_GRID_SIZE)
        grid_key = (grid_x, grid_y, grid_z)

        reward_exploration = 0.0
        if grid_key not in self.visited_cells:
            self.visited_cells.add(grid_key)
            reward_exploration = config.REWARD_EXPLORATION_BONUS
            reward += reward_exploration
        if config.DEBUG_MODE and config.LOG_REWARD_COMPONENTS:
            self.reward_components['exploration'] = reward_exploration

        # Proximity penalty based on raycasts - умеренный экспоненциальный штраф
        raycasts = self.raycast_sensor.cast_rays(drone_pos, drone_quat, self.CLIENT)
        min_ray = np.min(raycasts)

        # Convert normalized ray to actual distance
        min_dist = min_ray * config.RAY_LENGTH

        # Track min obstacle distance for debug
        if config.DEBUG_MODE and config.LOG_EPISODE_METRICS:
            self.episode_min_obstacle_dist = min(self.episode_min_obstacle_dist, min_dist)
            if min_dist < config.MIN_CLEARANCE:
                self.episode_near_misses += 1

        # Умеренный экспоненциальный штраф
        reward_obstacle = 0.0
        if min_dist < config.REWARD_PROXIMITY_THRESHOLD:
            penalty = config.REWARD_PROXIMITY_SCALE * np.exp(-min_dist)
            reward_obstacle = -penalty
            reward += reward_obstacle
        if config.DEBUG_MODE and config.LOG_REWARD_COMPONENTS:
            self.reward_components['obstacle'] = reward_obstacle

        # Step penalty
        reward_step = -config.REWARD_STEP_PENALTY
        reward += reward_step
        if config.DEBUG_MODE and config.LOG_REWARD_COMPONENTS:
            self.reward_components['step_penalty'] = reward_step

        # Track navigation metrics
        if config.DEBUG_MODE and config.LOG_NAVIGATION_METRICS:
            # Heading error (angle between velocity and goal direction)
            speed = np.linalg.norm(drone_vel)
            if speed > 0.1:
                vel_direction = drone_vel / speed
                heading_error = np.arccos(np.clip(np.dot(vel_direction, goal_direction), -1, 1))
                self.episode_heading_errors.append(np.degrees(heading_error))
                self.episode_speeds.append(speed)

        # Track min goal distance
        if config.DEBUG_MODE and config.LOG_EPISODE_METRICS:
            self.episode_min_goal_dist = min(self.episode_min_goal_dist, curr_dist)

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

        info = {
            "is_success": is_success,
            "is_crash": is_crash,
            "dist_to_goal": dist_to_goal,
            "min_ray_dist": min_ray_dist,
            "step": self.control_step_counter
        }

        # Add debug metrics if enabled
        if config.DEBUG_MODE:
            if config.LOG_REWARD_COMPONENTS and hasattr(self, 'reward_components'):
                info['reward_components'] = self.reward_components.copy()

            if config.LOG_EPISODE_METRICS:
                info['start_distance'] = np.linalg.norm(self.goal_pos - self.start_pos)
                info['min_goal_distance'] = self.episode_min_goal_dist
                info['closest_obstacle'] = self.episode_min_obstacle_dist
                info['n_near_misses'] = self.episode_near_misses

            if config.LOG_NAVIGATION_METRICS and len(self.episode_trajectory) > 1:
                # Path efficiency
                straight_dist = np.linalg.norm(self.goal_pos - self.start_pos)
                path_length = 0.0
                for i in range(1, len(self.episode_trajectory)):
                    path_length += np.linalg.norm(
                        self.episode_trajectory[i] - self.episode_trajectory[i-1]
                    )
                info['path_efficiency'] = straight_dist / (path_length + 1e-6)

                # Average metrics
                if len(self.episode_heading_errors) > 0:
                    info['avg_heading_error'] = np.mean(self.episode_heading_errors)
                if len(self.episode_speeds) > 0:
                    info['avg_speed'] = np.mean(self.episode_speeds)

            if config.LOG_ACTION_STATS and len(self.episode_actions) > 0:
                actions_array = np.array(self.episode_actions)
                info['action_mean'] = np.mean(actions_array, axis=0)
                info['action_std'] = np.std(actions_array, axis=0)

                # Action smoothness
                if len(self.episode_actions) > 1:
                    smoothness = np.mean([
                        np.linalg.norm(self.episode_actions[i] - self.episode_actions[i-1])
                        for i in range(1, len(self.episode_actions))
                    ])
                    info['action_smoothness'] = smoothness

        return info

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
        self.prev_prev_action = np.zeros(4)  # Для smoothness penalty
        self.visited_cells = set()  # Reset exploration tracking

        # Reset debug tracking
        if config.DEBUG_MODE:
            self.episode_trajectory = [self.start_pos.copy()]
            self.episode_actions = []
            self.episode_heading_errors = []
            self.episode_speeds = []
            self.episode_min_obstacle_dist = float('inf')
            self.episode_near_misses = 0
            self.episode_min_goal_dist = self.prev_dist_to_goal
            self.reward_components = {}

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
        self.prev_prev_action = self.prev_action.copy()  # Сохраняем предыдущее действие
        self.prev_action = action.copy()

        # Track action for debug
        if config.DEBUG_MODE and config.LOG_ACTION_STATS:
            self.episode_actions.append(action.copy())

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

        # Track trajectory for debug
        if config.DEBUG_MODE and config.LOG_NAVIGATION_METRICS:
            drone_pos = self._getDroneStateVector(0)[:3]
            self.episode_trajectory.append(drone_pos.copy())

        # Increment control step counter
        self.control_step_counter += 1

        # Add terminal rewards
        reward_terminal = 0.0
        if terminated:
            if info["is_success"]:
                reward_terminal = config.REWARD_SUCCESS

                # Add efficiency bonus for fast completion
                if config.REWARD_EFFICIENCY_BONUS:
                    efficiency = 1.0 - (self.control_step_counter / config.MAX_STEPS)
                    efficiency_bonus = config.REWARD_EFFICIENCY_SCALE * efficiency
                    reward_terminal += efficiency_bonus

                    # Store efficiency bonus separately for debug
                    if config.DEBUG_MODE and config.LOG_REWARD_COMPONENTS:
                        if 'reward_components' not in info:
                            info['reward_components'] = {}
                        info['reward_components']['efficiency_bonus'] = efficiency_bonus

                reward += reward_terminal
            elif info["is_crash"]:
                reward_terminal = config.REWARD_CRASH
                reward += reward_terminal

        # Add timeout penalty if episode ends without success
        if truncated and not terminated:
            reward_terminal = -200.0
            reward += reward_terminal

        # Store terminal reward component
        if config.DEBUG_MODE and config.LOG_REWARD_COMPONENTS:
            if 'reward_components' not in info:
                info['reward_components'] = {}
            info['reward_components']['terminal'] = reward_terminal

        return obs, reward, terminated, truncated, info
