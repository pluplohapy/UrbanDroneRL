"""
Navigation environment with RRT* global planner.
Extends NavAviary to use waypoint-based navigation.
"""

import numpy as np
import pybullet as p
from gymnasium import spaces
from envs.nav_aviary import NavAviary
from planners.rrt_star import RRTStarPlanner
import config


class NavAviaryWithPlanner(NavAviary):
    """
    Navigation environment with RRT* global planner.

    The planner generates waypoints, and the RL policy learns to follow them.
    """

    def __init__(
        self,
        scenario,
        gui: bool = False,
        use_planner: bool = True,
        replan_freq: int = 0,
        waypoint_threshold: float = None,
        planner_params: dict = None
    ):
        """
        Initialize navigation environment with planner.

        Args:
            scenario: Scenario object providing obstacles and start/goal
            gui: Whether to show PyBullet GUI
            use_planner: Whether to use RRT* planner
            replan_freq: Replan every N steps (0 = no replanning)
            waypoint_threshold: Distance to consider waypoint reached (default: config.WAYPOINT_THRESHOLD)
            planner_params: Dictionary of RRT* parameters (optional)
        """
        self.use_planner = use_planner
        self.replan_freq = replan_freq
        self.waypoint_threshold = waypoint_threshold if waypoint_threshold is not None else config.WAYPOINT_THRESHOLD
        self.steps_since_replan = 0

        # Planner parameters
        default_params = {
            'max_iter': 2000,
            'step_size': 1.0,
            'goal_bias': 0.15,
            'goal_threshold': 0.8,
            'rewire_radius': 3.0,
            'collision_check_resolution': 0.1,
            'verbose': 0  # Quiet by default
        }
        if planner_params:
            default_params.update(planner_params)
        self.planner_params = default_params

        # Planner state
        self.planner = None
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.planning_failed = False

        # Call parent constructor
        super().__init__(scenario, gui)

    def reset(self, seed=None, options=None):
        """
        Reset environment and plan path.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            observation, info
        """
        # Call parent reset (generates start/goal and obstacles)
        obs, info = super().reset(seed=seed, options=options)

        # Plan path if planner enabled
        if self.use_planner:
            self._plan_path()
        else:
            self.waypoints = [self.goal_pos]
            self.current_waypoint_idx = 0
            self.planning_failed = False

        self.steps_since_replan = 0

        # Initialize path following metrics tracking
        if config.DEBUG_MODE and config.LOG_PATH_FOLLOWING_METRICS:
            self.cross_track_errors = []  # List of cross-track errors

        # Add planning info
        info['planning_failed'] = self.planning_failed
        info['n_waypoints'] = len(self.waypoints)
        if len(self.waypoints) > 0:
            info['current_waypoint'] = self.waypoints[self.current_waypoint_idx]

        return obs, info

    def _plan_path(self):
        """Plan path using RRT* planner."""
        verbose = self.planner_params.get('verbose', 0)

        if verbose >= 2:
            print(f"\n[Planner] Planning path...")
            print(f"[Planner]   Start: {self.start_pos}")
            print(f"[Planner]   Goal: {self.goal_pos}")
            print(f"[Planner]   Obstacles: {len(self.scenario.obstacles)}")

        # Create planner
        if self.planner is None:
            self.planner = RRTStarPlanner(
                arena_bounds=(config.ARENA_SIZE_X, config.ARENA_SIZE_Y, config.ARENA_HEIGHT),
                **self.planner_params
            )

        # Plan
        path = self.planner.plan(
            self.start_pos,
            self.goal_pos,
            self.scenario.obstacles
        )

        if path is not None:
            self.waypoints = path
            self.current_waypoint_idx = 0
            self.planning_failed = False
            if verbose >= 2:
                print(f"[Planner] ✓ Path found with {len(path)} waypoints")
        else:
            # Fallback: fly directly to goal
            if verbose >= 1:
                print(f"[Planner] ✗ Planning failed, using direct path")
            self.waypoints = [self.goal_pos]
            self.current_waypoint_idx = 0
            self.planning_failed = True

    def _computeObs(self):
        """
        Compute observation vector.
        Modified to use current waypoint instead of final goal.

        Returns:
            np.ndarray of shape (29,)
        """
        # Get drone state
        drone_pos = self._getDroneStateVector(0)[:3]
        drone_quat = self._getDroneStateVector(0)[3:7]
        drone_vel = self._getDroneStateVector(0)[10:13]

        # Get current target (waypoint or goal)
        if self.use_planner and len(self.waypoints) > 0:
            # Если достигли последнего waypoint, переключаемся на реальную цель
            if self.current_waypoint_idx >= len(self.waypoints) - 1:
                # Проверяем достигли ли последний waypoint
                last_waypoint = self.waypoints[-1]
                dist_to_last_wp = np.linalg.norm(last_waypoint - drone_pos)
                if dist_to_last_wp < self.waypoint_threshold:
                    # Последний waypoint достигнут - переключаемся на реальную цель
                    current_target = self.goal_pos
                else:
                    # Еще летим к последнему waypoint
                    current_target = self.waypoints[self.current_waypoint_idx]
            else:
                # Летим к промежуточному waypoint
                current_target = self.waypoints[self.current_waypoint_idx]
        else:
            current_target = self.goal_pos

        # 1. Target in body frame (3)
        target_world = current_target - drone_pos
        rot_matrix = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)
        target_body = rot_matrix.T @ target_world

        # Normalize target vector
        max_dist = np.sqrt(config.ARENA_SIZE_X**2 + config.ARENA_SIZE_Y**2 + config.ARENA_HEIGHT**2)
        target_body_norm = target_body / max_dist

        # 2. Distance to target (1)
        dist_to_target = np.linalg.norm(target_world)
        dist_to_target_norm = np.clip(dist_to_target / max_dist, 0, 1)

        # 3. Linear velocity in body frame, normalized (3)
        vel_body = rot_matrix.T @ drone_vel
        vel_norm = np.clip(vel_body / np.array([config.VX_MAX, config.VY_MAX, config.VZ_MAX]), -1, 1)

        # 4. Normalized height (1)
        height_norm = drone_pos[2] / config.ARENA_HEIGHT

        # 5. Yaw angle (1)
        qx, qy, qz, qw = drone_quat
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
        yaw_norm = yaw / np.pi

        # 6. Previous action (4)
        prev_action = self.prev_action

        # 7. Raycasts (16)
        raycasts = self.raycast_sensor.cast_rays(drone_pos, drone_quat, self.CLIENT)

        # Concatenate all features
        obs = np.concatenate([
            target_body_norm,      # 3
            [dist_to_target_norm], # 1
            vel_norm,              # 3
            [height_norm],         # 1
            [yaw_norm],            # 1
            prev_action,           # 4
            raycasts               # 16
        ])

        return obs.astype(np.float32)

    def _computeReward(self):
        """
        Compute reward based on progress towards current waypoint.

        Returns:
            float reward value
        """
        drone_pos = self._getDroneStateVector(0)[:3]
        drone_quat = self._getDroneStateVector(0)[3:7]
        drone_vel = self._getDroneStateVector(0)[10:13]

        # Get current target (FIXED: same logic as _computeObs to avoid obs/reward mismatch)
        if self.use_planner and len(self.waypoints) > 0:
            # Если достигли последнего waypoint, переключаемся на реальную цель
            if self.current_waypoint_idx >= len(self.waypoints) - 1:
                # Проверяем достигли ли последний waypoint
                last_waypoint = self.waypoints[-1]
                dist_to_last_wp = np.linalg.norm(last_waypoint - drone_pos)
                if dist_to_last_wp < self.waypoint_threshold:
                    # Последний waypoint достигнут - переключаемся на реальную цель
                    current_target = self.goal_pos
                else:
                    # Еще летим к последнему waypoint
                    current_target = self.waypoints[self.current_waypoint_idx]
            else:
                # Летим к промежуточному waypoint
                current_target = self.waypoints[self.current_waypoint_idx]
        else:
            current_target = self.goal_pos

        curr_dist = np.linalg.norm(current_target - drone_pos)

        # Initialize reward components dict
        if config.DEBUG_MODE and config.LOG_REWARD_COMPONENTS:
            self.reward_components = {}

        # Progress reward (towards current waypoint)
        progress = self.prev_dist_to_goal - curr_dist
        reward_progress = self._log_reward_component('progress', config.REWARD_PROGRESS_SCALE * progress)
        reward = reward_progress

        # Velocity reward
        target_world = current_target - drone_pos
        target_direction = target_world / (np.linalg.norm(target_world) + 1e-6)
        velocity_towards_target = np.dot(drone_vel, target_direction)
        reward_velocity = self._log_reward_component('velocity', config.REWARD_VELOCITY_SCALE * max(0, velocity_towards_target))
        reward += reward_velocity

        # Yaw penalty - квадратичный штраф за избыточное вращение (всегда применяется)
        yaw_action = abs(self.prev_action[3]) if len(self.prev_action) > 3 else 0
        reward_yaw_penalty = self._log_reward_component('yaw_penalty', -config.REWARD_YAW_PENALTY_SCALE * (yaw_action ** 2))
        reward += reward_yaw_penalty

        # Heading reward
        speed = np.linalg.norm(drone_vel)
        if speed > 0.1:
            vel_direction = drone_vel / speed
            heading_alignment = np.dot(vel_direction, target_direction)
            reward_heading = self._log_reward_component('heading', config.REWARD_HEADING_SCALE * max(0, heading_alignment))
            reward += reward_heading
        else:
            self._log_reward_component('heading', 0.0)

        # Action smoothness penalty
        if hasattr(self, 'prev_prev_action') and len(self.prev_prev_action) > 0:
            action_change = np.linalg.norm(self.prev_action - self.prev_prev_action)
            reward_smoothness = self._log_reward_component('smoothness', -config.REWARD_ACTION_SMOOTHNESS_SCALE * action_change)
            reward += reward_smoothness
        else:
            self._log_reward_component('smoothness', 0.0)

        # Proximity bonus - награда за близость к цели
        reward_proximity = self._log_reward_component('proximity', config.REWARD_PROXIMITY_SCALE * np.exp(-curr_dist))
        reward += reward_proximity

        # Update previous distance
        self.prev_dist_to_goal = curr_dist

        # Exploration bonus
        grid_x = int((drone_pos[0] + config.ARENA_SIZE_X / 2) / config.EXPLORATION_GRID_SIZE)
        grid_y = int((drone_pos[1] + config.ARENA_SIZE_Y / 2) / config.EXPLORATION_GRID_SIZE)
        grid_z = int(drone_pos[2] / config.EXPLORATION_GRID_SIZE)
        grid_key = (grid_x, grid_y, grid_z)

        if grid_key not in self.visited_cells:
            self.visited_cells.add(grid_key)
            reward_exploration = self._log_reward_component('exploration', config.REWARD_EXPLORATION_BONUS)
            reward += reward_exploration
        else:
            self._log_reward_component('exploration', 0.0)

        # Proximity penalty based on raycasts
        raycasts = self.raycast_sensor.cast_rays(drone_pos, drone_quat, self.CLIENT)
        min_ray = np.min(raycasts)
        min_dist = min_ray * config.RAY_LENGTH

        if config.DEBUG_MODE and config.LOG_EPISODE_METRICS:
            self.episode_min_obstacle_dist = min(self.episode_min_obstacle_dist, min_dist)
            if min_dist < config.MIN_CLEARANCE:
                self.episode_near_misses += 1

        if min_dist < config.REWARD_PROXIMITY_THRESHOLD:
            penalty = config.REWARD_PROXIMITY_SCALE * np.exp(-min_dist)
            reward_obstacle = self._log_reward_component('obstacle', -penalty)
            reward += reward_obstacle
        else:
            self._log_reward_component('obstacle', 0.0)

        # Step penalty
        reward_step = self._log_reward_component('step_penalty', -config.REWARD_STEP_PENALTY)
        reward += reward_step

        # Track navigation metrics
        if config.DEBUG_MODE and config.LOG_NAVIGATION_METRICS:
            speed = np.linalg.norm(drone_vel)
            if speed > 0.1:
                vel_direction = drone_vel / speed
                heading_error = np.arccos(np.clip(np.dot(vel_direction, target_direction), -1, 1))
                self.episode_heading_errors.append(np.degrees(heading_error))
                self.episode_speeds.append(speed)

        # Track min goal distance (to final goal, not waypoint)
        if config.DEBUG_MODE and config.LOG_EPISODE_METRICS:
            dist_to_final_goal = np.linalg.norm(self.goal_pos - drone_pos)
            self.episode_min_goal_dist = min(self.episode_min_goal_dist, dist_to_final_goal)

        # Track extended episode metrics
        if config.DEBUG_MODE and config.LOG_EXTENDED_EPISODE_METRICS:
            # Track clearance (distance to nearest obstacle)
            self.episode_clearances.append(min_dist)

            # Track goal seeking (is velocity pointing towards final goal?)
            final_goal_direction = (self.goal_pos - drone_pos) / (np.linalg.norm(self.goal_pos - drone_pos) + 1e-6)
            speed = np.linalg.norm(drone_vel)
            if speed > 0.1:
                vel_direction = drone_vel / speed
                alignment = np.dot(vel_direction, final_goal_direction)
                if alignment > 0:  # Flying towards final goal (angle < 90°)
                    self.episode_goal_seeking_steps += 1

            # Track yaw rate (convert normalized action to real yaw_rate in rad/s)
            yaw_rate = abs(self.prev_action[3]) * config.YAW_RATE_MAX if len(self.prev_action) > 3 else 0
            self.episode_yaw_rates.append(yaw_rate)

            self.episode_total_steps += 1

        return reward

    def step(self, action):
        """
        Execute one step with waypoint following logic.

        Args:
            action: Action vector [vx, vy, vz, yaw_rate] normalized to [-1, 1]

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Store action
        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = action.copy()

        # Track action for debug
        if config.DEBUG_MODE and config.LOG_ACTION_STATS:
            self.episode_actions.append(action.copy())

        # Update dynamic obstacles
        dt = 1.0 / self.CTRL_FREQ
        self.scenario.update_dynamic_obstacles(dt)

        # Execute action through parent class
        # IMPORTANT: Pass normalized action [-1, 1], _preprocessAction() will handle scaling
        obs, reward, terminated, truncated, info = super(NavAviary, self).step(
            np.array([action])
        )

        # Check waypoint reached AFTER getting reward
        drone_pos = self._getDroneStateVector(0)[:3]
        if self.use_planner and len(self.waypoints) > 0:
            current_waypoint = self.waypoints[self.current_waypoint_idx]
            dist_to_waypoint = np.linalg.norm(drone_pos - current_waypoint)

            if dist_to_waypoint < self.waypoint_threshold:
                # Move to next waypoint
                if self.current_waypoint_idx < len(self.waypoints) - 1:
                    # Промежуточный waypoint достигнут - дать награду (как в DRL-DroneNavigation)
                    reward += config.REWARD_WAYPOINT
                    self.current_waypoint_idx += 1

                    # IMPORTANT: Update prev_dist for new waypoint to reset progress tracking
                    # This is correct - we need to reset distance when switching to next waypoint
                    new_waypoint = self.waypoints[self.current_waypoint_idx]
                    self.prev_dist_to_goal = np.linalg.norm(new_waypoint - drone_pos)

        # Track trajectory for debug
        if config.DEBUG_MODE and config.LOG_NAVIGATION_METRICS:
            drone_pos = self._getDroneStateVector(0)[:3]
            self.episode_trajectory.append(drone_pos.copy())

        # Track cross-track error (deviation from RRT* path)
        if config.DEBUG_MODE and config.LOG_PATH_FOLLOWING_METRICS and self.use_planner and len(self.waypoints) > 1:
            drone_pos = self._getDroneStateVector(0)[:3]
            cte = self._compute_cross_track_error(drone_pos)
            self.cross_track_errors.append(cte)

        # Increment control step counter
        self.control_step_counter += 1
        self.steps_since_replan += 1

        # Replanning logic
        if self.use_planner and self.replan_freq > 0 and self.steps_since_replan >= self.replan_freq:
            verbose = self.planner_params.get('verbose', 0)
            if verbose >= 2:
                print(f"[Planner] Replanning at step {self.control_step_counter}")
            self._plan_path()
            self.steps_since_replan = 0

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

        # Add planner info
        info['current_waypoint_idx'] = self.current_waypoint_idx
        info['n_waypoints'] = len(self.waypoints)
        info['planning_failed'] = self.planning_failed

        # Add path following metrics
        if config.DEBUG_MODE and config.LOG_PATH_FOLLOWING_METRICS and len(self.cross_track_errors) > 0:
            info['avg_cross_track_error'] = np.mean(self.cross_track_errors)
            info['max_cross_track_error'] = np.max(self.cross_track_errors)
            # Path following score: % of time when CTE < threshold
            within_threshold = np.sum(np.array(self.cross_track_errors) < config.CROSS_TRACK_ERROR_THRESHOLD)
            info['path_following_score'] = within_threshold / len(self.cross_track_errors)

        return obs, reward, terminated, truncated, info

    def _compute_cross_track_error(self, drone_pos: np.ndarray) -> float:
        """
        Compute cross-track error: perpendicular distance from drone to nearest path segment.

        Args:
            drone_pos: Current drone position [x, y, z]

        Returns:
            Cross-track error in meters
        """
        if len(self.waypoints) < 2:
            return 0.0

        # Find the relevant path segment (from previous waypoint to current waypoint)
        if self.current_waypoint_idx == 0:
            # Before first waypoint: segment from start to first waypoint
            p1 = self.start_pos
            p2 = self.waypoints[0]
        else:
            # Between waypoints: segment from previous to current
            p1 = self.waypoints[self.current_waypoint_idx - 1]
            p2 = self.waypoints[self.current_waypoint_idx]

        # Vector from p1 to p2 (path segment)
        segment = p2 - p1
        segment_length = np.linalg.norm(segment)

        if segment_length < 1e-6:
            # Degenerate segment, return distance to waypoint
            return np.linalg.norm(drone_pos - p2)

        # Vector from p1 to drone
        p1_to_drone = drone_pos - p1

        # Project drone position onto the line segment
        # t = how far along the segment (0 = at p1, 1 = at p2)
        t = np.dot(p1_to_drone, segment) / (segment_length ** 2)
        t = np.clip(t, 0, 1)  # Clamp to segment

        # Closest point on segment
        closest_point = p1 + t * segment

        # Cross-track error is distance from drone to closest point
        cte = np.linalg.norm(drone_pos - closest_point)

        return cte
