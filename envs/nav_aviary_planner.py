
import numpy as np
import pybullet as p
from gymnasium import spaces
from envs.nav_aviary import NavAviary
from planners.rrt_star import RRTStarPlanner
import config


class NavAviaryWithPlanner(NavAviary):

    def __init__(
        self,
        scenario,
        gui: bool = False,
        watch_fps: float = None,
        fixed_map: bool = False,
        show_trajectory: bool = False,
        use_planner: bool = True,
        replan_freq: int = 0,
        waypoint_threshold: float = None,
        planner_params: dict = None
    ):
        self.use_planner = use_planner
        self.replan_freq = replan_freq
        self.waypoint_threshold = waypoint_threshold if waypoint_threshold is not None else config.WAYPOINT_THRESHOLD
        self.steps_since_replan = 0


        default_params = {
            'max_iter': 2000,
            'step_size': 1.0,
            'goal_bias': 0.15,
            'goal_threshold': 0.8,
            'rewire_radius': 3.0,
            'collision_check_resolution': 0.1,
            'verbose': 0
        }
        if planner_params:
            default_params.update(planner_params)
        self.planner_params = default_params


        self.planner = None
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.planning_failed = False


        super().__init__(
            scenario=scenario,
            gui=gui,
            watch_fps=watch_fps,
            fixed_map=fixed_map,
            show_trajectory=show_trajectory
        )

    def reset(self, seed=None, options=None):

        obs, info = super().reset(seed=seed, options=options)


        if self.use_planner:
            self._plan_path()
        else:
            self.waypoints = [self.goal_pos]
            self.current_waypoint_idx = 0
            self.planning_failed = False

        self.steps_since_replan = 0


        if config.DEBUG_MODE and config.LOG_PATH_FOLLOWING_METRICS:
            self.cross_track_errors = []


        info['planning_failed'] = self.planning_failed
        info['n_waypoints'] = len(self.waypoints)
        if len(self.waypoints) > 0:
            info['current_waypoint'] = self.waypoints[self.current_waypoint_idx]

        return obs, info

    def _plan_path(self):
        verbose = self.planner_params.get('verbose', 0)

        if verbose >= 2:
            print(f"\n[Planner] Planning path...")
            print(f"[Planner]   Start: {self.start_pos}")
            print(f"[Planner]   Goal: {self.goal_pos}")
            print(f"[Planner]   Obstacles: {len(self.scenario.obstacles)}")


        if self.planner is None:
            self.planner = RRTStarPlanner(
                arena_bounds=(config.ARENA_SIZE_X, config.ARENA_SIZE_Y, config.ARENA_HEIGHT),
                **self.planner_params
            )


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

            if verbose >= 1:
                print(f"[Planner] ✗ Planning failed, using direct path")
            self.waypoints = [self.goal_pos]
            self.current_waypoint_idx = 0
            self.planning_failed = True

    def _computeObs(self):

        drone_pos = self._getDroneStateVector(0)[:3]
        drone_quat = self._getDroneStateVector(0)[3:7]
        drone_vel = self._getDroneStateVector(0)[10:13]


        if self.use_planner and len(self.waypoints) > 0:

            if self.current_waypoint_idx >= len(self.waypoints) - 1:

                last_waypoint = self.waypoints[-1]
                dist_to_last_wp = np.linalg.norm(last_waypoint - drone_pos)
                if dist_to_last_wp < self.waypoint_threshold:

                    current_target = self.goal_pos
                else:

                    current_target = self.waypoints[self.current_waypoint_idx]
            else:

                current_target = self.waypoints[self.current_waypoint_idx]
        else:
            current_target = self.goal_pos


        target_world = current_target - drone_pos
        rot_matrix = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)
        target_body = rot_matrix.T @ target_world


        max_dist = np.sqrt(config.ARENA_SIZE_X**2 + config.ARENA_SIZE_Y**2 + config.ARENA_HEIGHT**2)
        target_body_norm = target_body / max_dist


        dist_to_target = np.linalg.norm(target_world)
        dist_to_target_norm = np.clip(dist_to_target / max_dist, 0, 1)


        vel_body = rot_matrix.T @ drone_vel
        vel_norm = np.clip(vel_body / np.array([config.VX_MAX, config.VY_MAX, config.VZ_MAX]), -1, 1)


        height_norm = drone_pos[2] / config.ARENA_HEIGHT


        qx, qy, qz, qw = drone_quat
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
        yaw_norm = yaw / np.pi


        prev_action = self.prev_action


        raycasts = self.raycast_sensor.cast_rays(drone_pos, drone_quat, self.CLIENT)


        obs = np.concatenate([
            target_body_norm,
            [dist_to_target_norm],
            vel_norm,
            [height_norm],
            [yaw_norm],
            prev_action,
            raycasts
        ])

        return obs.astype(np.float32)

    def _computeReward(self):
        drone_pos = self._getDroneStateVector(0)[:3]
        drone_quat = self._getDroneStateVector(0)[3:7]
        drone_vel = self._getDroneStateVector(0)[10:13]


        if self.use_planner and len(self.waypoints) > 0:

            if self.current_waypoint_idx >= len(self.waypoints) - 1:

                last_waypoint = self.waypoints[-1]
                dist_to_last_wp = np.linalg.norm(last_waypoint - drone_pos)
                if dist_to_last_wp < self.waypoint_threshold:

                    current_target = self.goal_pos
                else:

                    current_target = self.waypoints[self.current_waypoint_idx]
            else:

                current_target = self.waypoints[self.current_waypoint_idx]
        else:
            current_target = self.goal_pos

        curr_dist = np.linalg.norm(current_target - drone_pos)


        if config.DEBUG_MODE and config.LOG_REWARD_COMPONENTS:
            self.reward_components = {}


        progress = self.prev_dist_to_goal - curr_dist
        reward_progress = self._log_reward_component('progress', config.REWARD_PROGRESS_SCALE * progress)
        reward = reward_progress


        target_world = current_target - drone_pos
        target_direction = target_world / (np.linalg.norm(target_world) + 1e-6)
        velocity_towards_target = np.dot(drone_vel, target_direction)
        reward_velocity = self._log_reward_component('velocity', config.REWARD_VELOCITY_SCALE * max(0, velocity_towards_target))
        reward += reward_velocity


        yaw_action = abs(self.prev_action[3]) if len(self.prev_action) > 3 else 0
        reward_yaw_penalty = self._log_reward_component('yaw_penalty', -config.REWARD_YAW_PENALTY_SCALE * (yaw_action ** 2))
        reward += reward_yaw_penalty


        speed = np.linalg.norm(drone_vel)
        if speed > 0.1:
            vel_direction = drone_vel / speed
            heading_alignment = np.dot(vel_direction, target_direction)
            reward_heading = self._log_reward_component('heading', config.REWARD_HEADING_SCALE * max(0, heading_alignment))
            reward += reward_heading
        else:
            self._log_reward_component('heading', 0.0)


        if hasattr(self, 'prev_prev_action') and len(self.prev_prev_action) > 0:
            action_change = np.linalg.norm(self.prev_action - self.prev_prev_action)
            reward_smoothness = self._log_reward_component('smoothness', -config.REWARD_ACTION_SMOOTHNESS_SCALE * action_change)
            reward += reward_smoothness
        else:
            self._log_reward_component('smoothness', 0.0)


        reward_proximity = self._log_reward_component('proximity', config.REWARD_PROXIMITY_SCALE * np.exp(-curr_dist))
        reward += reward_proximity


        self.prev_dist_to_goal = curr_dist


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


        boundary_dist = self._boundary_clearance(drone_pos)
        if boundary_dist < config.REWARD_BOUNDARY_THRESHOLD:
            boundary_penalty = config.REWARD_BOUNDARY_SCALE * np.exp(-max(boundary_dist, 0.0))
            reward_boundary = self._log_reward_component('boundary', -boundary_penalty)
            reward += reward_boundary
        else:
            self._log_reward_component('boundary', 0.0)


        reward_step = self._log_reward_component('step_penalty', -config.REWARD_STEP_PENALTY)
        reward += reward_step


        if config.DEBUG_MODE and config.LOG_NAVIGATION_METRICS:
            speed = np.linalg.norm(drone_vel)
            if speed > 0.1:
                vel_direction = drone_vel / speed
                heading_error = np.arccos(np.clip(np.dot(vel_direction, target_direction), -1, 1))
                self.episode_heading_errors.append(np.degrees(heading_error))
                self.episode_speeds.append(speed)


        if config.DEBUG_MODE and config.LOG_EPISODE_METRICS:
            dist_to_final_goal = np.linalg.norm(self.goal_pos - drone_pos)
            self.episode_min_goal_dist = min(self.episode_min_goal_dist, dist_to_final_goal)


        if config.DEBUG_MODE and config.LOG_EXTENDED_EPISODE_METRICS:

            self.episode_clearances.append(min_dist)


            final_goal_direction = (self.goal_pos - drone_pos) / (np.linalg.norm(self.goal_pos - drone_pos) + 1e-6)
            speed = np.linalg.norm(drone_vel)
            if speed > 0.1:
                vel_direction = drone_vel / speed
                alignment = np.dot(vel_direction, final_goal_direction)
                if alignment > 0:
                    self.episode_goal_seeking_steps += 1


            yaw_rate = abs(self.prev_action[3]) * config.YAW_RATE_MAX if len(self.prev_action) > 3 else 0
            self.episode_yaw_rates.append(yaw_rate)

            self.episode_total_steps += 1

        return reward

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        action = self._apply_safety_shield_single(action, drone_id=0)


        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = action.copy()


        if config.DEBUG_MODE and config.LOG_ACTION_STATS:
            self.episode_actions.append(action.copy())


        dt = 1.0 / self.CTRL_FREQ
        self.scenario.update_dynamic_obstacles(dt)



        obs, reward, terminated, truncated, info = super(NavAviary, self).step(
            np.array([action])
        )

        self._draw_trajectory_segment()


        drone_pos = self._getDroneStateVector(0)[:3]
        if self.use_planner and len(self.waypoints) > 0:
            current_waypoint = self.waypoints[self.current_waypoint_idx]
            dist_to_waypoint = np.linalg.norm(drone_pos - current_waypoint)

            if dist_to_waypoint < self.waypoint_threshold:

                if self.current_waypoint_idx < len(self.waypoints) - 1:

                    reward += config.REWARD_WAYPOINT
                    self.current_waypoint_idx += 1



                    new_waypoint = self.waypoints[self.current_waypoint_idx]
                    self.prev_dist_to_goal = np.linalg.norm(new_waypoint - drone_pos)


        if config.DEBUG_MODE and config.LOG_NAVIGATION_METRICS:
            drone_pos = self._getDroneStateVector(0)[:3]
            self.episode_trajectory.append(drone_pos.copy())


        if config.DEBUG_MODE and config.LOG_PATH_FOLLOWING_METRICS and self.use_planner and len(self.waypoints) > 1:
            drone_pos = self._getDroneStateVector(0)[:3]
            cte = self._compute_cross_track_error(drone_pos)
            self.cross_track_errors.append(cte)


        self.control_step_counter += 1
        self.steps_since_replan += 1


        if self.use_planner and self.replan_freq > 0 and self.steps_since_replan >= self.replan_freq:
            verbose = self.planner_params.get('verbose', 0)
            if verbose >= 2:
                print(f"[Planner] Replanning at step {self.control_step_counter}")
            self._plan_path()
            self.steps_since_replan = 0


        reward_terminal = 0.0
        if terminated:
            if info["is_success"]:
                reward_terminal = config.REWARD_SUCCESS


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
                if info.get("out_of_bounds", False):
                    reward_terminal += config.REWARD_OUT_OF_BOUNDS_EXTRA
                if info.get("has_contact", False) and not info.get("out_of_bounds", False):
                    reward_terminal += config.REWARD_COLLISION_EXTRA
                reward += reward_terminal


        if truncated and not terminated:
            reward_terminal = -200.0
            reward += reward_terminal


        if config.DEBUG_MODE and config.LOG_REWARD_COMPONENTS:
            if 'reward_components' not in info:
                info['reward_components'] = {}
            info['reward_components']['terminal'] = reward_terminal


        info['current_waypoint_idx'] = self.current_waypoint_idx
        info['n_waypoints'] = len(self.waypoints)
        info['planning_failed'] = self.planning_failed


        if config.DEBUG_MODE and config.LOG_PATH_FOLLOWING_METRICS and len(self.cross_track_errors) > 0:
            info['avg_cross_track_error'] = np.mean(self.cross_track_errors)
            info['max_cross_track_error'] = np.max(self.cross_track_errors)

            within_threshold = np.sum(np.array(self.cross_track_errors) < config.CROSS_TRACK_ERROR_THRESHOLD)
            info['path_following_score'] = within_threshold / len(self.cross_track_errors)

        self._apply_watch_timing()

        return obs, reward, terminated, truncated, info

    def _compute_cross_track_error(self, drone_pos: np.ndarray) -> float:
        if len(self.waypoints) < 2:
            return 0.0


        if self.current_waypoint_idx == 0:

            p1 = self.start_pos
            p2 = self.waypoints[0]
        else:

            p1 = self.waypoints[self.current_waypoint_idx - 1]
            p2 = self.waypoints[self.current_waypoint_idx]


        segment = p2 - p1
        segment_length = np.linalg.norm(segment)

        if segment_length < 1e-6:

            return np.linalg.norm(drone_pos - p2)


        p1_to_drone = drone_pos - p1



        t = np.dot(p1_to_drone, segment) / (segment_length ** 2)
        t = np.clip(t, 0, 1)


        closest_point = p1 + t * segment


        cte = np.linalg.norm(drone_pos - closest_point)

        return cte