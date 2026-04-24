"""
Navigation environment for RL drone training.
Inherits from BaseRLAviary and implements velocity control.
"""

import numpy as np
import time
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from envs.raycasts import RaycastSensor
from envs.visualization_utils import draw_goal_marker
import config


class NavAviary(BaseRLAviary):
    """Navigation environment with velocity control."""

    def __init__(self, scenario, gui: bool = False, watch_fps: float = None,
                 fixed_map: bool = False, show_trajectory: bool = False,
                 num_drones: int = 1, physics=Physics.PYB):
        """
        Initialize navigation environment.

        Args:
            scenario: Scenario object providing obstacles and start/goal
            gui: Whether to show PyBullet GUI
            watch_fps: Optional target FPS when GUI is enabled (for watch mode)
            fixed_map: Keep same start/goal/obstacles across episodes
            show_trajectory: Draw drone trajectory in GUI
            num_drones: Number of drones in one shared environment
            physics: Physics engine (must be Physics.PYB)
        """
        if num_drones < 1:
            raise ValueError("num_drones must be >= 1")

        self.scenario = scenario
        self.start_pos = None
        self.goal_pos = None
        self.start_pos_all = None
        self.goal_pos_all = None
        self.prev_dist_to_goal = None

        # Get arena bounds from scenario config
        self.arena_size_x = scenario.config.ARENA_SIZE_X
        self.arena_size_y = scenario.config.ARENA_SIZE_Y
        self.arena_height = scenario.config.ARENA_HEIGHT
        self.num_drones = num_drones
        self.swarm_mode = num_drones > 1
        self.control_step_counter = 0  # Renamed to avoid conflict with BaseRLAviary
        if self.swarm_mode:
            self.prev_action = np.zeros((self.num_drones, 4))
            self.prev_prev_action = np.zeros((self.num_drones, 4))
            self.visited_cells = [set() for _ in range(self.num_drones)]
            self._goal_hold_counter = np.zeros(self.num_drones, dtype=np.int32)
        else:
            self.prev_action = np.zeros(4)
            self.prev_prev_action = np.zeros(4)  # Для smoothness penalty
            self.visited_cells = set()  # For exploration bonus
            self._goal_hold_counter = 0
        self.fixed_map = fixed_map
        self.show_trajectory = show_trajectory and gui
        self._trajectory_prev_pos = None
        self._fixed_map_initialized = False
        self._fixed_start_pos = None
        self._fixed_goal_pos = None
        self._fixed_obstacle_specs = []
        self._watch_step_duration = None
        self._last_watch_step_ts = None
        self.swarm_successes_total = 0
        self.swarm_crashes_total = 0
        self.swarm_respawns_total = 0
        self._drone_body_ids = set()
        self._shield_steps = 0
        self._shield_interventions = 0
        self._shield_last_min_dist = float(config.RAY_LENGTH)
        self._trajectory_draw_every = 6 if self.swarm_mode else 1
        self._trajectory_line_life = 3.0 if self.swarm_mode else 0.0
        self._trajectory_line_width = 1 if self.swarm_mode else 3
        self._swarm_colors = [
            [1.0, 0.2, 0.2],
            [0.2, 1.0, 0.2],
            [0.2, 0.6, 1.0],
            [1.0, 0.8, 0.2],
            [1.0, 0.2, 0.8],
            [0.2, 1.0, 1.0],
            [0.9, 0.6, 0.3],
            [0.6, 0.4, 1.0],
        ]

        if gui and watch_fps is not None and watch_fps > 0:
            self._watch_step_duration = 1.0 / watch_fps

        # Debug tracking
        self.episode_trajectory = []  # List of positions
        self.episode_actions = []  # List of actions
        self.episode_heading_errors = []  # List of heading errors
        self.episode_speeds = []  # List of speeds
        self.episode_min_obstacle_dist = float('inf')  # Closest obstacle distance
        self.episode_near_misses = 0  # Count of near misses
        self.episode_min_goal_dist = float('inf')  # Closest to goal
        self.reward_components = {}  # Dict of reward components

        # Extended episode metrics
        self.episode_clearances = []  # List of clearances to nearest obstacle
        self.episode_goal_seeking_steps = 0  # Steps when flying towards goal
        self.episode_total_steps = 0  # Total steps for ratio calculation
        self.episode_yaw_rates = []  # List of yaw rates

        # Initialize raycast sensor
        self.raycast_sensor = RaycastSensor(ray_length=config.RAY_LENGTH)

        # Call parent constructor
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=self.num_drones,
            neighbourhood_radius=np.inf,
            physics=physics,
            pyb_freq=config.PYB_FREQ,
            ctrl_freq=config.CTRL_FREQ,
            gui=gui,
            record=False,
            obs=ObservationType.KIN,
            act=ActionType.VEL
        )
        # Disable heavy GUI debug controls/overlays for better rendering FPS.
        self.USER_DEBUG = False

    def _apply_watch_timing(self):
        """Throttle GUI simulation speed in watch mode."""
        if self._watch_step_duration is None:
            return

        now = time.time()
        if self._last_watch_step_ts is not None:
            elapsed = now - self._last_watch_step_ts
            remaining = self._watch_step_duration - elapsed
            if remaining > 0:
                time.sleep(remaining)
                now = time.time()
        self._last_watch_step_ts = now

    def _optimize_gui_rendering(self):
        """Disable expensive PyBullet GUI overlays for faster rendering."""
        if not self.GUI:
            return
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.CLIENT)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=self.CLIENT)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=self.CLIENT)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=self.CLIENT)

    def _sample_start_goal(self):
        """Sample start/goal pair for a new map."""
        if hasattr(self.scenario, '_generate_start_goal_zones'):
            return self.scenario._generate_start_goal_zones()
        return self.scenario._generate_start_goal()

    def _get_swarm_offsets(self):
        """Generate deterministic offsets for swarm spawn/goal positions."""
        side = int(np.ceil(np.sqrt(self.num_drones)))
        indices = np.arange(self.num_drones)
        row = indices // side
        col = indices % side

        # Keep formation compact and safely inside arena width
        spacing_x = min(0.5, max(0.2, (self.arena_size_x - 1.0) / max(1, side - 1)))
        spacing_y = min(0.5, max(0.2, (self.arena_size_y - 1.0) / max(1, side - 1)))

        offsets = np.zeros((self.num_drones, 3), dtype=float)
        offsets[:, 0] = (col - (side - 1) / 2.0) * spacing_x
        offsets[:, 1] = (row - (side - 1) / 2.0) * spacing_y
        return offsets

    def _build_swarm_positions(self, base_start, base_goal):
        """Create per-drone start/goal positions for swarm mode."""
        offsets = self._get_swarm_offsets()
        starts = np.array(base_start, dtype=float) + offsets
        goals = np.array(base_goal, dtype=float) + offsets

        margin_xy = 0.3
        starts[:, 0] = np.clip(starts[:, 0], -self.arena_size_x / 2 + margin_xy, self.arena_size_x / 2 - margin_xy)
        starts[:, 1] = np.clip(starts[:, 1], -self.arena_size_y / 2 + margin_xy, self.arena_size_y / 2 - margin_xy)
        goals[:, 0] = np.clip(goals[:, 0], -self.arena_size_x / 2 + margin_xy, self.arena_size_x / 2 - margin_xy)
        goals[:, 1] = np.clip(goals[:, 1], -self.arena_size_y / 2 + margin_xy, self.arena_size_y / 2 - margin_xy)
        starts[:, 2] = np.clip(starts[:, 2], 0.5, self.arena_height - 0.2)
        goals[:, 2] = np.clip(goals[:, 2], 0.5, self.arena_height - 0.2)
        return starts, goals

    def _refresh_drone_body_ids(self):
        """Cache current drone body ids after each reset."""
        drone_ids = np.array(self.DRONE_IDS).reshape(-1)
        self._drone_body_ids = {int(body_id) for body_id in drone_ids}

    def _get_ray_ignore_ids(self):
        """Bodies ignored by ray sensor to decouple swarm drones from each other."""
        if self.swarm_mode:
            if not self._drone_body_ids and hasattr(self, "DRONE_IDS"):
                self._refresh_drone_body_ids()
            return self._drone_body_ids
        return None

    def _has_non_peer_contact(self, drone_id, contact_points):
        """True if drone touched obstacle/world (peer drone contacts are ignored)."""
        if not contact_points:
            return False

        if not self.swarm_mode:
            return True

        drone_body_id = int(self.DRONE_IDS[drone_id])
        for contact in contact_points:
            body_a = int(contact[1])
            body_b = int(contact[2])
            other_body = body_b if body_a == drone_body_id else body_a
            if other_body not in self._drone_body_ids:
                return True
        return False

    def _disable_inter_drone_collisions(self):
        """Disable drone-drone collisions for stable parallel rollout."""
        if not self.swarm_mode:
            return
        self._refresh_drone_body_ids()
        for i in range(self.num_drones):
            body_a = int(self.DRONE_IDS[i])
            links_a = [-1] + list(range(p.getNumJoints(body_a, physicsClientId=self.CLIENT)))
            for j in range(i + 1, self.num_drones):
                body_b = int(self.DRONE_IDS[j])
                links_b = [-1] + list(range(p.getNumJoints(body_b, physicsClientId=self.CLIENT)))
                for link_a in links_a:
                    for link_b in links_b:
                        p.setCollisionFilterPair(
                            bodyUniqueIdA=body_a,
                            bodyUniqueIdB=body_b,
                            linkIndexA=link_a,
                            linkIndexB=link_b,
                            enableCollision=0,
                            physicsClientId=self.CLIENT
                        )

    def _check_drone_terminal(self, drone_id):
        """Return terminal flags for one drone in swarm mode."""
        drone_pos = self._getDroneStateVector(drone_id)[:3]
        goal = self.goal_pos_all[drone_id]
        dist_to_goal = np.linalg.norm(goal - drone_pos)
        is_success = self._check_goal_success(dist_to_goal, drone_id=drone_id, update_counter=True)

        contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[drone_id], physicsClientId=self.CLIENT) or []
        has_relevant_contact = self._has_non_peer_contact(drone_id, contact_points)
        out_of_bounds = self._is_out_of_bounds(drone_pos)
        is_crash = has_relevant_contact or out_of_bounds
        return is_success, is_crash, out_of_bounds, has_relevant_contact

    def _check_goal_success(self, dist_to_goal: float, drone_id: int = 0, update_counter: bool = True) -> bool:
        """
        Success with small hysteresis near goal.

        Immediate success is preserved for SUCCESS_DIST.
        Additionally, staying inside SUCCESS_HOLD_RADIUS for SUCCESS_HOLD_STEPS
        consecutive control steps also counts as success to avoid endless circling.
        """
        dist_to_goal = float(dist_to_goal)
        success_dist = float(config.SUCCESS_DIST)
        hold_radius = max(float(getattr(config, "SUCCESS_HOLD_RADIUS", success_dist)), success_dist)
        hold_steps = int(max(1, getattr(config, "SUCCESS_HOLD_STEPS", 1)))

        if dist_to_goal < success_dist:
            if update_counter:
                if self.swarm_mode:
                    self._goal_hold_counter[drone_id] = hold_steps
                else:
                    self._goal_hold_counter = hold_steps
            return True

        if hold_steps <= 1:
            if update_counter:
                if self.swarm_mode:
                    self._goal_hold_counter[drone_id] = 0
                else:
                    self._goal_hold_counter = 0
            return False

        counter = int(self._goal_hold_counter[drone_id] if self.swarm_mode else self._goal_hold_counter)
        if dist_to_goal < hold_radius:
            if update_counter:
                counter += 1
                if self.swarm_mode:
                    self._goal_hold_counter[drone_id] = counter
                else:
                    self._goal_hold_counter = counter
            return counter >= hold_steps

        if update_counter:
            if self.swarm_mode:
                self._goal_hold_counter[drone_id] = 0
            else:
                self._goal_hold_counter = 0
        return False

    def _boundary_clearance(self, drone_pos):
        """Distance to nearest arena boundary (including floor/ceiling)."""
        dx = self.arena_size_x / 2 - abs(float(drone_pos[0]))
        dy = self.arena_size_y / 2 - abs(float(drone_pos[1]))
        dz_low = float(drone_pos[2]) - 0.1
        dz_high = self.arena_height - float(drone_pos[2])
        return min(dx, dy, dz_low, dz_high)

    def _is_out_of_bounds(self, drone_pos):
        """Check arena bounds with small tolerance for numerical jitter."""
        eps = float(getattr(config, "OUT_OF_BOUNDS_EPS", 0.0))
        return (
            abs(float(drone_pos[0])) > (self.arena_size_x / 2 + eps) or
            abs(float(drone_pos[1])) > (self.arena_size_y / 2 + eps) or
            float(drone_pos[2]) < (0.1 - eps) or
            float(drone_pos[2]) > (self.arena_height + eps)
        )

    def _boundary_outward_speed(self, drone_pos, drone_vel):
        """
        Positive velocity component pointing out of arena near nearest boundary.
        Returns 0 when moving inward/tangentially.
        """
        x, y, z = float(drone_pos[0]), float(drone_pos[1]), float(drone_pos[2])
        vx, vy, vz = float(drone_vel[0]), float(drone_vel[1]), float(drone_vel[2])
        half_x = self.arena_size_x / 2
        half_y = self.arena_size_y / 2

        clearances = {
            "x_pos": half_x - x,
            "x_neg": x + half_x,
            "y_pos": half_y - y,
            "y_neg": y + half_y,
            "z_low": z - 0.1,
            "z_high": self.arena_height - z,
        }
        nearest = min(clearances, key=clearances.get)
        if nearest == "x_pos":
            return max(0.0, vx)
        if nearest == "x_neg":
            return max(0.0, -vx)
        if nearest == "y_pos":
            return max(0.0, vy)
        if nearest == "y_neg":
            return max(0.0, -vy)
        if nearest == "z_low":
            return max(0.0, -vz)
        return max(0.0, vz)

    def _apply_safety_shield_single(self, action: np.ndarray, drone_id: int) -> np.ndarray:
        """
        Apply a lightweight safety shield in action space.

        The shield keeps RL policy behavior untouched when clearance is safe.
        When obstacles are close, it brakes and biases velocity away from danger.
        """
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if not getattr(config, "SAFETY_SHIELD_ENABLED", False):
            return action

        state = self._getDroneStateVector(drone_id)
        drone_pos = state[:3]
        drone_quat = state[3:7]
        rays_norm = self.raycast_sensor.cast_rays(
            drone_pos,
            drone_quat,
            self.CLIENT,
            ignore_body_ids=self._get_ray_ignore_ids()
        )
        ray_dist = rays_norm * config.RAY_LENGTH
        min_dist = float(np.min(ray_dist))
        self._shield_steps += 1
        self._shield_last_min_dist = min_dist

        soft = float(config.SAFETY_SHIELD_SOFT_CLEARANCE)
        hard = float(config.SAFETY_SHIELD_HARD_CLEARANCE)
        if min_dist >= soft:
            return action

        top_k = max(1, int(config.SAFETY_SHIELD_TOPK))
        top_idx = np.argsort(ray_dist)[:top_k]
        top_dist = ray_dist[top_idx]
        top_dirs = self.raycast_sensor.ray_directions[top_idx]
        weights = np.clip((soft - top_dist) / max(soft, 1e-6), 0.0, 1.0)

        avoid_body = -np.sum(top_dirs * weights[:, None], axis=0)
        avoid_body[2] *= float(config.SAFETY_SHIELD_VERTICAL_GAIN)

        # Add boundary-aware inward push to prevent wall hits.
        x, y, z = float(drone_pos[0]), float(drone_pos[1]), float(drone_pos[2])
        half_x = self.arena_size_x / 2
        half_y = self.arena_size_y / 2
        inward_world = np.zeros(3, dtype=np.float32)

        if (half_x - x) < soft:   # near +X wall -> push to -X
            inward_world[0] -= (soft - (half_x - x)) / max(soft, 1e-6)
        if (x + half_x) < soft:   # near -X wall -> push to +X
            inward_world[0] += (soft - (x + half_x)) / max(soft, 1e-6)
        if (half_y - y) < soft:   # near +Y wall -> push to -Y
            inward_world[1] -= (soft - (half_y - y)) / max(soft, 1e-6)
        if (y + half_y) < soft:   # near -Y wall -> push to +Y
            inward_world[1] += (soft - (y + half_y)) / max(soft, 1e-6)
        if (z - 0.1) < soft:      # near floor -> push up
            inward_world[2] += (soft - (z - 0.1)) / max(soft, 1e-6)
        if (self.arena_height - z) < soft:  # near ceiling -> push down
            inward_world[2] -= (soft - (self.arena_height - z)) / max(soft, 1e-6)

        if np.linalg.norm(inward_world) > 1e-6:
            rot_matrix = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)
            inward_body = rot_matrix.T @ inward_world
            avoid_body = avoid_body + 0.5 * inward_body

        avoid_norm = np.linalg.norm(avoid_body)
        avoid_dir = avoid_body / avoid_norm if avoid_norm > 1e-6 else np.zeros(3, dtype=np.float32)

        # Risk in [0, 1]: 0 at soft-clearance, 1 at/under hard-clearance.
        risk = np.clip((soft - min_dist) / max(soft - hard, 1e-6), 0.0, 1.0)
        safe_action = action.copy()

        brake_scale = np.clip(1.0 - risk * float(config.SAFETY_SHIELD_BRAKE_GAIN), 0.1, 1.0)
        safe_vel = safe_action[:3] * brake_scale + avoid_dir * float(config.SAFETY_SHIELD_AVOID_GAIN) * risk

        # Hard zone: remove component flying directly into nearest obstacle and brake harder.
        if min_dist < hard:
            closest_idx = int(np.argmin(ray_dist))
            closest_dir = self.raycast_sensor.ray_directions[closest_idx]
            toward_obstacle = float(np.dot(safe_vel, closest_dir))
            if toward_obstacle > 0.0:
                safe_vel = safe_vel - toward_obstacle * closest_dir
            safe_vel *= float(config.SAFETY_SHIELD_HARD_BRAKE_SCALE)

        safe_action[:3] = np.clip(safe_vel, -1.0, 1.0)

        yaw_limit = 1.0 - risk * (1.0 - float(config.SAFETY_SHIELD_MAX_YAW))
        safe_action[3] = float(np.clip(safe_action[3], -yaw_limit, yaw_limit))

        if np.any(np.abs(safe_action - action) > 1e-3):
            self._shield_interventions += 1

        return safe_action

    def _respawn_drone(self, drone_id):
        """Respawn one drone at its start location (swarm mode)."""
        start = self.start_pos_all[drone_id]
        p.resetBasePositionAndOrientation(
            self.DRONE_IDS[drone_id],
            start,
            [0, 0, 0, 1],
            physicsClientId=self.CLIENT
        )
        p.resetBaseVelocity(
            self.DRONE_IDS[drone_id],
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0],
            physicsClientId=self.CLIENT
        )
        self.prev_dist_to_goal[drone_id] = np.linalg.norm(self.goal_pos_all[drone_id] - start)
        self._goal_hold_counter[drone_id] = 0
        self.visited_cells[drone_id] = set()
        if self.show_trajectory and self.GUI and self._trajectory_prev_pos is not None:
            self._trajectory_prev_pos[drone_id] = None

    def _generate_obstacles_for_current_map(self):
        """Generate obstacles for the current start/goal pair."""
        self.scenario.obstacles = []

        if hasattr(self.scenario, '_generate_obstacles'):
            if hasattr(self.scenario, '_resolve_obstacle_type'):
                chosen_type = self.scenario._resolve_obstacle_type()
            else:
                obstacle_type = self.scenario.obstacle_type
                if obstacle_type == 'random':
                    chosen_type = self.scenario.rng.choice(list(self.scenario.config.OBSTACLE_TYPES.keys()))
                else:
                    chosen_type = obstacle_type
            self.scenario._generate_obstacles(chosen_type, self.start_pos, self.goal_pos, self.CLIENT)
            return

        if hasattr(self.scenario, 'n_obstacles'):
            from envs.obstacles import StaticObstacle

            self.scenario.n_obstacles = self.scenario.rng.randint(
                self.scenario.config.STAGE1_N_OBSTACLES[0],
                self.scenario.config.STAGE1_N_OBSTACLES[1] + 1
            )
            for _ in range(self.scenario.n_obstacles):
                radius = self.scenario.rng.uniform(
                    self.scenario.config.STAGE1_RADIUS[0],
                    self.scenario.config.STAGE1_RADIUS[1]
                )
                for _ in range(50):
                    x = self.scenario.rng.uniform(-self.scenario.config.ARENA_SIZE_X / 2 + 1,
                                                   self.scenario.config.ARENA_SIZE_X / 2 - 1)
                    y = self.scenario.rng.uniform(-self.scenario.config.ARENA_SIZE_Y / 2 + 1,
                                                   self.scenario.config.ARENA_SIZE_Y / 2 - 1)
                    pos = np.array([x, y, 0.0])

                    dist_to_start = np.linalg.norm(pos[:2] - self.start_pos[:2])
                    dist_to_goal = np.linalg.norm(pos[:2] - self.goal_pos[:2])

                    if (dist_to_start < self.scenario.config.MIN_CLEARANCE + radius or
                            dist_to_goal < self.scenario.config.MIN_CLEARANCE + radius):
                        continue

                    obstacle = StaticObstacle(
                        position=pos,
                        radius=radius,
                        height=self.scenario.config.ARENA_HEIGHT,
                        physics_client=self.CLIENT
                    )
                    self.scenario.obstacles.append(obstacle)
                    break
            return

        # Stage 0 has no obstacles
        self.scenario.obstacles = []

    def _serialize_obstacles(self):
        """Serialize generated obstacles so fixed-map mode can recreate them."""
        specs = []
        for obstacle in self.scenario.obstacles:
            obstacle_type = obstacle.__class__.__name__
            if obstacle_type == 'StaticObstacle':
                specs.append({
                    'type': obstacle_type,
                    'position': np.array(obstacle.position, dtype=float).tolist(),
                    'radius': float(obstacle.radius),
                    'height': float(obstacle.height)
                })
            elif obstacle_type == 'CylinderObstacle':
                specs.append({
                    'type': obstacle_type,
                    'position': np.array(obstacle.position, dtype=float).tolist(),
                    'radius': float(obstacle.radius),
                    'height': float(obstacle.height)
                })
            elif obstacle_type == 'SphereObstacle':
                specs.append({
                    'type': obstacle_type,
                    'position': np.array(obstacle.initial_position, dtype=float).tolist(),
                    'radius': float(obstacle.radius),
                    'speed': float(obstacle.speed),
                    'amplitude': float(obstacle.amplitude),
                    'frequency': float(obstacle.frequency),
                    'direction': np.array(obstacle.direction, dtype=float).tolist()
                })
            elif obstacle_type == 'WallObstacle':
                specs.append({
                    'type': obstacle_type,
                    'position': np.array(obstacle.position, dtype=float).tolist(),
                    'width': float(obstacle.width),
                    'height': float(obstacle.height),
                    'thickness': float(obstacle.thickness)
                })
            elif obstacle_type == 'BeamObstacle':
                specs.append({
                    'type': obstacle_type,
                    'position': np.array(obstacle.position, dtype=float).tolist(),
                    'length': float(obstacle.length),
                    'height': float(obstacle.height),
                    'thickness': float(obstacle.thickness),
                    'swing_angle': float(np.degrees(obstacle.swing_angle)),
                    'swing_period': float(obstacle.swing_period),
                    'swing_axis': getattr(obstacle, 'swing_axis', 'yaw'),
                    'phase': float(getattr(obstacle, 'phase', 0.0))
                })
            elif obstacle_type == 'BoxObstacle':
                specs.append({
                    'type': obstacle_type,
                    'position': np.array(obstacle.position, dtype=float).tolist(),
                    'size': float(obstacle.size),
                    'height': float(obstacle.height)
                })
            elif obstacle_type == 'SwingingStickObstacle':
                specs.append({
                    'type': obstacle_type,
                    'position': np.array(obstacle.initial_position, dtype=float).tolist(),
                    'length': float(obstacle.length),
                    'thickness': float(obstacle.thickness),
                    'swing_angle': float(np.degrees(obstacle.swing_angle)),
                    'swing_period': float(obstacle.swing_period),
                    'vertical_swing': bool(obstacle.vertical_swing),
                    'vertical_amplitude': float(obstacle.vertical_amplitude),
                    'phase': float(getattr(obstacle, 'phase', 0.0))
                })
        return specs

    def _restore_obstacles_from_specs(self):
        """Restore cached obstacles for fixed-map mode."""
        self.scenario.obstacles = []
        if not self._fixed_obstacle_specs:
            if hasattr(self.scenario, 'n_obstacles'):
                self.scenario.n_obstacles = 0
            return

        from envs.obstacles import StaticObstacle
        from scenarios.pretrain_obstacles import (
            CylinderObstacle,
            SphereObstacle,
            WallObstacle,
            BeamObstacle,
            BoxObstacle,
            SwingingStickObstacle
        )

        for spec in self._fixed_obstacle_specs:
            obstacle = None
            obstacle_type = spec.get('type')

            if obstacle_type == 'StaticObstacle':
                obstacle = StaticObstacle(
                    position=np.array(spec['position'], dtype=float),
                    radius=spec['radius'],
                    height=spec['height'],
                    physics_client=self.CLIENT
                )
            elif obstacle_type == 'CylinderObstacle':
                obstacle = CylinderObstacle(
                    position=np.array(spec['position'], dtype=float),
                    radius=spec['radius'],
                    height=spec['height'],
                    physics_client=self.CLIENT
                )
            elif obstacle_type == 'SphereObstacle':
                obstacle = SphereObstacle(
                    position=np.array(spec['position'], dtype=float),
                    radius=spec['radius'],
                    speed=spec['speed'],
                    amplitude=spec['amplitude'],
                    frequency=spec['frequency'],
                    physics_client=self.CLIENT
                )
                if 'direction' in spec:
                    obstacle.direction = np.array(spec['direction'], dtype=float)
            elif obstacle_type == 'WallObstacle':
                obstacle = WallObstacle(
                    position=np.array(spec['position'], dtype=float),
                    width=spec['width'],
                    height=spec['height'],
                    thickness=spec['thickness'],
                    physics_client=self.CLIENT
                )
            elif obstacle_type == 'BeamObstacle':
                obstacle = BeamObstacle(
                    position=np.array(spec['position'], dtype=float),
                    length=spec['length'],
                    height=spec['height'],
                    thickness=spec['thickness'],
                    swing_angle=spec['swing_angle'],
                    swing_period=spec['swing_period'],
                    physics_client=self.CLIENT,
                    swing_axis=spec.get('swing_axis', 'yaw'),
                    phase=spec.get('phase', 0.0)
                )
            elif obstacle_type == 'BoxObstacle':
                obstacle = BoxObstacle(
                    position=np.array(spec['position'], dtype=float),
                    size=spec['size'],
                    height=spec['height'],
                    physics_client=self.CLIENT
                )
            elif obstacle_type == 'SwingingStickObstacle':
                obstacle = SwingingStickObstacle(
                    position=np.array(spec['position'], dtype=float),
                    length=spec['length'],
                    thickness=spec['thickness'],
                    swing_angle=spec['swing_angle'],
                    swing_period=spec['swing_period'],
                    physics_client=self.CLIENT,
                    vertical_swing=spec.get('vertical_swing', False),
                    vertical_amplitude=spec.get('vertical_amplitude', 0.15),
                    phase=spec.get('phase', 0.0)
                )
            else:
                print(f"[WARN] Unsupported obstacle type in fixed-map cache: {obstacle_type}")

            if obstacle is not None:
                self.scenario.obstacles.append(obstacle)

        if hasattr(self.scenario, 'n_obstacles'):
            self.scenario.n_obstacles = len(self.scenario.obstacles)

    def _draw_watch_markers(self):
        """Draw markers for watch mode."""
        if not (self.show_trajectory and self.GUI):
            return

        if self.swarm_mode:
            starts = self.start_pos_all
            goals = self.goal_pos_all
            for i in range(self.num_drones):
                color = self._swarm_colors[i % len(self._swarm_colors)]
                goal = goals[i]
                start = starts[i]
                p.addUserDebugLine(
                    [goal[0] - 0.15, goal[1], goal[2]],
                    [goal[0] + 0.15, goal[1], goal[2]],
                    color,
                    2,
                    0,
                    physicsClientId=self.CLIENT
                )
                p.addUserDebugLine(
                    [goal[0], goal[1] - 0.15, goal[2]],
                    [goal[0], goal[1] + 0.15, goal[2]],
                    color,
                    2,
                    0,
                    physicsClientId=self.CLIENT
                )
                p.addUserDebugLine(
                    [start[0] - 0.12, start[1], start[2]],
                    [start[0] + 0.12, start[1], start[2]],
                    [0.2, 0.8, 1.0],
                    2,
                    0,
                    physicsClientId=self.CLIENT
                )
        else:
            draw_goal_marker(self.goal_pos, self.CLIENT)
            start = self.start_pos
            p.addUserDebugLine(
                [start[0] - 0.2, start[1], start[2]],
                [start[0] + 0.2, start[1], start[2]],
                [0.2, 0.8, 1.0],
                3,
                0,
                physicsClientId=self.CLIENT
            )
            p.addUserDebugLine(
                [start[0], start[1] - 0.2, start[2]],
                [start[0], start[1] + 0.2, start[2]],
                [0.2, 0.8, 1.0],
                3,
                0,
                physicsClientId=self.CLIENT
            )

    def _draw_trajectory_segment(self):
        """Draw one trajectory segment in GUI."""
        if not (self.show_trajectory and self.GUI):
            return

        # Limit draw rate to avoid GUI slowdown in swarm mode.
        if self.control_step_counter % self._trajectory_draw_every != 0:
            return

        if self.swarm_mode:
            if self._trajectory_prev_pos is None:
                self._trajectory_prev_pos = [None for _ in range(self.num_drones)]
            for i in range(self.num_drones):
                drone_pos = self._getDroneStateVector(i)[:3]
                prev_pos = self._trajectory_prev_pos[i]
                if prev_pos is not None:
                    color = self._swarm_colors[i % len(self._swarm_colors)]
                    p.addUserDebugLine(
                        prev_pos,
                        drone_pos,
                        color,
                        self._trajectory_line_width,
                        self._trajectory_line_life,
                        physicsClientId=self.CLIENT
                    )
                self._trajectory_prev_pos[i] = drone_pos.copy()
        else:
            drone_pos = self._getDroneStateVector(0)[:3]
            if self._trajectory_prev_pos is not None:
                p.addUserDebugLine(
                    self._trajectory_prev_pos,
                    drone_pos,
                    [1.0, 0.2, 0.2],
                    self._trajectory_line_width,
                    self._trajectory_line_life,
                    physicsClientId=self.CLIENT
                )
            self._trajectory_prev_pos = drone_pos.copy()

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

            # Soft boundary-aware attenuation of outward velocity (works without safety shield).
            # This reduces boundary overshoot while preserving inward/tangential control authority.
            drone_pos = state[0:3]
            soft_margin = float(getattr(config, "BOUNDARY_VEL_SOFT_MARGIN", 0.0))
            min_scale = float(getattr(config, "BOUNDARY_VEL_MIN_SCALE", 0.0))
            if soft_margin > 1e-6:
                half_x = self.arena_size_x / 2
                half_y = self.arena_size_y / 2

                def _scale_from_clearance(clearance):
                    clearance = max(0.0, float(clearance))
                    return float(np.clip(clearance / soft_margin, min_scale, 1.0))

                # X boundaries
                if target_vel_world[0] > 0.0:
                    target_vel_world[0] *= _scale_from_clearance(half_x - float(drone_pos[0]))
                elif target_vel_world[0] < 0.0:
                    target_vel_world[0] *= _scale_from_clearance(float(drone_pos[0]) + half_x)

                # Y boundaries
                if target_vel_world[1] > 0.0:
                    target_vel_world[1] *= _scale_from_clearance(half_y - float(drone_pos[1]))
                elif target_vel_world[1] < 0.0:
                    target_vel_world[1] *= _scale_from_clearance(float(drone_pos[1]) + half_y)

                # Z boundaries (floor/ceiling)
                if target_vel_world[2] > 0.0:
                    target_vel_world[2] *= _scale_from_clearance(self.arena_height - float(drone_pos[2]))
                elif target_vel_world[2] < 0.0:
                    target_vel_world[2] *= _scale_from_clearance(float(drone_pos[2]) - 0.1)

            # Smooth down near goal to reduce circling and overshoot in the final approach.
            goal_speed_scale = 1.0
            goal_soft_radius = float(getattr(config, "GOAL_VEL_SOFT_RADIUS", 0.0))
            goal_min_scale = float(getattr(config, "GOAL_VEL_MIN_SCALE", 0.0))
            if goal_soft_radius > 1e-6:
                goal = self.goal_pos_all[k] if self.swarm_mode else self.goal_pos
                if goal is not None:
                    dist_to_goal = float(np.linalg.norm(goal - drone_pos))
                    if dist_to_goal < goal_soft_radius:
                        ratio = np.clip(dist_to_goal / goal_soft_radius, 0.0, 1.0)
                        goal_speed_scale = float(
                            np.clip(goal_min_scale + (1.0 - goal_min_scale) * ratio, goal_min_scale, 1.0)
                        )
                        target_vel_world *= goal_speed_scale

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
        if self.swarm_mode:
            return spaces.Box(low=-1.0, high=1.0, shape=(self.num_drones, 4), dtype=np.float32)
        return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _observationSpace(self):
        """
        Define observation space: 33 features.
        3 (goal) + 1 (dist) + 3 (vel) + 1 (height) + 1 (yaw) + 4 (prev_action) + 20 (raycasts) = 33

        Returns:
            Box space for 33-dimensional observation
        """
        if self.swarm_mode:
            return spaces.Box(low=-1.0, high=1.0, shape=(self.num_drones, 33), dtype=np.float32)
        return spaces.Box(low=-1.0, high=1.0, shape=(33,), dtype=np.float32)

    def _computeObs(self):
        """
        Compute observation vector (33 features).
        3 (goal) + 1 (dist) + 3 (vel) + 1 (height) + 1 (yaw) + 4 (prev_action) + 20 (raycasts) = 33

        Returns:
            np.ndarray of shape (33,)
        """
        max_dist = np.sqrt(self.arena_size_x ** 2 + self.arena_size_y ** 2 + self.arena_height ** 2)

        if self.swarm_mode:
            obs_all = np.zeros((self.num_drones, 33), dtype=np.float32)
            for i in range(self.num_drones):
                state = self._getDroneStateVector(i)
                drone_pos = state[:3]
                drone_quat = state[3:7]
                drone_vel = state[10:13]
                goal = self.goal_pos_all[i]

                goal_world = goal - drone_pos
                rot_matrix = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)
                goal_body = rot_matrix.T @ goal_world
                goal_body_norm = goal_body / max_dist

                dist_to_goal = np.linalg.norm(goal_world)
                dist_to_goal_norm = np.clip(dist_to_goal / max_dist, 0, 1)

                vel_body = rot_matrix.T @ drone_vel
                vel_norm = np.clip(vel_body / np.array([config.VX_MAX, config.VY_MAX, config.VZ_MAX]), -1, 1)

                height_norm = drone_pos[2] / self.arena_height
                qx, qy, qz, qw = drone_quat
                yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy ** 2 + qz ** 2))
                yaw_norm = yaw / np.pi
                prev_action = self.prev_action[i]
                raycasts = self.raycast_sensor.cast_rays(
                    drone_pos,
                    drone_quat,
                    self.CLIENT,
                    ignore_body_ids=self._get_ray_ignore_ids()
                )

                obs_all[i] = np.concatenate([
                    goal_body_norm,
                    [dist_to_goal_norm],
                    vel_norm,
                    [height_norm],
                    [yaw_norm],
                    prev_action,
                    raycasts
                ])
            return obs_all.astype(np.float32)

        # Single-drone mode
        drone_pos = self._getDroneStateVector(0)[:3]
        drone_quat = self._getDroneStateVector(0)[3:7]
        drone_vel = self._getDroneStateVector(0)[10:13]
        goal_world = self.goal_pos - drone_pos
        rot_matrix = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)
        goal_body = rot_matrix.T @ goal_world
        goal_body_norm = goal_body / max_dist
        dist_to_goal = np.linalg.norm(goal_world)
        dist_to_goal_norm = np.clip(dist_to_goal / max_dist, 0, 1)
        vel_body = rot_matrix.T @ drone_vel
        vel_norm = np.clip(vel_body / np.array([config.VX_MAX, config.VY_MAX, config.VZ_MAX]), -1, 1)
        height_norm = drone_pos[2] / self.arena_height
        qx, qy, qz, qw = drone_quat
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
        yaw_norm = yaw / np.pi
        prev_action = self.prev_action
        raycasts = self.raycast_sensor.cast_rays(
            drone_pos,
            drone_quat,
            self.CLIENT,
            ignore_body_ids=self._get_ray_ignore_ids()
        )

        obs = np.concatenate([
            goal_body_norm,
            [dist_to_goal_norm],
            vel_norm,
            [height_norm],
            [yaw_norm],
            prev_action,
            raycasts
        ])
        return obs.astype(np.float32)

    def _log_reward_component(self, name: str, value: float) -> float:
        """
        Helper method to log reward component if debug mode is enabled.

        Args:
            name: Component name
            value: Component value

        Returns:
            The same value (for chaining)
        """
        if config.DEBUG_MODE and config.LOG_REWARD_COMPONENTS:
            self.reward_components[name] = value
        return value

    def _computeReward(self):
        """
        Compute reward based on progress and safety.

        Returns:
            float reward value
        """
        if self.swarm_mode:
            total_reward = 0.0
            for i in range(self.num_drones):
                state = self._getDroneStateVector(i)
                drone_pos = state[:3]
                drone_quat = state[3:7]
                drone_vel = state[10:13]
                goal = self.goal_pos_all[i]
                curr_dist = np.linalg.norm(goal - drone_pos)

                progress = self.prev_dist_to_goal[i] - curr_dist
                reward_i = config.REWARD_PROGRESS_SCALE * progress
                if curr_dist < config.REWARD_NEAR_GOAL_RADIUS:
                    stall = max(0.0, config.REWARD_NEAR_GOAL_PROGRESS_EPS - progress)
                    reward_i += -config.REWARD_NEAR_GOAL_STALL_SCALE * stall

                goal_world = goal - drone_pos
                goal_direction = goal_world / (np.linalg.norm(goal_world) + 1e-6)
                velocity_towards_goal = np.dot(drone_vel, goal_direction)
                reward_i += config.REWARD_VELOCITY_SCALE * max(0, velocity_towards_goal)

                yaw_action = abs(self.prev_action[i, 3])
                reward_i += -config.REWARD_YAW_PENALTY_SCALE * (yaw_action ** 2)

                speed = np.linalg.norm(drone_vel)
                if speed > 0.1:
                    vel_direction = drone_vel / speed
                    heading_alignment = np.dot(vel_direction, goal_direction)
                    reward_i += config.REWARD_HEADING_SCALE * max(0, heading_alignment)

                action_change = np.linalg.norm(self.prev_action[i] - self.prev_prev_action[i])
                reward_i += -config.REWARD_ACTION_SMOOTHNESS_SCALE * action_change

                reward_i += config.REWARD_PROXIMITY_SCALE * np.exp(-curr_dist)
                self.prev_dist_to_goal[i] = curr_dist

                grid_x = int((drone_pos[0] + self.arena_size_x / 2) / config.EXPLORATION_GRID_SIZE)
                grid_y = int((drone_pos[1] + self.arena_size_y / 2) / config.EXPLORATION_GRID_SIZE)
                grid_z = int(drone_pos[2] / config.EXPLORATION_GRID_SIZE)
                grid_key = (grid_x, grid_y, grid_z)
                if grid_key not in self.visited_cells[i]:
                    self.visited_cells[i].add(grid_key)
                    reward_i += config.REWARD_EXPLORATION_BONUS

                raycasts = self.raycast_sensor.cast_rays(
                    drone_pos,
                    drone_quat,
                    self.CLIENT,
                    ignore_body_ids=self._get_ray_ignore_ids()
                )
                min_dist = np.min(raycasts) * config.RAY_LENGTH
                if min_dist < config.REWARD_PROXIMITY_THRESHOLD:
                    reward_i += -config.REWARD_PROXIMITY_SCALE * np.exp(-min_dist)
                    rot_matrix = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)
                    vel_body = rot_matrix.T @ drone_vel
                    closest_idx = int(np.argmin(raycasts))
                    closest_dir_body = self.raycast_sensor.ray_directions[closest_idx]
                    toward_obstacle = float(np.dot(vel_body, closest_dir_body))
                    if toward_obstacle > 0.0:
                        proximity_factor = (config.REWARD_PROXIMITY_THRESHOLD - min_dist) / max(config.REWARD_PROXIMITY_THRESHOLD, 1e-6)
                        reward_i += -config.REWARD_OBSTACLE_APPROACH_SCALE * toward_obstacle * max(0.0, proximity_factor)

                boundary_dist = self._boundary_clearance(drone_pos)
                if boundary_dist < config.REWARD_BOUNDARY_THRESHOLD:
                    reward_i += -config.REWARD_BOUNDARY_SCALE * np.exp(-max(boundary_dist, 0.0))
                    outward_speed = self._boundary_outward_speed(drone_pos, drone_vel)
                    reward_i += -config.REWARD_BOUNDARY_OUTWARD_SCALE * outward_speed

                reward_i += -config.REWARD_STEP_PENALTY
                total_reward += reward_i

            return total_reward / self.num_drones

        drone_pos = self._getDroneStateVector(0)[:3]
        drone_quat = self._getDroneStateVector(0)[3:7]
        drone_vel = self._getDroneStateVector(0)[10:13]
        curr_dist = np.linalg.norm(self.goal_pos - drone_pos)

        # Initialize reward components dict
        if config.DEBUG_MODE and config.LOG_REWARD_COMPONENTS:
            self.reward_components = {}

        # Progress reward
        progress = self.prev_dist_to_goal - curr_dist
        reward_progress = self._log_reward_component('progress', config.REWARD_PROGRESS_SCALE * progress)
        reward = reward_progress
        if curr_dist < config.REWARD_NEAR_GOAL_RADIUS:
            near_goal_stall = max(0.0, config.REWARD_NEAR_GOAL_PROGRESS_EPS - progress)
            reward_near_goal_stall = self._log_reward_component(
                'near_goal_stall',
                -config.REWARD_NEAR_GOAL_STALL_SCALE * near_goal_stall
            )
            reward += reward_near_goal_stall
        else:
            self._log_reward_component('near_goal_stall', 0.0)

        # Velocity reward - награда за полёт в направлении цели
        goal_world = self.goal_pos - drone_pos
        goal_direction = goal_world / (np.linalg.norm(goal_world) + 1e-6)
        rot_matrix = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)
        velocity_towards_goal = np.dot(drone_vel, goal_direction)
        reward_velocity = self._log_reward_component('velocity', config.REWARD_VELOCITY_SCALE * max(0, velocity_towards_goal))
        reward += reward_velocity

        # Yaw penalty - квадратичный штраф за избыточное вращение (всегда применяется)
        yaw_action = abs(self.prev_action[3]) if len(self.prev_action) > 3 else 0
        reward_yaw_penalty = self._log_reward_component('yaw_penalty', -config.REWARD_YAW_PENALTY_SCALE * (yaw_action ** 2))
        reward += reward_yaw_penalty

        # Heading reward - награда за правильное направление к цели
        speed = np.linalg.norm(drone_vel)
        if speed > 0.1:  # Только если дрон движется
            vel_direction = drone_vel / speed
            heading_alignment = np.dot(vel_direction, goal_direction)  # cos угла между скоростью и направлением к цели
            reward_heading = self._log_reward_component('heading', config.REWARD_HEADING_SCALE * max(0, heading_alignment))
            reward += reward_heading
        else:
            self._log_reward_component('heading', 0.0)

        # Action smoothness penalty - штраф за резкие изменения действий
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

        # Exploration bonus - награда за посещение новых клеток
        grid_x = int((drone_pos[0] + self.arena_size_x / 2) / config.EXPLORATION_GRID_SIZE)
        grid_y = int((drone_pos[1] + self.arena_size_y / 2) / config.EXPLORATION_GRID_SIZE)
        grid_z = int(drone_pos[2] / config.EXPLORATION_GRID_SIZE)
        grid_key = (grid_x, grid_y, grid_z)

        if grid_key not in self.visited_cells:
            self.visited_cells.add(grid_key)
            reward_exploration = self._log_reward_component('exploration', config.REWARD_EXPLORATION_BONUS)
            reward += reward_exploration
        else:
            self._log_reward_component('exploration', 0.0)

        # Proximity penalty based on raycasts - умеренный экспоненциальный штраф
        raycasts = self.raycast_sensor.cast_rays(
            drone_pos,
            drone_quat,
            self.CLIENT,
            ignore_body_ids=self._get_ray_ignore_ids()
        )
        min_ray = np.min(raycasts)

        # Convert normalized ray to actual distance
        min_dist = min_ray * config.RAY_LENGTH

        # Track min obstacle distance for debug
        if config.DEBUG_MODE and config.LOG_EPISODE_METRICS:
            self.episode_min_obstacle_dist = min(self.episode_min_obstacle_dist, min_dist)
            if min_dist < config.MIN_CLEARANCE:
                self.episode_near_misses += 1

        # Умеренный экспоненциальный штраф
        if min_dist < config.REWARD_PROXIMITY_THRESHOLD:
            penalty = config.REWARD_PROXIMITY_SCALE * np.exp(-min_dist)
            reward_obstacle = self._log_reward_component('obstacle', -penalty)
            reward += reward_obstacle
            vel_body = rot_matrix.T @ drone_vel
            closest_idx = int(np.argmin(raycasts))
            closest_dir_body = self.raycast_sensor.ray_directions[closest_idx]
            toward_obstacle = float(np.dot(vel_body, closest_dir_body))
            if toward_obstacle > 0.0:
                proximity_factor = (config.REWARD_PROXIMITY_THRESHOLD - min_dist) / max(config.REWARD_PROXIMITY_THRESHOLD, 1e-6)
                reward_obstacle_approach = self._log_reward_component(
                    'obstacle_approach',
                    -config.REWARD_OBSTACLE_APPROACH_SCALE * toward_obstacle * max(0.0, proximity_factor)
                )
                reward += reward_obstacle_approach
            else:
                self._log_reward_component('obstacle_approach', 0.0)
        else:
            self._log_reward_component('obstacle', 0.0)
            self._log_reward_component('obstacle_approach', 0.0)

        # Boundary penalty - discourages flying too close to arena borders
        boundary_dist = self._boundary_clearance(drone_pos)
        if boundary_dist < config.REWARD_BOUNDARY_THRESHOLD:
            boundary_penalty = config.REWARD_BOUNDARY_SCALE * np.exp(-max(boundary_dist, 0.0))
            reward_boundary = self._log_reward_component('boundary', -boundary_penalty)
            reward += reward_boundary
            outward_speed = self._boundary_outward_speed(drone_pos, drone_vel)
            reward_boundary_outward = self._log_reward_component(
                'boundary_outward',
                -config.REWARD_BOUNDARY_OUTWARD_SCALE * outward_speed
            )
            reward += reward_boundary_outward
        else:
            self._log_reward_component('boundary', 0.0)
            self._log_reward_component('boundary_outward', 0.0)

        # Step penalty
        reward_step = self._log_reward_component('step_penalty', -config.REWARD_STEP_PENALTY)
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

        # Track extended episode metrics
        if config.DEBUG_MODE and config.LOG_EXTENDED_EPISODE_METRICS:
            # Track clearance (distance to nearest obstacle)
            self.episode_clearances.append(min_dist)

            # Track goal seeking (is velocity pointing towards goal?)
            speed = np.linalg.norm(drone_vel)
            if speed > 0.1:
                vel_direction = drone_vel / speed
                alignment = np.dot(vel_direction, goal_direction)
                if alignment > 0:  # Flying towards goal (angle < 90°)
                    self.episode_goal_seeking_steps += 1

            # Track yaw rate (convert normalized action to real yaw_rate in rad/s)
            yaw_rate = abs(self.prev_action[3]) * config.YAW_RATE_MAX if len(self.prev_action) > 3 else 0
            self.episode_yaw_rates.append(yaw_rate)

            self.episode_total_steps += 1

        return reward

    def _computeTerminated(self):
        """
        Check if episode should terminate (success, crash, or out of bounds).

        Returns:
            bool indicating termination
        """
        if self.swarm_mode:
            return False

        drone_pos = self._getDroneStateVector(0)[:3]

        # Success: reached goal
        dist_to_goal = np.linalg.norm(self.goal_pos - drone_pos)
        if self._check_goal_success(dist_to_goal, drone_id=0, update_counter=True):
            return True

        # Crash: collision detected
        contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[0], physicsClientId=self.CLIENT) or []
        if len(contact_points) > 0:
            return True

        # Crash: out of bounds (treat as collision with arena boundary)
        if self._is_out_of_bounds(drone_pos):
            return True

        return False

    def _computeTruncated(self):
        """
        Check if episode should be truncated (timeout only).

        Returns:
            bool indicating truncation
        """
        # Timeout
        if self.control_step_counter >= config.MAX_STEPS:
            return True

        return False

    def _computeInfo(self):
        """
        Compute info dictionary.

        Returns:
            dict with episode information
        """
        if self.swarm_mode:
            dists = []
            min_rays = []
            for i in range(self.num_drones):
                state = self._getDroneStateVector(i)
                drone_pos = state[:3]
                drone_quat = state[3:7]
                dists.append(np.linalg.norm(self.goal_pos_all[i] - drone_pos))
                rays = self.raycast_sensor.cast_rays(
                    drone_pos,
                    drone_quat,
                    self.CLIENT,
                    ignore_body_ids=self._get_ray_ignore_ids()
                )
                min_rays.append(np.min(rays) * config.RAY_LENGTH)

            return {
                "is_success": False,
                "is_crash": False,
                "dist_to_goal": float(np.mean(dists)),
                "min_ray_dist": float(np.mean(min_rays)),
                "shield_enabled": bool(config.SAFETY_SHIELD_ENABLED),
                "shield_last_min_dist": float(self._shield_last_min_dist),
                "shield_intervention_ratio": float(self._shield_interventions / max(1, self._shield_steps)),
                "step": self.control_step_counter,
                "swarm_mode": True,
                "swarm_n_drones": self.num_drones,
                "swarm_successes": int(self.swarm_successes_total),
                "swarm_crashes": int(self.swarm_crashes_total),
                "swarm_respawns": int(self.swarm_respawns_total)
            }

        drone_pos = self._getDroneStateVector(0)[:3]
        drone_quat = self._getDroneStateVector(0)[3:7]
        dist_to_goal = np.linalg.norm(self.goal_pos - drone_pos)

        # Check success and crash
        is_success = self._check_goal_success(dist_to_goal, drone_id=0, update_counter=False)
        contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[0], physicsClientId=self.CLIENT) or []

        # Check if out of bounds (also counts as crash)
        out_of_bounds = self._is_out_of_bounds(drone_pos)

        is_crash = len(contact_points) > 0 or out_of_bounds

        # Get min ray distance
        raycasts = self.raycast_sensor.cast_rays(
            drone_pos,
            drone_quat,
            self.CLIENT,
            ignore_body_ids=self._get_ray_ignore_ids()
        )
        min_ray_dist = np.min(raycasts) * config.RAY_LENGTH

        info = {
            "is_success": is_success,
            "is_crash": is_crash,
            "dist_to_goal": dist_to_goal,
            "min_ray_dist": min_ray_dist,
            "boundary_dist": self._boundary_clearance(drone_pos),
            "shield_enabled": bool(config.SAFETY_SHIELD_ENABLED),
            "shield_last_min_dist": float(self._shield_last_min_dist),
            "shield_intervention_ratio": float(self._shield_interventions / max(1, self._shield_steps)),
            "step": self.control_step_counter,
            "has_contact": bool(len(contact_points) > 0),
            "out_of_bounds": bool(out_of_bounds),
            "start_pos": self.start_pos.tolist() if self.start_pos is not None else None,
            "goal_pos": self.goal_pos.tolist() if self.goal_pos is not None else None,
            "final_pos": drone_pos.tolist()
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

            if config.LOG_EXTENDED_EPISODE_METRICS:
                # Average clearance
                if len(self.episode_clearances) > 0:
                    info['avg_clearance'] = np.mean(self.episode_clearances)

                # Hovering time (% of time with low speed)
                if len(self.episode_speeds) > 0:
                    hovering_steps = np.sum(np.array(self.episode_speeds) < config.HOVERING_SPEED_THRESHOLD)
                    info['hovering_time'] = hovering_steps / len(self.episode_speeds)

                # Goal seeking ratio
                if self.episode_total_steps > 0:
                    info['goal_seeking_ratio'] = self.episode_goal_seeking_steps / self.episode_total_steps

                # Spinning time (% of time with high yaw rate)
                if len(self.episode_yaw_rates) > 0:
                    spinning_steps = np.sum(np.array(self.episode_yaw_rates) > config.SPINNING_YAW_THRESHOLD)
                    info['spinning_time'] = spinning_steps / len(self.episode_yaw_rates)

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
        # Sample map once and then keep it fixed across episodes if requested
        if self.fixed_map and self._fixed_map_initialized:
            base_start = self._fixed_start_pos.copy()
            base_goal = self._fixed_goal_pos.copy()
        else:
            base_start, base_goal = self._sample_start_goal()

        self.start_pos = base_start.copy()
        self.goal_pos = base_goal.copy()

        if self.swarm_mode:
            self.start_pos_all, self.goal_pos_all = self._build_swarm_positions(base_start, base_goal)
            self.INIT_XYZS = self.start_pos_all.copy()
            self.INIT_RPYS = np.zeros((self.num_drones, 3))
        else:
            self.start_pos_all = np.array([self.start_pos.copy()])
            self.goal_pos_all = np.array([self.goal_pos.copy()])
            self.INIT_XYZS = np.array([self.start_pos])
            self.INIT_RPYS = np.array([[0, 0, 0]])

        # Call parent reset (this calls p.resetSimulation() and spawns drone at INIT_XYZS)
        obs, info = super().reset(seed=seed, options=options)
        self._optimize_gui_rendering()
        self._refresh_drone_body_ids()

        # Create or restore obstacles AFTER resetSimulation
        if self.fixed_map and self._fixed_map_initialized:
            self._restore_obstacles_from_specs()
        else:
            self._generate_obstacles_for_current_map()
            if self.fixed_map:
                self._fixed_start_pos = base_start.copy()
                self._fixed_goal_pos = base_goal.copy()
                self._fixed_obstacle_specs = self._serialize_obstacles()
                self._fixed_map_initialized = True

        self._disable_inter_drone_collisions()

        # Reset internal state
        self.control_step_counter = 0
        self._shield_steps = 0
        self._shield_interventions = 0
        self._shield_last_min_dist = float(config.RAY_LENGTH)
        if self.swarm_mode:
            self.prev_dist_to_goal = np.linalg.norm(self.goal_pos_all - self.start_pos_all, axis=1)
            self._goal_hold_counter = np.zeros(self.num_drones, dtype=np.int32)
            self.prev_action = np.zeros((self.num_drones, 4))
            self.prev_prev_action = np.zeros((self.num_drones, 4))
            self.visited_cells = [set() for _ in range(self.num_drones)]
            self.swarm_successes_total = 0
            self.swarm_crashes_total = 0
            self.swarm_respawns_total = 0
        else:
            self.prev_dist_to_goal = np.linalg.norm(self.goal_pos - self.start_pos)
            self._goal_hold_counter = 0
            self.prev_action = np.zeros(4)
            self.prev_prev_action = np.zeros(4)  # Для smoothness penalty
            self.visited_cells = set()  # Reset exploration tracking

        # Reset debug tracking
        if config.DEBUG_MODE and not self.swarm_mode:
            self.episode_trajectory = [self.start_pos.copy()]
            self.episode_actions = []
            self.episode_heading_errors = []
            self.episode_speeds = []
            self.episode_min_obstacle_dist = float('inf')
            self.episode_near_misses = 0
            self.episode_min_goal_dist = self.prev_dist_to_goal
            self.reward_components = {}

            # Extended episode metrics
            if config.LOG_EXTENDED_EPISODE_METRICS:
                self.episode_clearances = []  # List of clearances to nearest obstacle
                self.episode_goal_seeking_steps = 0  # Steps when flying towards goal
                self.episode_total_steps = 0  # Total steps for ratio calculation
                self.episode_yaw_rates = []  # List of yaw rates

        if self._watch_step_duration is not None:
            self._last_watch_step_ts = None

        if self.show_trajectory and self.GUI:
            if self.swarm_mode:
                self._trajectory_prev_pos = [self.start_pos_all[i].copy() for i in range(self.num_drones)]
            else:
                self._trajectory_prev_pos = self.start_pos.copy()
            self._draw_watch_markers()
        else:
            self._trajectory_prev_pos = None

        # Recompute observation with correct start position
        obs = self._computeObs()

        return obs, info

    def step(self, action):
        """
        Execute one step.

        Args:
            action: Action vector [vx, vy, vz, yaw_rate] normalized to [-1, 1]

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.swarm_mode:
            action_arr = np.array(action, dtype=np.float32)
            if action_arr.shape == (self.num_drones * 4,):
                action_arr = action_arr.reshape(self.num_drones, 4)
            if action_arr.shape != (self.num_drones, 4):
                raise ValueError(f"Expected action shape ({self.num_drones}, 4), got {action_arr.shape}")

            action_arr = np.clip(action_arr, -1.0, 1.0)
            for i in range(self.num_drones):
                action_arr[i] = self._apply_safety_shield_single(action_arr[i], drone_id=i)
            self.prev_prev_action = self.prev_action.copy()
            self.prev_action = action_arr.copy()

            dt = 1.0 / self.CTRL_FREQ
            self.scenario.update_dynamic_obstacles(dt)

            obs, reward, terminated, truncated, info = super().step(action_arr)
            self._draw_trajectory_segment()
            self.control_step_counter += 1

            reward_terminal = 0.0
            for i in range(self.num_drones):
                is_success, is_crash, out_of_bounds, has_contact = self._check_drone_terminal(i)
                if is_success:
                    self.swarm_successes_total += 1
                    bonus = config.REWARD_SUCCESS
                    if config.REWARD_EFFICIENCY_BONUS:
                        efficiency = 1.0 - (self.control_step_counter / config.MAX_STEPS)
                        bonus += config.REWARD_EFFICIENCY_SCALE * efficiency
                    reward_terminal += bonus
                    self.swarm_respawns_total += 1
                    self._respawn_drone(i)
                elif is_crash:
                    self.swarm_crashes_total += 1
                    reward_terminal += config.REWARD_CRASH
                    if out_of_bounds:
                        reward_terminal += config.REWARD_OUT_OF_BOUNDS_EXTRA
                    if has_contact and not out_of_bounds:
                        reward_terminal += config.REWARD_COLLISION_EXTRA
                    self.swarm_respawns_total += 1
                    self._respawn_drone(i)

            reward = float(reward) + (reward_terminal / self.num_drones)
            terminated = False
            truncated = self.control_step_counter >= config.MAX_STEPS

            if truncated:
                reward += -200.0

            info = self._computeInfo()
            self._apply_watch_timing()
            return obs, reward, terminated, truncated, info

        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        action = self._apply_safety_shield_single(action, drone_id=0)

        # Store action
        self.prev_prev_action = self.prev_action.copy()  # Сохраняем предыдущее действие
        self.prev_action = action.copy()

        # Track action for debug
        if config.DEBUG_MODE and config.LOG_ACTION_STATS:
            self.episode_actions.append(action.copy())

        # Update dynamic obstacles
        dt = 1.0 / self.CTRL_FREQ
        self.scenario.update_dynamic_obstacles(dt)

        # Execute action through parent class
        # IMPORTANT: Pass normalized action [-1, 1], _preprocessAction() will handle scaling
        obs, reward, terminated, truncated, info = super().step(
            np.array([action])
        )

        self._draw_trajectory_segment()

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
                if info.get("out_of_bounds", False):
                    reward_terminal += config.REWARD_OUT_OF_BOUNDS_EXTRA
                if info.get("has_contact", False) and not info.get("out_of_bounds", False):
                    reward_terminal += config.REWARD_COLLISION_EXTRA
                reward += reward_terminal

        # Timeout penalty (only for timeout, not crash)
        if truncated and not terminated:
            reward_terminal = -200.0
            reward += reward_terminal

        # Store terminal reward component
        if config.DEBUG_MODE and config.LOG_REWARD_COMPONENTS:
            if 'reward_components' not in info:
                info['reward_components'] = {}
            info['reward_components']['terminal'] = reward_terminal

        self._apply_watch_timing()

        return obs, reward, terminated, truncated, info
