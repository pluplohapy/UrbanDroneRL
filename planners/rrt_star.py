"""
RRT* (Rapidly-exploring Random Tree Star) planner for 3D drone navigation.

Features:
- Asymptotically optimal path planning
- KD-tree for efficient nearest neighbor search
- Rewiring for path optimization
- Goal biasing for faster convergence
- Collision checking with cylindrical obstacles
- Path smoothing
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial import KDTree
import time


class Node:
    """Node in the RRT* tree."""

    def __init__(self, position: np.ndarray, parent: Optional['Node'] = None):
        """
        Initialize node.

        Args:
            position: [x, y, z] position
            parent: Parent node in tree
        """
        self.position = np.array(position, dtype=np.float32)
        self.parent = parent
        self.cost = 0.0  # Cost from start to this node
        self.children = []  # List of child nodes

    def __repr__(self):
        return f"Node(pos={self.position}, cost={self.cost:.2f})"


class RRTStarPlanner:
    """
    RRT* planner for 3D navigation with cylindrical obstacles.

    Algorithm:
    1. Sample random point in space (with goal bias)
    2. Find nearest node in tree
    3. Steer towards sample with step size limit
    4. Check collision-free path
    5. Find neighbors within rewire radius
    6. Choose best parent (minimum cost)
    7. Add new node to tree
    8. Rewire neighbors if better path through new node
    9. Repeat until goal reached or max iterations
    """

    def __init__(
        self,
        arena_bounds: Tuple[float, float, float],
        max_iter: int = 2000,
        step_size: float = 1.0,
        goal_bias: float = 0.1,
        goal_threshold: float = 0.8,
        rewire_radius: float = 3.0,
        collision_check_resolution: float = 0.1,
        verbose: int = 0
    ):
        """
        Initialize RRT* planner.

        Args:
            arena_bounds: (size_x, size_y, height) arena dimensions
            max_iter: Maximum number of iterations
            step_size: Maximum distance to extend tree
            goal_bias: Probability of sampling goal (0.0-1.0)
            goal_threshold: Distance to goal for success
            rewire_radius: Radius for rewiring neighbors
            collision_check_resolution: Resolution for line collision checks (meters)
            verbose: Verbosity level (0=quiet, 1=summary, 2=detailed)
        """
        self.arena_bounds = arena_bounds
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.goal_threshold = goal_threshold
        self.rewire_radius = rewire_radius
        self.collision_check_resolution = collision_check_resolution
        self.verbose = verbose

        # Tree storage
        self.nodes = []
        self.kdtree = None

        # Obstacles
        self.obstacles = []

        # Statistics
        self.planning_time = 0.0
        self.iterations_used = 0
        self.path_cost = 0.0

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List
    ) -> Optional[List[np.ndarray]]:
        """
        Plan path from start to goal avoiding obstacles.

        Args:
            start: [x, y, z] start position
            goal: [x, y, z] goal position
            obstacles: List of obstacle objects with .position, .radius, .height

        Returns:
            List of waypoints [start, wp1, wp2, ..., goal] or None if no path found
        """
        start_time = time.time()

        # Initialize
        self.nodes = []
        self.obstacles = obstacles
        start_node = Node(start, parent=None)
        start_node.cost = 0.0
        self.nodes.append(start_node)

        goal_node = None
        best_goal_node = None
        best_goal_cost = float('inf')

        if self.verbose >= 2:
            print(f"[RRT*] Planning from {start} to {goal}")
            print(f"[RRT*] Obstacles: {len(obstacles)}")
            print(f"[RRT*] Max iterations: {self.max_iter}")

        # Main RRT* loop
        for iteration in range(self.max_iter):
            # 1. Sample random point
            if np.random.random() < self.goal_bias:
                sample = goal
            else:
                sample = self._sample_free()

            # 2. Find nearest node
            nearest_node = self._nearest(sample)

            # 3. Steer towards sample
            new_pos = self._steer(nearest_node.position, sample)

            # 4. Check collision
            if not self._collision_free(nearest_node.position, new_pos):
                continue

            # 5. Find neighbors for rewiring
            neighbors = self._near(new_pos)

            # 6. Choose best parent
            min_cost = nearest_node.cost + self._distance(nearest_node.position, new_pos)
            best_parent = nearest_node

            for neighbor in neighbors:
                if self._collision_free(neighbor.position, new_pos):
                    cost = neighbor.cost + self._distance(neighbor.position, new_pos)
                    if cost < min_cost:
                        min_cost = cost
                        best_parent = neighbor

            # 7. Add new node
            new_node = Node(new_pos, parent=best_parent)
            new_node.cost = min_cost
            best_parent.children.append(new_node)
            self.nodes.append(new_node)

            # Rebuild KD-tree (every 50 iterations for efficiency, was 10)
            if len(self.nodes) % 50 == 0:
                self._rebuild_kdtree()

            # 8. Rewire neighbors
            for neighbor in neighbors:
                new_cost = new_node.cost + self._distance(new_node.position, neighbor.position)
                if new_cost < neighbor.cost:
                    if self._collision_free(new_node.position, neighbor.position):
                        # Remove from old parent
                        if neighbor.parent:
                            neighbor.parent.children.remove(neighbor)

                        # Set new parent
                        neighbor.parent = new_node
                        neighbor.cost = new_cost
                        new_node.children.append(neighbor)

                        # Update costs of descendants
                        self._update_descendants_cost(neighbor)

            # 9. Check if goal reached
            dist_to_goal = self._distance(new_pos, goal)
            if dist_to_goal < self.goal_threshold:
                if new_node.cost < best_goal_cost:
                    best_goal_cost = new_node.cost
                    best_goal_node = new_node
                    goal_node = new_node
                    if self.verbose >= 2:
                        print(f"[RRT*] Goal reached at iteration {iteration+1}, cost: {best_goal_cost:.2f}")

            # Progress update
            if self.verbose >= 2 and (iteration + 1) % 500 == 0:
                best_cost_str = f"{best_goal_cost:.2f}" if best_goal_node else "inf"
                print(f"[RRT*] Iteration {iteration+1}/{self.max_iter}, nodes: {len(self.nodes)}, best_cost: {best_cost_str}")

        self.planning_time = time.time() - start_time
        self.iterations_used = self.max_iter

        # Extract path
        if best_goal_node is not None:
            path = self._extract_path(best_goal_node)
            self.path_cost = best_goal_node.cost

            if self.verbose >= 1:
                print(f"[RRT*] ✓ Path: {len(path)} waypoints, cost: {self.path_cost:.2f}m, time: {self.planning_time:.2f}s")

            if self.verbose >= 2:
                print(f"[RRT*]   Tree nodes: {len(self.nodes)}")

            # Smooth path
            smoothed_path = self._smooth_path(path)
            if self.verbose >= 2:
                print(f"[RRT*]   Smoothed waypoints: {len(smoothed_path)}")

            return smoothed_path
        else:
            if self.verbose >= 1:
                print(f"[RRT*] ✗ No path found after {self.max_iter} iterations")
            if self.verbose >= 2:
                print(f"[RRT*]   Planning time: {self.planning_time:.3f}s")
                print(f"[RRT*]   Tree nodes: {len(self.nodes)}")
            return None

    def _sample_free(self) -> np.ndarray:
        """
        Sample random point in free space.

        Returns:
            [x, y, z] random position
        """
        x = np.random.uniform(-self.arena_bounds[0]/2, self.arena_bounds[0]/2)
        y = np.random.uniform(-self.arena_bounds[1]/2, self.arena_bounds[1]/2)
        z = np.random.uniform(0.5, self.arena_bounds[2] - 0.5)
        return np.array([x, y, z])

    def _nearest(self, point: np.ndarray) -> Node:
        """
        Find nearest node in tree to point.

        Args:
            point: [x, y, z] query point

        Returns:
            Nearest node
        """
        if self.kdtree is None or len(self.nodes) % 10 == 1:
            self._rebuild_kdtree()

        if len(self.nodes) == 1:
            return self.nodes[0]

        # Query KD-tree
        _, idx = self.kdtree.query(point)
        return self.nodes[idx]

    def _near(self, point: np.ndarray) -> List[Node]:
        """
        Find all nodes within rewire radius of point.

        Args:
            point: [x, y, z] query point

        Returns:
            List of nearby nodes
        """
        if self.kdtree is None:
            self._rebuild_kdtree()

        if len(self.nodes) == 1:
            return []

        # Query KD-tree for neighbors
        indices = self.kdtree.query_ball_point(point, self.rewire_radius)
        return [self.nodes[i] for i in indices]

    def _rebuild_kdtree(self):
        """Rebuild KD-tree from current nodes."""
        if len(self.nodes) > 0:
            positions = np.array([node.position for node in self.nodes])
            self.kdtree = KDTree(positions)

    def _steer(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        """
        Steer from from_pos towards to_pos with step size limit.

        Args:
            from_pos: Starting position
            to_pos: Target position

        Returns:
            New position (at most step_size away from from_pos)
        """
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)

        if distance <= self.step_size:
            return to_pos
        else:
            return from_pos + (direction / distance) * self.step_size

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Euclidean distance between two positions."""
        return np.linalg.norm(pos2 - pos1)

    def _collision_free(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """
        Check if line segment from pos1 to pos2 is collision-free.

        Args:
            pos1: Start position
            pos2: End position

        Returns:
            True if collision-free, False otherwise
        """
        # Discretize line segment
        distance = self._distance(pos1, pos2)
        n_checks = max(2, int(distance / self.collision_check_resolution))

        for i in range(n_checks + 1):
            t = i / n_checks
            point = pos1 + t * (pos2 - pos1)

            # Check against all obstacles
            for obstacle in self.obstacles:
                if self._point_in_cylinder(point, obstacle):
                    return False

        return True

    def _point_in_cylinder(self, point: np.ndarray, obstacle) -> bool:
        """
        Check if point is inside cylindrical obstacle.

        Args:
            point: [x, y, z] point to check
            obstacle: Obstacle object with .position, .radius, .height

        Returns:
            True if point is inside obstacle
        """
        obs_pos = obstacle.get_position()

        # Check horizontal distance
        horizontal_dist = np.linalg.norm(point[:2] - obs_pos[:2])
        if horizontal_dist > obstacle.radius:
            return False

        # Check vertical range
        if point[2] < obs_pos[2] or point[2] > obs_pos[2] + obstacle.height:
            return False

        return True

    def _update_descendants_cost(self, node: Node):
        """
        Recursively update costs of all descendants after rewiring.

        Args:
            node: Node whose descendants need cost update
        """
        for child in node.children:
            child.cost = node.cost + self._distance(node.position, child.position)
            self._update_descendants_cost(child)

    def _extract_path(self, goal_node: Node) -> List[np.ndarray]:
        """
        Extract path from start to goal by backtracking through parents.

        Args:
            goal_node: Goal node in tree

        Returns:
            List of waypoints from start to goal
        """
        path = []
        node = goal_node

        while node is not None:
            path.append(node.position.copy())
            node = node.parent

        # Reverse to get start -> goal
        path.reverse()
        return path

    def _smooth_path(self, path: List[np.ndarray], max_segment_length: float = 5.0) -> List[np.ndarray]:
        """
        Smooth path by removing unnecessary waypoints.

        Uses shortcutting: try to connect non-adjacent waypoints directly.
        Limits maximum segment length to ensure RL agent can follow accurately.
        Default 5.0m prevents drift on long straight segments.

        Args:
            path: Original path
            max_segment_length: Maximum distance between waypoints (meters, default 5.0)

        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            # Check if direct path is too long
            if len(path) == 2:
                dist = self._distance(path[0], path[1])
                if dist > max_segment_length:
                    # Add intermediate points
                    return self._add_intermediate_points(path[0], path[1], max_segment_length)
            return path

        smoothed = [path[0]]
        current_idx = 0

        while current_idx < len(path) - 1:
            # Try to connect to furthest visible waypoint
            best_idx = None

            for target_idx in range(len(path) - 1, current_idx, -1):
                if self._collision_free(path[current_idx], path[target_idx]):
                    # Check segment length
                    dist = self._distance(path[current_idx], path[target_idx])

                    if dist <= max_segment_length:
                        # Good segment length, use it
                        best_idx = target_idx
                        break
                    # If too long, continue searching for closer waypoint

            if best_idx is not None:
                smoothed.append(path[best_idx])
                current_idx = best_idx
            else:
                # No shortcut found, move to next waypoint
                current_idx += 1
                if current_idx < len(path):
                    # Check if segment to next waypoint is too long
                    dist = self._distance(path[current_idx - 1], path[current_idx])
                    if dist > max_segment_length:
                        # Add intermediate points
                        intermediate = self._add_intermediate_points(
                            path[current_idx - 1],
                            path[current_idx],
                            max_segment_length
                        )
                        # Add all except first (already in smoothed) and last (will be added below)
                        smoothed.extend(intermediate[1:-1])
                    smoothed.append(path[current_idx])

        return smoothed

    def _add_intermediate_points(self, start: np.ndarray, goal: np.ndarray,
                                  max_dist: float) -> List[np.ndarray]:
        """
        Add intermediate points along a straight line if distance is too large.

        Args:
            start: Start position
            goal: Goal position
            max_dist: Maximum distance between points

        Returns:
            List of points including start, intermediates, and goal
        """
        dist = self._distance(start, goal)

        if dist <= max_dist:
            return [start, goal]

        # Calculate number of intermediate points needed
        n_segments = int(np.ceil(dist / max_dist))
        points = [start]

        # Add intermediate points
        for i in range(1, n_segments):
            t = i / n_segments
            point = start + t * (goal - start)
            points.append(point)

        points.append(goal)
        return points

    def get_tree_edges(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get all edges in tree for visualization.

        Returns:
            List of (parent_pos, child_pos) tuples
        """
        edges = []
        for node in self.nodes:
            if node.parent is not None:
                edges.append((node.parent.position, node.position))
        return edges
