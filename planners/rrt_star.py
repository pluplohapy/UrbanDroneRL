
import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial import KDTree
import time

class Node:

    def __init__(self, position: np.ndarray, parent: Optional['Node'] = None):
        self.position = np.array(position, dtype=np.float32)
        self.parent = parent
        self.cost = 0.0
        self.children = []

    def __repr__(self):
        return f"Node(pos={self.position}, cost={self.cost:.2f})"

class RRTStarPlanner:

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
        self.arena_bounds = arena_bounds
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.goal_threshold = goal_threshold
        self.rewire_radius = rewire_radius
        self.collision_check_resolution = collision_check_resolution
        self.verbose = verbose

        self.nodes = []
        self.kdtree = None

        self.obstacles = []

        self.planning_time = 0.0
        self.iterations_used = 0
        self.path_cost = 0.0

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List
    ) -> Optional[List[np.ndarray]]:
        start_time = time.time()

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

        for iteration in range(self.max_iter):

            if np.random.random() < self.goal_bias:
                sample = goal
            else:
                sample = self._sample_free()

            nearest_node = self._nearest(sample)

            new_pos = self._steer(nearest_node.position, sample)

            if not self._collision_free(nearest_node.position, new_pos):
                continue

            neighbors = self._near(new_pos)

            min_cost = nearest_node.cost + self._distance(nearest_node.position, new_pos)
            best_parent = nearest_node

            for neighbor in neighbors:
                if self._collision_free(neighbor.position, new_pos):
                    cost = neighbor.cost + self._distance(neighbor.position, new_pos)
                    if cost < min_cost:
                        min_cost = cost
                        best_parent = neighbor

            new_node = Node(new_pos, parent=best_parent)
            new_node.cost = min_cost
            best_parent.children.append(new_node)
            self.nodes.append(new_node)

            if len(self.nodes) % 100 == 0:
                self._rebuild_kdtree()

            for neighbor in neighbors:
                new_cost = new_node.cost + self._distance(new_node.position, neighbor.position)
                if new_cost < neighbor.cost:
                    if self._collision_free(new_node.position, neighbor.position):

                        if neighbor.parent:
                            neighbor.parent.children.remove(neighbor)

                        neighbor.parent = new_node
                        neighbor.cost = new_cost
                        new_node.children.append(neighbor)

                        self._update_descendants_cost(neighbor)

            dist_to_goal = self._distance(new_pos, goal)
            if dist_to_goal < self.goal_threshold:
                if new_node.cost < best_goal_cost:
                    best_goal_cost = new_node.cost
                    best_goal_node = new_node
                    goal_node = new_node
                    if self.verbose >= 2:
                        print(f"[RRT*] Goal reached at iteration {iteration+1}, cost: {best_goal_cost:.2f}")

            if self.verbose >= 2 and (iteration + 1) % 500 == 0:
                best_cost_str = f"{best_goal_cost:.2f}" if best_goal_node else "inf"
                print(f"[RRT*] Iteration {iteration+1}/{self.max_iter}, nodes: {len(self.nodes)}, best_cost: {best_cost_str}")

        self.planning_time = time.time() - start_time
        self.iterations_used = self.max_iter

        if best_goal_node is not None:
            path = self._extract_path(best_goal_node)
            self.path_cost = best_goal_node.cost

            if self.verbose >= 1:
                print(f"[RRT*] ✓ Path: {len(path)} waypoints, cost: {self.path_cost:.2f}m, time: {self.planning_time:.2f}s")

            if self.verbose >= 2:
                print(f"[RRT*]   Tree nodes: {len(self.nodes)}")

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
        x = np.random.uniform(-self.arena_bounds[0]/2, self.arena_bounds[0]/2)
        y = np.random.uniform(-self.arena_bounds[1]/2, self.arena_bounds[1]/2)
        z = np.random.uniform(0.5, self.arena_bounds[2] - 0.5)
        return np.array([x, y, z])

    def _nearest(self, point: np.ndarray) -> Node:
        if self.kdtree is None or len(self.nodes) % 10 == 1:
            self._rebuild_kdtree()

        if len(self.nodes) == 1:
            return self.nodes[0]

        _, idx = self.kdtree.query(point)
        return self.nodes[idx]

    def _near(self, point: np.ndarray) -> List[Node]:
        if self.kdtree is None:
            self._rebuild_kdtree()

        if len(self.nodes) == 1:
            return []

        indices = self.kdtree.query_ball_point(point, self.rewire_radius)
        return [self.nodes[i] for i in indices]

    def _rebuild_kdtree(self):
        if len(self.nodes) > 0:
            positions = np.array([node.position for node in self.nodes])
            self.kdtree = KDTree(positions)

    def _steer(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)

        if distance <= self.step_size:
            return to_pos
        else:
            return from_pos + (direction / distance) * self.step_size

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return np.linalg.norm(pos2 - pos1)

    def _collision_free(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:

        distance = self._distance(pos1, pos2)
        n_checks = max(2, int(np.ceil(distance / self.collision_check_resolution)))

        if len(self.obstacles) > 0:
            for obstacle in self.obstacles:
                if self._point_in_cylinder(pos1, obstacle) or self._point_in_cylinder(pos2, obstacle):
                    return False

        for i in range(1, n_checks):
            t = i / n_checks
            point = pos1 + t * (pos2 - pos1)

            for obstacle in self.obstacles:
                if self._point_in_cylinder(point, obstacle):
                    return False

        return True

    def _point_in_cylinder(self, point: np.ndarray, obstacle) -> bool:
        obs_pos = obstacle.get_position()

        horizontal_dist = np.linalg.norm(point[:2] - obs_pos[:2])
        if horizontal_dist > obstacle.radius:
            return False

        if point[2] < obs_pos[2] or point[2] > obs_pos[2] + obstacle.height:
            return False

        return True

    def _update_descendants_cost(self, node: Node):
        for child in node.children:
            child.cost = node.cost + self._distance(node.position, child.position)
            self._update_descendants_cost(child)

    def _extract_path(self, goal_node: Node) -> List[np.ndarray]:
        path = []
        node = goal_node

        while node is not None:
            path.append(node.position.copy())
            node = node.parent

        path.reverse()
        return path

    def _smooth_path(self, path: List[np.ndarray], max_segment_length: float = 5.0) -> List[np.ndarray]:
        if len(path) <= 2:

            if len(path) == 2:
                dist = self._distance(path[0], path[1])
                if dist > max_segment_length:

                    return self._add_intermediate_points(path[0], path[1], max_segment_length)
            return path

        smoothed = [path[0]]
        current_idx = 0

        while current_idx < len(path) - 1:

            best_idx = None

            for target_idx in range(len(path) - 1, current_idx, -1):
                if self._collision_free(path[current_idx], path[target_idx]):

                    dist = self._distance(path[current_idx], path[target_idx])

                    if dist <= max_segment_length:

                        best_idx = target_idx
                        break

            if best_idx is not None:
                smoothed.append(path[best_idx])
                current_idx = best_idx
            else:

                current_idx += 1
                if current_idx < len(path):

                    dist = self._distance(path[current_idx - 1], path[current_idx])
                    if dist > max_segment_length:

                        intermediate = self._add_intermediate_points(
                            path[current_idx - 1],
                            path[current_idx],
                            max_segment_length
                        )

                        smoothed.extend(intermediate[1:-1])
                    smoothed.append(path[current_idx])

        return smoothed

    def _add_intermediate_points(self, start: np.ndarray, goal: np.ndarray,
                                  max_dist: float) -> List[np.ndarray]:
        dist = self._distance(start, goal)

        if dist <= max_dist:
            return [start, goal]

        n_segments = int(np.ceil(dist / max_dist))
        points = [start]

        for i in range(1, n_segments):
            t = i / n_segments
            point = start + t * (goal - start)
            points.append(point)

        points.append(goal)
        return points

    def get_tree_edges(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        edges = []
        for node in self.nodes:
            if node.parent is not None:
                edges.append((node.parent.position, node.position))
        return edges
