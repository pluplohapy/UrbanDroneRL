import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple
import config

def visualize_rrt_tree_3d(
    planner,
    path: List[np.ndarray] = None,
    start: np.ndarray = None,
    goal: np.ndarray = None,
    obstacles: List = None,
    save_path: str = None
):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(-config.ARENA_SIZE_X/2, config.ARENA_SIZE_X/2)
    ax.set_ylim(-config.ARENA_SIZE_Y/2, config.ARENA_SIZE_Y/2)
    ax.set_zlim(0, config.ARENA_HEIGHT)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('RRT* Tree and Path')

    if obstacles:
        for obs in obstacles:
            pos = obs.get_position()
            radius = obs.radius
            height = obs.height

            theta = np.linspace(0, 2*np.pi, 30)
            z_cyl = np.linspace(pos[2], pos[2] + height, 20)
            theta_grid, z_grid = np.meshgrid(theta, z_cyl)
            x_cyl = pos[0] + radius * np.cos(theta_grid)
            y_cyl = pos[1] + radius * np.sin(theta_grid)

            ax.plot_surface(x_cyl, y_cyl, z_grid, alpha=0.3, color='gray')

    edges = planner.get_tree_edges()
    for parent_pos, child_pos in edges:
        ax.plot([parent_pos[0], child_pos[0]],
                [parent_pos[1], child_pos[1]],
                [parent_pos[2], child_pos[2]],
                'b-', alpha=0.2, linewidth=0.5)

    if len(planner.nodes) > 0:
        positions = np.array([node.position for node in planner.nodes])
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c='blue', s=5, alpha=0.3, label='Tree nodes')

    if path is not None and len(path) > 0:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2],
                'r-', linewidth=3, label='Path', zorder=10)
        ax.scatter(path_array[:, 0], path_array[:, 1], path_array[:, 2],
                   c='red', s=50, zorder=11)

    if start is not None:
        ax.scatter([start[0]], [start[1]], [start[2]],
                   c='green', s=200, marker='o', label='Start', zorder=12)

    if goal is not None:
        ax.scatter([goal[0]], [goal[1]], [goal[2]],
                   c='orange', s=200, marker='*', label='Goal', zorder=12)

    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Viz] Saved to {save_path}")

    plt.show()

def visualize_rrt_tree_2d(
    planner,
    path: List[np.ndarray] = None,
    start: np.ndarray = None,
    goal: np.ndarray = None,
    obstacles: List = None,
    view: str = 'xy',
    save_path: str = None
):
    fig, ax = plt.subplots(figsize=(10, 10))

    if view == 'xy':
        idx1, idx2 = 0, 1
        label1, label2 = 'X (m)', 'Y (m)'
        ax.set_xlim(-config.ARENA_SIZE_X/2, config.ARENA_SIZE_X/2)
        ax.set_ylim(-config.ARENA_SIZE_Y/2, config.ARENA_SIZE_Y/2)
    elif view == 'xz':
        idx1, idx2 = 0, 2
        label1, label2 = 'X (m)', 'Z (m)'
        ax.set_xlim(-config.ARENA_SIZE_X/2, config.ARENA_SIZE_X/2)
        ax.set_ylim(0, config.ARENA_HEIGHT)
    elif view == 'yz':
        idx1, idx2 = 1, 2
        label1, label2 = 'Y (m)', 'Z (m)'
        ax.set_xlim(-config.ARENA_SIZE_Y/2, config.ARENA_SIZE_Y/2)
        ax.set_ylim(0, config.ARENA_HEIGHT)
    else:
        raise ValueError(f"Invalid view: {view}")

    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    ax.set_title(f'RRT* Tree and Path ({view.upper()} view)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if obstacles:
        for obs in obstacles:
            pos = obs.get_position()
            radius = obs.radius

            if view == 'xy':

                circle = plt.Circle((pos[0], pos[1]), radius, color='gray', alpha=0.5)
                ax.add_patch(circle)
            else:

                if view == 'xz':
                    rect = plt.Rectangle((pos[0] - radius, pos[2]),
                                          2*radius, obs.height,
                                          color='gray', alpha=0.5)
                else:
                    rect = plt.Rectangle((pos[1] - radius, pos[2]),
                                          2*radius, obs.height,
                                          color='gray', alpha=0.5)
                ax.add_patch(rect)

    edges = planner.get_tree_edges()
    for parent_pos, child_pos in edges:
        ax.plot([parent_pos[idx1], child_pos[idx1]],
                [parent_pos[idx2], child_pos[idx2]],
                'b-', alpha=0.2, linewidth=0.5)

    if len(planner.nodes) > 0:
        positions = np.array([node.position for node in planner.nodes])
        ax.scatter(positions[:, idx1], positions[:, idx2],
                   c='blue', s=10, alpha=0.3, label='Tree nodes')

    if path is not None and len(path) > 0:
        path_array = np.array(path)
        ax.plot(path_array[:, idx1], path_array[:, idx2],
                'r-', linewidth=3, label='Path', zorder=10)
        ax.scatter(path_array[:, idx1], path_array[:, idx2],
                   c='red', s=50, zorder=11)

    if start is not None:
        ax.scatter([start[idx1]], [start[idx2]],
                   c='green', s=200, marker='o', label='Start', zorder=12)

    if goal is not None:
        ax.scatter([goal[idx1]], [goal[idx2]],
                   c='orange', s=200, marker='*', label='Goal', zorder=12)

    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Viz] Saved to {save_path}")

    plt.show()

def plot_path_comparison(
    paths: List[Tuple[str, List[np.ndarray]]],
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: List = None,
    save_path: str = None
):
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_xlim(-config.ARENA_SIZE_X/2, config.ARENA_SIZE_X/2)
    ax.set_ylim(-config.ARENA_SIZE_Y/2, config.ARENA_SIZE_Y/2)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Path Comparison')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if obstacles:
        for obs in obstacles:
            pos = obs.get_position()
            radius = obs.radius
            circle = plt.Circle((pos[0], pos[1]), radius, color='gray', alpha=0.5)
            ax.add_patch(circle)

    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for i, (name, path) in enumerate(paths):
        if path is not None and len(path) > 0:
            path_array = np.array(path)
            color = colors[i % len(colors)]
            ax.plot(path_array[:, 0], path_array[:, 1],
                    '-', linewidth=2, label=name, color=color)
            ax.scatter(path_array[:, 0], path_array[:, 1],
                       s=30, color=color)

    ax.scatter([start[0]], [start[1]],
               c='green', s=200, marker='o', label='Start', zorder=12)
    ax.scatter([goal[0]], [goal[1]],
               c='orange', s=200, marker='*', label='Goal', zorder=12)

    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Viz] Saved to {save_path}")

    plt.show()

def plot_planning_stats(planners_data: List[Tuple[str, dict]]):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    names = [name for name, _ in planners_data]
    times = [data['time'] for _, data in planners_data]
    costs = [data['cost'] for _, data in planners_data]
    nodes = [data['nodes'] for _, data in planners_data]
    waypoints = [data['waypoints'] for _, data in planners_data]

    axes[0, 0].bar(names, times)
    axes[0, 0].set_ylabel('Time (s)')
    axes[0, 0].set_title('Planning Time')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].bar(names, costs)
    axes[0, 1].set_ylabel('Cost (m)')
    axes[0, 1].set_title('Path Cost')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].bar(names, nodes)
    axes[1, 0].set_ylabel('Nodes')
    axes[1, 0].set_title('Tree Nodes')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].bar(names, waypoints)
    axes[1, 1].set_ylabel('Waypoints')
    axes[1, 1].set_title('Path Waypoints')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
