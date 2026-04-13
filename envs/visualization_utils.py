"""
Utility functions for drawing arena boundaries and markers in PyBullet.
"""

import pybullet as p
import config


def draw_arena_boundaries(client_id):
    """
    Draw arena boundaries as transparent walls.

    Args:
        client_id: PyBullet physics client ID
    """
    half_x = config.ARENA_SIZE_X / 2
    half_y = config.ARENA_SIZE_Y / 2
    height = config.ARENA_HEIGHT

    # Draw vertical edges (corners)
    corners = [
        [-half_x, -half_y],
        [half_x, -half_y],
        [half_x, half_y],
        [-half_x, half_y]
    ]

    # Draw vertical lines at corners (thicker)
    for corner in corners:
        p.addUserDebugLine(
            [corner[0], corner[1], 0],
            [corner[0], corner[1], height],
            [0.5, 0.5, 0.5],  # Gray
            5,  # Thicker
            0,
            physicsClientId=client_id
        )

    # Draw horizontal edges at bottom (thicker)
    for i in range(4):
        next_i = (i + 1) % 4
        p.addUserDebugLine(
            [corners[i][0], corners[i][1], 0],
            [corners[next_i][0], corners[next_i][1], 0],
            [0.5, 0.5, 0.5],
            5,  # Thicker
            0,
            physicsClientId=client_id
        )

    # Draw horizontal edges at top (thicker)
    for i in range(4):
        next_i = (i + 1) % 4
        p.addUserDebugLine(
            [corners[i][0], corners[i][1], height],
            [corners[next_i][0], corners[next_i][1], height],
            [0.5, 0.5, 0.5],
            5,  # Thicker
            0,
            physicsClientId=client_id
        )

    # Draw grid on floor (more lines, thicker)
    grid_step = 1.0  # Smaller step = more lines

    # Lines parallel to X axis
    y = -half_y
    while y <= half_y:
        p.addUserDebugLine(
            [-half_x, y, 0],
            [half_x, y, 0],
            [0.3, 0.3, 0.3],
            2,  # Thicker
            0,
            physicsClientId=client_id
        )
        y += grid_step

    # Lines parallel to Y axis
    x = -half_x
    while x <= half_x:
        p.addUserDebugLine(
            [x, -half_y, 0],
            [x, half_y, 0],
            [0.3, 0.3, 0.3],
            2,  # Thicker
            0,
            physicsClientId=client_id
        )
        x += grid_step


def draw_goal_marker(goal_pos, client_id):
    """
    Draw goal marker as green cross.

    Args:
        goal_pos: [x, y, z] goal position
        client_id: PyBullet physics client ID
    """
    p.addUserDebugLine(
        [goal_pos[0] - 0.2, goal_pos[1], goal_pos[2]],
        [goal_pos[0] + 0.2, goal_pos[1], goal_pos[2]],
        [0, 1, 0], 3, 0,
        physicsClientId=client_id
    )
    p.addUserDebugLine(
        [goal_pos[0], goal_pos[1] - 0.2, goal_pos[2]],
        [goal_pos[0], goal_pos[1] + 0.2, goal_pos[2]],
        [0, 1, 0], 3, 0,
        physicsClientId=client_id
    )
    p.addUserDebugLine(
        [goal_pos[0], goal_pos[1], goal_pos[2] - 0.2],
        [goal_pos[0], goal_pos[1], goal_pos[2] + 0.2],
        [0, 1, 0], 3, 0,
        physicsClientId=client_id
    )
