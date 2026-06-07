import pybullet as p


def draw_arena_boundaries(client_id, arena_size_x, arena_size_y, arena_height):
    half_x = arena_size_x / 2
    half_y = arena_size_y / 2
    height = arena_height


    corners = [
        [-half_x, -half_y],
        [half_x, -half_y],
        [half_x, half_y],
        [-half_x, half_y]
    ]

    for corner in corners:
        p.addUserDebugLine(
            [corner[0], corner[1], 0],
            [corner[0], corner[1], height],
            [0.5, 0.5, 0.5],
            5,
            0,
            physicsClientId=client_id
        )

    for i in range(4):
        next_i = (i + 1) % 4
        p.addUserDebugLine(
            [corners[i][0], corners[i][1], 0],
            [corners[next_i][0], corners[next_i][1], 0],
            [0.5, 0.5, 0.5],
            5,
            0,
            physicsClientId=client_id
        )

    for i in range(4):
        next_i = (i + 1) % 4
        p.addUserDebugLine(
            [corners[i][0], corners[i][1], height],
            [corners[next_i][0], corners[next_i][1], height],
            [0.5, 0.5, 0.5],
            5,
            0,
            physicsClientId=client_id
        )

    grid_step = 1.0

    y = -half_y
    while y <= half_y:
        p.addUserDebugLine(
            [-half_x, y, 0],
            [half_x, y, 0],
            [0.3, 0.3, 0.3],
            2,
            0,
            physicsClientId=client_id
        )
        y += grid_step

    x = -half_x
    while x <= half_x:
        p.addUserDebugLine(
            [x, -half_y, 0],
            [x, half_y, 0],
            [0.3, 0.3, 0.3],
            2,
            0,
            physicsClientId=client_id
        )
        x += grid_step


def draw_goal_marker(goal_pos, client_id):
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