"""
Waypoint utilities for path planning.
Adapted from DRL-DroneNavigation project.
"""

import numpy as np


def dilate_waypoints(waypoints, factor: int) -> list:
    """
    Add intermediate waypoints between RRT* path points.

    This is a key technique from DRL-DroneNavigation that helps the RL agent
    by providing more frequent targets along the path.

    Args:
        waypoints: List of waypoint positions [(x,y,z), ...]
        factor: Number of intermediate points to add between each pair
                factor=0: no change
                factor=1: add 1 point between each pair (doubles waypoints)
                factor=2: add 2 points between each pair (triples waypoints)

    Returns:
        List of dilated waypoints

    Example:
        Original: [A, B, C]
        factor=1: [A, A_B_mid, B, B_C_mid, C]
        factor=2: [A, A_B_1/3, A_B_2/3, B, B_C_1/3, B_C_2/3, C]
    """
    if factor == 0:
        return waypoints

    dilated_points = []

    for i in range(len(waypoints) - 1):
        start_point = waypoints[i]
        end_point = waypoints[i + 1]

        # Generate intermediate points using linear interpolation
        # num=factor+2 creates: [start, intermediate_1, ..., intermediate_factor, end]
        intermediate_points = np.linspace(start_point, end_point, num=factor + 2)

        # Append all except the last point to avoid duplication
        dilated_points.extend(intermediate_points[:-1])

    # Add the final waypoint
    dilated_points.append(waypoints[-1])

    return dilated_points
