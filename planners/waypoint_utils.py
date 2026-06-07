import numpy as np

def dilate_waypoints(waypoints, factor: int) -> list:
    if factor == 0:
        return waypoints

    dilated_points = []

    for i in range(len(waypoints) - 1):
        start_point = waypoints[i]
        end_point = waypoints[i + 1]

        intermediate_points = np.linspace(start_point, end_point, num=factor + 2)

        dilated_points.extend(intermediate_points[:-1])

    dilated_points.append(waypoints[-1])

    return dilated_points
