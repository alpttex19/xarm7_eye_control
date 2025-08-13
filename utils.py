import numpy as np


def linear_interpolation_3d(start_pos, end_pos, num_points=100):
    """
    三维线性插值

    Args:
        start_pos: 起始位置 [x, y, z]
        end_pos: 结束位置 [x, y, z]
        t: 插值参数 [0, 1]

    Returns:
        interpolated_pos: 插值后的位置
    """
    trajectory_points = []

    for i in range(num_points):
        t = i / (num_points - 1)  # 参数从0到1

        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)
        point = start_pos + t * (end_pos - start_pos)

        trajectory_points.append(point)

    return np.array(trajectory_points)
