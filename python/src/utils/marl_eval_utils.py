import numpy as np


def get_pos_grid(x_scale: float, y_scale: float, z_scale: float, n: int):
    x_centers = np.linspace(-x_scale, x_scale, n, endpoint=False) + x_scale / n
    y_centers = np.linspace(-y_scale, y_scale, n, endpoint=False) + y_scale / n
    z_centers = np.linspace(-z_scale, z_scale, n, endpoint=False) + z_scale / n
    (x_coords, y_coords, z_coords) = np.meshgrid(
        x_centers, y_centers, z_centers, indexing="ij"
    )
    centers = np.vstack((x_coords.ravel(), y_coords.ravel(), z_coords.ravel())).T
    return centers
