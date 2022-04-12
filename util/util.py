import numpy as onp

def get_points_near_line(slope, X, intercept=0, slice_radius=1.):
    dists = onp.abs(-slope * X[:, 0] + X[:, 1] - intercept) / onp.sqrt(slope ** 2 + 1)
    observed_idx = onp.where(dists <= slice_radius)[0]
    return observed_idx