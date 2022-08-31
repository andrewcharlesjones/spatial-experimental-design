import numpy as np
import sys
sys.path.append("../util/")
from util import get_points_near_line

class Tissue():

	def __init__(self, spatial_locations, readouts, slice_radius):

		assert len(spatial_locations) == len(readouts)

		self.X = spatial_locations
		self.Y = readouts
		self.slice_radius = slice_radius
		self.n_total, self.p = self.X.shape

		# Start with one tissue fragment
		self.X_fragment_idx = [np.arange(self.n_total)]

	def slice(self, design, fragment_num):
		# design should be of the form [b0, b1] or [b0, b1, b2]
		# b0 is the intercept and b1, b2 are slopes (depending on if
		# 2D or 3D)
		curr_fragment_idx = self.X_fragment_idx[fragment_num]
		fragment_X = self.X[curr_fragment_idx]

		# Split into fragments above and below slice
		above_fragment_idx = np.where(
			fragment_X[:, -1]
			>= design[0] + np.dot(fragment_X[:, :-1], design[1:])
		)[0]
		below_fragment_idx = np.where(
			fragment_X[:, -1]
			<= design[0] + np.dot(fragment_X[:, :-1], design[1:])
		)[0]

		above_idx = curr_fragment_idx[above_fragment_idx]
		below_idx = curr_fragment_idx[below_fragment_idx]
		self.X_fragment_idx.pop(fragment_num)
		self.X_fragment_idx.append(above_idx)
		self.X_fragment_idx.append(below_idx)

	def get_X_idx_near_slice(self, design, fragment_num):
		curr_fragment_idx = self.X_fragment_idx[fragment_num]
		fragment_X = self.X[curr_fragment_idx]

		intercept, slope = design
		dists = np.abs(-slope * fragment_X[:, 0] + fragment_X[:, 1] - intercept) / np.sqrt(slope ** 2 + 1)
		observed_idx = np.where(dists <= self.slice_radius)[0]
		return curr_fragment_idx[observed_idx]


