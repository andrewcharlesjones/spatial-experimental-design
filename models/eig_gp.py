import numpy as np

def compute_eig_gp(cov, noise_variance):
	L = cov.shape[0]
	eig = (
	    0.5 * np.linalg.slogdet(1 / noise_variance * cov + np.eye(L))[1]
	)
	return eig