import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.metrics import pairwise_distances
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import r2_score
import seaborn as sns
from scipy.stats import multivariate_normal as mvn

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

n = 3
ntest = 400
lim = 5
noise_scale = 1e-8
X = np.random.uniform(low=-lim, high=lim, size=n).reshape(-1, 1)
# X = np.linspace(-lim, lim, n).reshape(-1, 1)
Xtest = np.linspace(-lim, lim, ntest).reshape(-1, 1)

kernel_true = RBF(length_scale=1)
K_XX = kernel_true(X) + noise_scale * np.eye(n)
Y = mvn.rvs(mean=np.zeros(n), cov=K_XX)

gp1 = GPR(
    kernel=RBF(length_scale=1, length_scale_bounds="fixed")
    + WhiteKernel(noise_level=1, noise_level_bounds="fixed")
)
gp1.fit(X, Y)
_, preds1_stddev = gp1.predict(Xtest, return_std=True)

gp2 = GPR(
    kernel=RBF(length_scale=2, length_scale_bounds="fixed")
    + WhiteKernel(noise_level=2, noise_level_bounds="fixed")
)
gp2.fit(X, Y)
_, preds2_stddev = gp2.predict(Xtest, return_std=True)


plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(Xtest, preds1_stddev)
plt.plot(Xtest, preds2_stddev)
# plt.show()

plt.subplot(122)
plt.scatter(np.argsort(preds1_stddev), np.argsort(preds2_stddev))
plt.show()
import ipdb

ipdb.set_trace()
