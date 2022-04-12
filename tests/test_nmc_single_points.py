import numpy as onp
import jax.numpy as jnp
from jax.random import PRNGKey
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys

sys.path.append("../models")
from spherical_classification import Model, ApproximateModel, fit_variational_model
from nmc import EIGComputer

sys.path.append("../util")
from util import get_points_near_line

prior_mean = 0.0
prior_variance = 1.0

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


model = Model(prior_mean=prior_mean, prior_stddev=onp.sqrt(prior_variance))


n_experimental_iters = 1
limits = [-3, 3]
grid_size = 50
radius = 1.5
center = onp.array([0, 0])
x1s = onp.linspace(*limits, num=grid_size)
x2s = onp.linspace(*limits, num=grid_size)
X1, X2 = onp.meshgrid(x1s, x2s)
X = onp.vstack([X1.ravel(), X2.ravel()]).T
norms = onp.linalg.norm(X - center, axis=1)
Y = (norms < radius).astype(int)


# Randomly flip some of the tumor cells to be healthy
tumor_idx = onp.where(Y == 1)[0]
flip_list = onp.random.binomial(n=1, p=0.8, size=len(tumor_idx))
Y[tumor_idx] = flip_list


idx_for_designs = onp.arange(len(X))


observed_idx = onp.random.choice(onp.arange(len(X)), size=100, replace=False).tolist()
# observed_idx = onp.arange(len(X))
# observed_idx = []

variational_model = ApproximateModel(key=PRNGKey(0))
eig_computer = EIGComputer(
    X_full=X,
    model=model,
    variational_model=variational_model,
)


if len(observed_idx) == 0:
    fitted_params = jnp.zeros(8) - 1
else:
    fitted_params = fit_variational_model(
        X=X[onp.unique(onp.array(observed_idx))],
        Y=Y[onp.unique(onp.array(observed_idx))],
        model_object=model,
    )

radius_mean = jnp.exp(fitted_params[0] + 0.5 * jnp.exp(fitted_params[4]) ** 2)
print(radius_mean)
print("Center: ", fitted_params[2:4])

best_eig = -onp.inf

# Get data for this fragment

eigs = []
for dd in tqdm(idx_for_designs):

    curr_eig = eig_computer.compute_eig(
        X=X[dd].reshape(1, -1),
        variational_params=fitted_params,
        key_int=0,
    )
    eigs.append(curr_eig)

    if curr_eig > best_eig:
        best_observed_idx = dd
        best_eig = curr_eig

observed_idx.append(best_observed_idx)

plt.close()
fig, axs = plt.subplots(
    1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1, 1, 1.2]}
)
plt.sca(axs[0])
plt.title("Data")
for c in [0, 1]:
    curr_idx = onp.where(Y == c)[0]
    plt.scatter(X[curr_idx, 0], X[curr_idx, 1], label="Healthy" if c == 0 else "Tumor")

plt.legend(loc="center right", bbox_to_anchor=(0, 0.5))
plt.axis("off")
plt.tight_layout()

plt.sca(axs[1])
plt.title("Observations")
plt.scatter(X[:, 0], X[:, 1], c="gray", alpha=0.5)

for c in [0, 1]:
    curr_idx = onp.where(Y[onp.array(observed_idx)] == c)[0]
    plt.scatter(
        X[onp.array(observed_idx)[curr_idx], 0],
        X[onp.array(observed_idx)[curr_idx], 1],
    )

plt.axis("off")
plt.tight_layout()

plt.sca(axs[2])
plt.title("Point-wise EIG")
plt.scatter(X[idx_for_designs, 0], X[idx_for_designs, 1], c=eigs)
plt.colorbar()
plt.axis("off")


plt.show()
import ipdb

ipdb.set_trace()
