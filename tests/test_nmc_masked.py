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
# Y[tumor_idx] = flip_list


slice_radius = 0.1

# Discretize design space
n_slope_discretizations = 10
n_intercept_discretizations = 10
slope_angles = onp.linspace(0, onp.pi, n_slope_discretizations)
slopes = onp.tan(slope_angles)
intercepts = onp.linspace(limits[0] + 2, limits[1] - 2, n_intercept_discretizations)
designs1, designs2 = onp.meshgrid(intercepts, slopes)
candidate_designs = onp.vstack([designs1.ravel(), designs2.ravel()]).T
n_candidate_designs = len(candidate_designs)

# cd1 = onp.linspace(limits[0], limits[1], 30)
# cd2 = onp.linspace(limits[0], limits[1], 40)
# Xs_for_designs = [onp.stack([cd1, onp.repeat(cc, 30)], axis=1) for cc in cd2]

# idx_for_designs = onp.random.choice(onp.arange(len(X)), size=200, replace=False)
# Xs_for_designs = X[idx_for_designs]
# Xs_for_designs = [x.reshape(1, -1) for x in Xs_for_designs]


Xs_for_designs = []
for ii, dd in enumerate(candidate_designs):
    curr_idx = get_points_near_line(
        slope=dd[1], X=X, intercept=dd[0], slice_radius=slice_radius
    )
    curr_X = X[curr_idx]
    Xs_for_designs.append(curr_X)

# observed_idx = onp.random.choice(onp.arange(len(X)), size=20, replace=False).tolist()
# observed_idx = onp.arange(len(X))
# observed_idx = []

best_designs = []
X_fragment_idx = [onp.arange(len(X))]

variational_model = ApproximateModel(key=PRNGKey(0))

eig_computer = EIGComputer(
    X_full=X,
    model=model,
    variational_model=variational_model,
)

for iternum in range(n_experimental_iters):

    assert len(best_designs) == iternum
    assert len(X_fragment_idx) == iternum + 1

    variational_model = ApproximateModel(key=PRNGKey(iternum))

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
    best_design_idx, best_fragment_idx, best_observed_idx = None, None, None

    for ff in range(len(X_fragment_idx)):

        # Get data for this fragment
        curr_X = X[X_fragment_idx[ff]]

        eigs = []
        # for dd in tqdm(Xs_for_designs):
        for dd in tqdm(range(len(candidate_designs))):

            # Get points that would be observed by this slice
            curr_design = candidate_designs[dd]

            above_fragment_idx = onp.where(
                curr_X[:, 1] >= curr_design[0] + curr_X[:, 0] * curr_design[1]
            )[0]
            if len(above_fragment_idx) in [
                0,
                1,
                2,
                len(curr_X),
                len(curr_X) - 1,
                len(curr_X) - 2,
            ]:
                continue

            curr_observed_idx = get_points_near_line(
                X=curr_X,
                slope=curr_design[1],
                intercept=curr_design[0],
                slice_radius=slice_radius,
            )
            if len(curr_observed_idx) == 0:
                continue

            design_mask = onp.zeros(len(X))
            design_mask[curr_observed_idx] = 1

            # curr_eig = nmc(
            #     # design=curr_X[curr_observed_idx],
            #     design=jnp.array(design_mask),
            #     variational_model=variational_model,
            #     variational_params=fitted_params,
            # )

            curr_eig = eig_computer.compute_eig(
                mask=design_mask,
                variational_params=fitted_params,
                key_int=iternum,
            )

            if curr_eig > best_eig:
                best_design_idx = dd
                best_fragment_idx = ff
                best_observed_idx = X_fragment_idx[ff][curr_observed_idx]
                best_eig = curr_eig

    curr_best_design = candidate_designs[best_design_idx]
    best_fragment_X = X[X_fragment_idx[best_fragment_idx]]

    print("Best EIG: ", best_eig)

    above_fragment_idx = onp.where(
        best_fragment_X[:, 1]
        >= curr_best_design[0] + best_fragment_X[:, 0] * curr_best_design[1]
    )[0]
    below_fragment_idx = onp.where(
        best_fragment_X[:, 1]
        <= curr_best_design[0] + best_fragment_X[:, 0] * curr_best_design[1]
    )[0]

    above_idx = X_fragment_idx[best_fragment_idx][above_fragment_idx]
    below_idx = X_fragment_idx[best_fragment_idx][below_fragment_idx]
    X_fragment_idx.pop(best_fragment_idx)
    X_fragment_idx.append(above_idx)
    X_fragment_idx.append(below_idx)

    best_designs.append(curr_best_design)
    observed_idx.extend(best_observed_idx)

plt.close()
fig, axs = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1, 1, 1]})

plt.sca(axs[0])
for c in [0, 1]:
    curr_idx = onp.where(Y == c)[0]
    plt.scatter(X[curr_idx, 0], X[curr_idx, 1], label="Healthy" if c == 0 else "Tumor")
plt.title("Data")
plt.axis("off")


radius_estimated = onp.exp(fitted_params[0] + 0.5 * jnp.exp(fitted_params[4]) ** 2)
slope_estimated = fitted_params[1]
center_estimated = fitted_params[2:4]
preds = model.predict(X, radius_estimated, slope_estimated, center_estimated)


intercepts_naive = onp.linspace(limits[0], limits[1], n_experimental_iters + 2)[1:-1]
slopes_naive = onp.zeros(n_experimental_iters)
candidate_designs_naive = onp.stack([intercepts_naive, slopes_naive]).T

observed_idx_naive = []

for dd in candidate_designs_naive:
    curr_observed_idx = get_points_near_line(
        X=X, slope=dd[1], intercept=dd[0], slice_radius=slice_radius
    )
    observed_idx_naive.extend(curr_observed_idx.tolist())

observed_idx_naive = onp.array(observed_idx_naive)


plt.sca(axs[1])
observed_idx = onp.array(observed_idx)

plt.scatter(X[:, 0], X[:, 1], color="gray", alpha=0.5)

for c in [0, 1]:
    curr_idx = onp.where(Y[observed_idx_naive] == c)[0]
    plt.scatter(
        X[observed_idx_naive][curr_idx, 0],
        X[observed_idx_naive][curr_idx, 1],
        label="Healthy" if c == 0 else "Tumor",
    )


plt.title("Naive slices")
plt.axis("off")


plt.sca(axs[2])
plt.scatter(X[:, 0], X[:, 1], color="gray", alpha=0.5)

for c in [0, 1]:
    curr_idx = onp.where(Y[observed_idx] == c)[0]
    plt.scatter(
        X[observed_idx][curr_idx, 0],
        X[observed_idx][curr_idx, 1],
        label="Healthy" if c == 0 else "Tumor",
    )

plt.title("Designed slices")
plt.axis("off")

# plt.xlim(limits)
# plt.ylim(limits)
plt.title("Slices")
plt.axis("off")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
plt.close()

# import ipdb; ipdb.set_trace()
fitted_params_naive = fit_variational_model(
    X=X[onp.unique(onp.array(observed_idx_naive))],
    Y=Y[onp.unique(onp.array(observed_idx_naive))],
    model_object=model,
)

radius_estimated = onp.exp(
    fitted_params_naive[0] + 0.5 * jnp.exp(fitted_params_naive[4]) ** 2
)
slope_estimated = fitted_params_naive[1]
center_estimated = fitted_params[2:4]
preds_naive = model.predict(X, radius_estimated, slope_estimated, center_estimated)

preds_int = (preds > 0.5).astype(int)
preds_naive_int = (preds_naive > 0.5).astype(int)
accuracy = (preds_int == Y).mean()
accuracy_naive = (preds_naive_int == Y).mean()

print("Acc:", accuracy)
print("Acc, naive:", accuracy_naive)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=preds)
plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=preds_naive)
plt.show()
import ipdb

ipdb.set_trace()
