import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
from scipy.stats import multivariate_normal as mvn
from sklearn.metrics import pairwise_distances
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.metrics import r2_score
import seaborn as sns
from scipy.spatial import ConvexHull

import sys

sys.path.append("../../models/")
sys.path.append("../../util/")
from tissue import Tissue
from eig_gp import compute_eig_gp

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

sys.path.append("../../util/")
from util import get_points_near_line


# Create synthetic tissue
radius = 5
center_val = 4

center1 = np.array([-center_val, -center_val])
center2 = np.array([center_val, -center_val])
center3 = np.array([0, -center_val + np.sqrt((center_val * 2) ** 2 - center_val ** 2)])


limits = [-10, 10]

# When we take a slice, all points within this
# distance to the line/plane are "observed"
slice_radius = 0.2

n_iters = 8
n_repeats = 5
n_total_points = 2_000

r2_scores = np.zeros((n_repeats, n_iters))
r2_scores_naive = np.zeros((n_repeats, n_iters))
r2_scores_random = np.zeros((n_repeats, n_iters))

for rep_ii in range(n_repeats):
    X = np.random.uniform(low=limits[0], high=limits[1], size=(n_total_points, 2))
    # grid_size = 30
    # x1s = np.linspace(*limits, num=grid_size)
    # x2s = np.linspace(*limits, num=grid_size)
    # X1, X2 = np.meshgrid(x1s, x2s)
    # X = np.vstack([X1.ravel(), X2.ravel()]).T

    # Filter by radius
    in_circle1 = np.linalg.norm(X - center1, ord=2, axis=1) <= radius
    in_circle2 = np.linalg.norm(X - center2, ord=2, axis=1) <= radius
    in_circle3 = np.linalg.norm(X - center3, ord=2, axis=1) <= radius

    # in_circle3[np.random.choice(np.where(in_circle3)[0], size=int(len(np.where(in_circle3)[0]) / 1.2), replace=False)] = False
    # in_circle2[np.random.choice(np.where(in_circle2)[0], size=int(len(np.where(in_circle2)[0]) / 1.2), replace=False)] = False


    in_tissue = np.logical_or(in_circle1, in_circle2)
    in_tissue = np.logical_or(in_tissue, in_circle3)
    X = X[in_tissue]

    # Generate response
    Y = mvn.rvs(mean=np.zeros(X.shape[0]), cov=RBF(length_scale=2.)(X) + 1e-2 * np.eye(len(X)))

    # plt.scatter(X[:, 0], X[:, 1], c=Y)
    # plt.show()
    # import ipdb; ipdb.set_trace()

    ## BOED

    tissue = Tissue(spatial_locations=X, readouts=Y, slice_radius=slice_radius)

    # Discretize design space
    n_slope_discretizations = 20
    n_intercept_discretizations = 20
    slope_angles = np.linspace(0, np.pi, n_slope_discretizations)
    slopes = np.tan(slope_angles)
    intercepts = np.linspace(
        np.min(X[:, 0]) + 1, np.max(X[:, 0]) - 1, n_intercept_discretizations
    )
    designs1, designs2 = np.meshgrid(intercepts, slopes)
    candidate_designs = np.vstack([designs1.ravel(), designs2.ravel()]).T

    n_candidate_designs = len(candidate_designs)

    designs = []
    observed_idx = []
    X_fragment_idx = [np.arange(len(X))]

    for iternum in range(n_iters):

        assert len(designs) == iternum
        assert len(tissue.X_fragment_idx) == iternum + 1

        if len(observed_idx) > 0:
            kernel = RBF() + WhiteKernel()
            gpr = GPR(kernel=kernel)
            gpr.fit(
                X[np.unique(observed_idx)],
                Y[np.unique(observed_idx)],
            )

        best_eig = -np.inf
        best_design_idx, best_fragment_idx, best_observed_idx = None, None, None

        for ff in range(len(X_fragment_idx)):

            # Get data for this fragment
            curr_X = X[X_fragment_idx[ff]]

            for dd in range(n_candidate_designs):

                # Get points that would be observed by this slice
                curr_design = candidate_designs[dd]
                curr_observed_idx = tissue.get_X_idx_near_slice(
                    design=curr_design, fragment_num=ff
                )

                L = len(curr_observed_idx)
                if L == 0:
                    continue

                if iternum == 0:
                    kernel = RBF() + WhiteKernel()
                    cov = kernel(X[curr_observed_idx])
                    noise_variance = np.exp(kernel.k2.theta[0])
                else:
                    # Make predictions of expression
                    _, cov = gpr.predict(X[curr_observed_idx], return_cov=True)

                    noise_variance = np.exp(gpr.kernel_.k2.theta[0])

                curr_eig = compute_eig_gp(cov=cov, noise_variance=noise_variance)

                if curr_eig > best_eig:
                    best_design_idx = dd
                    best_fragment_idx = ff
                    best_observed_idx = curr_observed_idx
                    best_eig = curr_eig

        curr_best_design = candidate_designs[best_design_idx]

        tissue.slice(curr_best_design, best_fragment_idx)

        designs.append(curr_best_design)
        observed_idx.extend(best_observed_idx)

        ## Run prediction
        train_idx = np.array(observed_idx)
        test_idx = np.setdiff1d(np.arange(len(X)), train_idx)
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        gpr = GPR(kernel=kernel)
        gpr.fit(X_train, Y_train)
        preds = gpr.predict(X_test)
        curr_r2 = r2_score(Y_test, preds)
        r2_scores[rep_ii, iternum] = curr_r2

    ## Random design

    tissue = Tissue(spatial_locations=X, readouts=Y, slice_radius=slice_radius)
    designs_random = []
    observed_idx_random = []

    for iternum in range(n_iters):

        assert len(designs_random) == iternum
        assert len(tissue.X_fragment_idx) == iternum + 1

        curr_observed_idx = []
        while len(curr_observed_idx) == 0:
            rand_fragment_idx = np.random.choice(len(tissue.X_fragment_idx))
            rand_design_idx = np.random.choice(np.arange(n_candidate_designs))

            curr_design = candidate_designs[rand_design_idx]

            curr_observed_idx = tissue.get_X_idx_near_slice(
                design=curr_design, fragment_num=rand_fragment_idx
            )

        tissue.slice(curr_design, rand_fragment_idx)

        observed_idx_random.extend(
            curr_observed_idx.tolist()
        )

        designs_random.append(curr_design)

        train_idx = np.array(observed_idx_random)
        test_idx = np.setdiff1d(np.arange(len(X)), train_idx)
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        kernel = RBF() + WhiteKernel()
        gpr = GPR(kernel=kernel)
        gpr.fit(X_train, Y_train)
        preds = gpr.predict(X_test)
        curr_r2 = r2_score(Y_test, preds)
        r2_scores_random[rep_ii, iternum] = curr_r2

    ## Naive design

    slopes = np.zeros(n_iters)
    intercepts = np.linspace(X.min(0)[1] + 1, X.max(0)[1] - 1, n_iters)
    designs_naive = np.vstack([intercepts, slopes]).T

    tissue = Tissue(spatial_locations=X, readouts=Y, slice_radius=slice_radius)

    shuffled_idx = np.random.permutation(np.arange(len(designs_naive)))
    designs_naive_shuffled = designs_naive[shuffled_idx]

    observed_idx_naive = []

    for iternum in range(n_iters):

        # We don't slice here because we're guaranteed no intersecting slices (all are parallel)
        assert len(tissue.X_fragment_idx) == 1

        curr_design = designs_naive_shuffled[iternum]

        curr_observed_idx = tissue.get_X_idx_near_slice(
            design=curr_design, fragment_num=0
        )
        observed_idx_naive.extend(curr_observed_idx.tolist())

        train_idx = np.array(observed_idx_naive)
        test_idx = np.setdiff1d(np.arange(len(X)), train_idx)
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        kernel = RBF() + WhiteKernel()
        gpr = GPR(kernel=kernel)
        gpr.fit(X_train, Y_train)
        preds = gpr.predict(X_test)
        curr_r2 = r2_score(Y_test, preds)
        r2_scores_naive[rep_ii, iternum] = curr_r2


# plt.figure(figsize=(14, 5))
fig, ax = plt.subplots(
    1, 4, figsize=(24, 5), gridspec_kw={"width_ratios": [1, 1, 1, 1.3]}
)

plt.sca(ax[0])
plt.scatter(X[:, 0], X[:, 1], c="gray", alpha=0.5)
plt.scatter(
    X[np.array(observed_idx_random), 0],
    X[np.array(observed_idx_random), 1],
    label="Random",
    color="orange",
)
plt.axis("off")
plt.title("Random")

plt.sca(ax[1])
plt.scatter(X[:, 0], X[:, 1], c="gray", alpha=0.5)
plt.scatter(
    X[np.array(observed_idx_naive), 0],
    X[np.array(observed_idx_naive), 1],
    label="Naive",
    color="green",
)
plt.axis("off")
plt.title("Naive")
plt.legend([], [], frameon=False)
plt.tight_layout()

plt.sca(ax[2])
plt.scatter(X[:, 0], X[:, 1], c="gray", alpha=0.5)
plt.scatter(
    X[np.array(observed_idx), 0],
    X[np.array(observed_idx), 1],
    label="Optimal",
    color="blue",
)
plt.axis("off")
plt.title("Optimal")


plt.sca(ax[3])

results_df = pd.melt(pd.DataFrame(r2_scores))
results_df["variable"] += 1
results_df_naive = pd.melt(pd.DataFrame(r2_scores_naive))
results_df_naive["variable"] += 1
results_df_random = pd.melt(pd.DataFrame(r2_scores_random))
results_df_random["variable"] += 1

sns.lineplot(
    data=results_df,
    x="variable",
    y="value",
    label="Designed",
    color="blue",
)
sns.lineplot(
    data=results_df_naive,
    x="variable",
    y="value",
    label="Naive",
    color="green",
)
sns.lineplot(
    data=results_df_random,
    x="variable",
    y="value",
    label="Random",
    color="orange",
)
plt.xticks(np.arange(1, n_iters + 1))
plt.xlabel("Iteration")
plt.ylabel(r"$R^2$")
plt.ylim([0, 1])
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()


plt.savefig("./out/simulated_prediction_experiment.png")
plt.show()
import ipdb

ipdb.set_trace()
