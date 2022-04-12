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

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

sys.path.append("../../util/")
from util import get_points_near_line

DATA_DIR = "/Users/andrewjones/Documents/beehive/spatial-imputation/data/visium"
N_GENES = 100

## Load data
adata = sc.read_visium(DATA_DIR)
adata.var_names_make_unique()
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

sc.pp.filter_cells(adata, min_counts=2000)
sc.pp.filter_cells(adata, max_counts=35000)
adata = adata[adata.obs["pct_counts_mt"] < 20]
print(f"#cells after MT filter: {adata.n_obs}")
sc.pp.filter_genes(adata, min_cells=10)

sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=N_GENES, subset=True)


genes = adata[:, adata.var.highly_variable].var_names.values[:100]
sq.gr.spatial_neighbors(adata)
sq.gr.spatial_autocorr(
    adata,
    mode="moran",
)
moran_scores = adata.uns["moranI"]
top_gene_name = adata.uns["moranI"].index.values[0]

## Extract and normalize/standardize data
X = adata.obsm["spatial"].astype(float)
Y = np.array(adata[:, top_gene_name].X.todense()).squeeze()

X -= X.min(0)
X /= X.max(0)
X *= 10
X -= 5

Y = (Y - Y.mean()) / Y.std()


n_iters = 5
kernel = RBF() + WhiteKernel()

# When we take a slice, all points within this
# distance to the line/plane are "observed"
slice_radius = 0.1


# Discretize design space
n_slope_discretizations = 10
n_intercept_discretizations = 10
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
r2_scores = np.zeros(n_iters)


for iternum in range(n_iters):

    assert len(designs) == iternum
    assert len(X_fragment_idx) == iternum + 1

    if len(observed_idx) > 0:
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

            above_fragment_idx = np.where(
                curr_X[:, 1] >= curr_design[0] + curr_X[:, 0] * curr_design[1]
            )[0]

            curr_observed_idx = get_points_near_line(
                X=curr_X,
                slope=curr_design[1],
                intercept=curr_design[0],
                slice_radius=slice_radius,
            )

            L = len(curr_observed_idx)
            if L == 0:
                continue

            if iternum == 0:
                cov = kernel(curr_X[curr_observed_idx])
                noise_variance = np.exp(kernel.k2.theta[0])
            else:
                # Make predictions of expression
                _, cov = gpr.predict(curr_X[curr_observed_idx], return_cov=True)

                noise_variance = np.exp(gpr.kernel_.k2.theta[0])

            curr_eig = 0.5 * np.linalg.slogdet(1 / noise_variance * cov + np.eye(L))[1]

            if curr_eig > best_eig:
                best_design_idx = dd
                best_fragment_idx = ff
                best_observed_idx = X_fragment_idx[ff][curr_observed_idx]
                best_eig = curr_eig

    curr_best_design = candidate_designs[best_design_idx]

    best_fragment_X = X[X_fragment_idx[best_fragment_idx]]

    above_fragment_idx = np.where(
        best_fragment_X[:, 1]
        >= curr_best_design[0] + best_fragment_X[:, 0] * curr_best_design[1]
    )[0]
    below_fragment_idx = np.where(
        best_fragment_X[:, 1]
        <= curr_best_design[0] + best_fragment_X[:, 0] * curr_best_design[1]
    )[0]

    above_idx = X_fragment_idx[best_fragment_idx][above_fragment_idx]
    below_idx = X_fragment_idx[best_fragment_idx][below_fragment_idx]
    X_fragment_idx.pop(best_fragment_idx)
    X_fragment_idx.append(above_idx)
    X_fragment_idx.append(below_idx)

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
    r2_scores[iternum] = curr_r2


## Random slice



n_repeats = 10
r2_scores_random = np.zeros((n_repeats, n_iters))

for rep_ii in range(n_repeats):
    designs_random = []
    observed_idx_random = []
    X_fragment_idx_random = [np.arange(len(X))]

    for iternum in range(n_iters):

        assert len(designs_random) == iternum
        assert len(X_fragment_idx_random) == iternum + 1

        curr_observed_idx = []
        while len(curr_observed_idx) == 0:
            rand_fragment_idx = np.random.choice(np.arange(len(X_fragment_idx_random)))
            rand_design_idx = np.random.choice(np.arange(n_candidate_designs))

            curr_design = candidate_designs[rand_design_idx]
            curr_X = X[X_fragment_idx_random[rand_fragment_idx]]
            curr_observed_idx = get_points_near_line(
                X=curr_X,
                slope=curr_design[1],
                intercept=curr_design[0],
                slice_radius=slice_radius,
            )

        best_fragment_X = X[X_fragment_idx_random[rand_fragment_idx]]

        observed_idx_random.extend(
            X_fragment_idx_random[rand_fragment_idx][curr_observed_idx].tolist()
        )

        above_fragment_idx = np.where(
            best_fragment_X[:, 1]
            >= curr_design[0] + best_fragment_X[:, 0] * curr_design[1]
        )[0]
        below_fragment_idx = np.where(
            best_fragment_X[:, 1]
            <= curr_design[0] + best_fragment_X[:, 0] * curr_design[1]
        )[0]

        above_idx = X_fragment_idx_random[rand_fragment_idx][above_fragment_idx]
        below_idx = X_fragment_idx_random[rand_fragment_idx][below_fragment_idx]
        X_fragment_idx_random.pop(rand_fragment_idx)
        X_fragment_idx_random.append(above_idx)
        X_fragment_idx_random.append(below_idx)

        designs_random.append(curr_design)

        train_idx = np.array(observed_idx_random)
        test_idx = np.setdiff1d(np.arange(len(X)), train_idx)
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        gpr = GPR(kernel=kernel)
        gpr.fit(X_train, Y_train)
        preds = gpr.predict(X_test)
        curr_r2 = r2_score(Y_test, preds)
        r2_scores_random[rep_ii, iternum] = curr_r2



plt.figure(figsize=(14, 5))

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c="gray", alpha=0.5)
plt.scatter(X[np.array(observed_idx), 0], X[np.array(observed_idx), 1], label="Designed", color="blue")
plt.scatter(X[np.array(observed_idx_random), 0], X[np.array(observed_idx_random), 1], label="Random", color="orange")
plt.axis("off")
plt.legend([],[], frameon=False)
plt.tight_layout()

plt.subplot(122)

results_df = pd.DataFrame({"iter": np.arange(1, n_iters + 1), "r2": r2_scores})
results_df_random = pd.melt(pd.DataFrame(r2_scores_random))
results_df_random["variable"] += 1

sns.lineplot(data=results_df, x="iter", y="r2", label="Designed", color="blue")
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
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()


plt.savefig("./out/visium_prediction_experiment.png")
plt.show()
import ipdb

ipdb.set_trace()
