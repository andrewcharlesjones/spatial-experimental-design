import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.metrics import pairwise_distances
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import r2_score
import seaborn as sns

DATA_DIR = "/Users/andrewjones/Documents/beehive/spatial-imputation/data/visium"
N_GENES = 100

## Load data
adata = sc.read_visium(DATA_DIR)
adata.var_names_make_unique()
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

sc.pp.filter_cells(adata, min_counts=5000)
sc.pp.filter_cells(adata, max_counts=35000)
adata = adata[adata.obs["pct_counts_mt"] < 20]
print(f"#cells after MT filter: {adata.n_obs}")
sc.pp.filter_genes(adata, min_cells=10)

sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=N_GENES, subset=True)

## Extract and normalize/standardize data
X = adata.obsm["spatial"].astype(float)
Y = np.array(adata.X[:, 1].todense()).squeeze()

keep_idx = np.where(Y != 0)[0]
X = X[keep_idx]
Y = Y[keep_idx]

X -= X.min(0)
X /= X.max(0)
X *= 10

Y = (Y - Y.mean()) / Y.std()


prior_sample_size = 1
n_iters = 300
kernel = RBF(length_scale=1.0, length_scale_bounds="fixed") + WhiteKernel(
    noise_level=0.1, noise_level_bounds="fixed"
)

n_repeats = 1
results = np.zeros((n_repeats, n_iters))
results_random = np.zeros((n_repeats, n_iters))

for rr in range(n_repeats):

    designs = []
    observations = []
    observed_idx = []

    prior_design_idx = np.random.choice(
        np.arange(X.shape[0]), size=prior_sample_size, replace=False
    )
    prior_designs = X[prior_design_idx]
    prior_observations = Y[prior_design_idx]

    designs.append(prior_designs)
    observations.append(prior_observations)
    observed_idx.extend(prior_design_idx.tolist())

    for iternum in range(n_iters):

        gpr = GPR(kernel=kernel)
        gpr.fit(
            X[np.array(observed_idx)],
            Y[np.array(observed_idx)],
        )

        # Make predictions of expression
        _, preds_stddev = gpr.predict(X, return_std=True)

        # Take point with highest predictive variance
        curr_design_idx = np.argmax(preds_stddev)
        designs.append(X[curr_design_idx].reshape(1, -1))
        observations.append(Y[curr_design_idx])
        observed_idx.append(curr_design_idx)

    designs = np.concatenate(designs)

    # Compare this with choosing locations uniformly at random
    random_design_idx = np.random.choice(
        np.arange(X.shape[0]), size=n_iters, replace=False
    )
    random_designs = X[random_design_idx]
    random_observations = Y[random_design_idx]


plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(random_designs[:, 0], random_designs[:, 1], color="black")
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.scatter(designs[:, 0], designs[:, 1], color="black")
plt.xticks([])
plt.yticks([])
plt.show()

# Plot MSE across iterations
# sns.lineplot(data=pd.melt(pd.DataFrame(results_random)), x="variable", y="value", label="Random")
# sns.lineplot(data=pd.melt(pd.DataFrame(results)), x="variable", y="value", label="Designed")
# plt.legend()
# plt.show()


import ipdb

ipdb.set_trace()
