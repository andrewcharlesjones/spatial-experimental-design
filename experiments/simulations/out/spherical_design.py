import numpy as onp
import jax.numpy as jnp
import matplotlib.pyplot as plt

# from scipy.stats import bernoulli
from jax import random, vmap
from jax.example_libraries import optimizers
from jax import grad, value_and_grad
from jax import jit
from jax.scipy.stats import norm, bernoulli
from os.path import join as pjoin
from jax.random import bernoulli as bernoulli_jax, PRNGKey
from jax.scipy.special import logsumexp

from tqdm import tqdm

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

N_ROUNDS = 10
prior_mean = 0.0
prior_variance = 1.0
# true_thresholds = [-1.5, 1.5]


def expit(x):
    return 1 / (1 + jnp.exp(-x))


def unpack_variational_params(params):
    means, stddevs = params[:2], jnp.exp(params[2:])
    return means, stddevs


def unpack_model_params(params):
    radius, slope = params
    return radius, slope


def my_norm(x):
    return jnp.sqrt(jnp.sum(x ** 2, axis=1))


def lognormal_logpdf(x, mean, stddev):
    normalizer = -jnp.log(x) - jnp.log(stddev) - 0.5 * jnp.log(2 * jnp.pi)
    lik = -0.5 * (jnp.log(x) - mean) ** 2 / (stddev ** 2)
    return normalizer + lik


class Model:
    def __init__(self, prior_mean, prior_stddev, key=PRNGKey(4)):
        self.key = key
        self.prior_mean = prior_mean
        self.prior_stddev = prior_stddev

    def predict(self, X, radius, slope):
        # logits = jnp.linalg.norm(X - center, axis=1) - radius
        logits = jnp.linalg.norm(X, axis=1) - radius
        logits *= slope
        preds = expit(-logits)
        return preds

    def log_density(self, X, Y, params):
        radius, slope = unpack_model_params(params)
        transformed_params = self.predict(X, radius, slope)

        # log_prior_radius = norm.logpdf(
        #     radius, loc=self.prior_mean, scale=self.prior_stddev
        # )
        log_prior_radius = lognormal_logpdf(
            radius, mean=self.prior_mean, stddev=self.prior_stddev
        )
        log_prior_slope = norm.logpdf(
            slope, loc=self.prior_mean, scale=self.prior_stddev
        )

        log_likelihood = jnp.sum(bernoulli.logpmf(k=Y, p=transformed_params))
        return log_likelihood + log_prior_radius + log_prior_slope


class ApproximateModel:
    def __init__(self, key=PRNGKey(4)):
        self.key = key

    def sample(self, params, size):
        means, stddevs = unpack_variational_params(params)
        std_normal_samples = random.normal(key=self.key, shape=(size, 2))
        samples = std_normal_samples * stddevs + means
        return jnp.vstack([jnp.exp(samples[:, 0]), samples[:, 1]]).T

    def log_density(self, responses, params):
        means, stddevs = unpack_variational_params(params)
        # log_dens = norm.logpdf(responses, loc=means, scale=stddevs)

        radius, slope = unpack_model_params(responses)
        # log_prior_radius = norm.logpdf(radius, loc=means[0], scale=stddevs[0])
        log_prior_radius = lognormal_logpdf(radius, mean=means[0], stddev=stddevs[0])
        log_prior_slope = norm.logpdf(slope, loc=means[1], scale=stddevs[1])

        return log_prior_radius  # + log_prior_center


def fit_variational_model(
    X,
    Y,
    model_object,
    n_iters=3000,
    print_every=100,
    n_mcmc_samples_vi=5,
    learning_rate=1e-2,
):
    variational_model = ApproximateModel()

    # Initialize variational parameters
    # params = random.normal(variational_model.key, shape=(4,))
    params = jnp.concatenate(
        [
            jnp.ones(2) * 0.5,
            -4 + 1e-4 * random.normal(variational_model.key, shape=(2,)),
        ]
    )
    # params = 0.5 + 1e-4 * random.normal(variational_model.key, shape=(4,))

    ## Optimization loop
    opt_init, opt_update, get_params = optimizers.adam(step_size=learning_rate)
    opt_state = opt_init(params)

    VI_objective = lambda params: negative_elbo(
        jnp.array(X),
        jnp.array(Y),
        params,
        model_object,
        variational_model,
        n_mcmc_samples=n_mcmc_samples_vi,
    )

    @jit
    def step(step, opt_state):
        value, grads = value_and_grad(VI_objective)(get_params(opt_state))
        opt_state = opt_update(step, grads, opt_state)
        return value, opt_state

    last_mll = onp.inf
    for step_num in range(int(n_iters)):
        curr_mll, opt_state = step(step_num, opt_state)
        # print(get_params(opt_state))
        if step_num % print_every == 0:
            print(
                "Step: {:<15} ELBO: {}".format(
                    step_num, onp.round(-1 * onp.asarray(curr_mll), 2)
                )
            )

    fitted_params = get_params(opt_state)
    return fitted_params


def negative_elbo(X, Y, params, model_object, variational_model, n_mcmc_samples=5):
    # Sample from variational posterior

    curr_samples = variational_model.sample(params, size=n_mcmc_samples)

    # Evaluate model and variational log densities
    model_log_density_eval = jnp.array(
        [model_object.log_density(X, Y, s) for s in curr_samples]
    )

    variational_log_density = jnp.array(
        [variational_model.log_density(s, params) for s in curr_samples]
    )

    # Compute approximate ELBO
    curr_elbo = jnp.mean(model_log_density_eval - variational_log_density)

    return -curr_elbo


@jit
def sample_params(variational_params, n_outer_samples=100, n_inner_samples=50):
    # variational_model = ApproximateModel(key=PRNGKey(4))

    # Sample parameters from current model
    #   (either the prior or posterior conditioned on observed data so far)
    param_samples = variational_model.sample(
        params=variational_params, size=n_inner_samples
    )
    param_samples_for_y = variational_model.sample(
        params=variational_params, size=n_outer_samples
    )
    return param_samples, param_samples_for_y


@jit
def predict_one(x, sample):
    return model.predict(X=x.reshape(1, -1), radius=sample[0], slope=sample[1])


@jit
def eig_calculation(denominator_individual_loglikelihoods, log_numerator):
    denominator_loglikelihoods = jnp.sum(denominator_individual_loglikelihoods, axis=2)

    n_inner_samples = denominator_loglikelihoods.shape[1]
    log_denominator_mean = logsumexp(denominator_loglikelihoods, axis=1) - jnp.log(
        n_inner_samples
    )

    eig = jnp.mean(log_numerator - log_denominator_mean)
    return eig


# @jit
def nmc(
    design,
    variational_params,
    n_outer_samples=100,
    n_inner_samples=50,
):

    design = jnp.array(design)
    # curr_idx = get_points_near_line(slope=design[1], X=jnp.array(X), intercept=design[0])
    # curr_X = X[design]
    # curr_X = jnp.take(X, curr_idx)

    # design = design.astype(float)

    param_samples, param_samples_for_y = sample_params(variational_params)

    # logit_p = jnp.array(
    #     [
    #         model.predict(X=design, radius=sample[0], slope=sample[1])
    #         for sample in param_samples
    #     ]
    # )
    logit_p = vmap(lambda s: vmap(lambda x: predict_one(x.reshape(1, -1), s))(design))(
        param_samples
    ).squeeze(axis=-1)

    # logit_p_for_y = jnp.array(
    #     [
    #         model.predict(X=design, radius=sample[0], slope=sample[1])
    #         for sample in param_samples_for_y
    #     ]
    # )
    logit_p_for_y = vmap(
        lambda s: vmap(lambda x: predict_one(x.reshape(1, -1), s))(design)
    )(param_samples_for_y).squeeze(axis=-1)

    # Sample imaginary data
    # y_samples = random.bernoulli(key=variational_model.key, p=logit_p_for_y).astype(int)
    y_samples = vmap(lambda p: random.bernoulli(key=variational_model.key, p=p))(
        logit_p_for_y
    ).astype(int)

    # Evaluate likelihood of synthetic data
    # log_numerator_individual = bernoulli.logpmf(p=logit_p_for_y, k=y_samples)
    log_numerator_individual = vmap(lambda lp, y: bernoulli.logpmf(p=lp, k=y))(
        logit_p_for_y, y_samples
    )
    # import ipdb; ipdb.set_trace()
    log_numerator = log_numerator_individual.sum(1)

    # denominator_individual_loglikelihoods = jnp.stack(
    #     [bernoulli.logpmf(p=logit_p, k=y) for y in y_samples], axis=0
    # )
    denominator_individual_loglikelihoods = vmap(
        lambda y: bernoulli.logpmf(p=logit_p, k=y)
    )(y_samples)
    # import ipdb; ipdb.set_trace()

    # denominator_loglikelihoods = jnp.sum(denominator_individual_loglikelihoods, axis=2)

    # log_denominator_mean = logsumexp(denominator_loglikelihoods, axis=1) - jnp.log(
    #     n_inner_samples
    # )

    # eig = jnp.mean(log_numerator - log_denominator_mean)
    # import ipdb; ipdb.set_trace()
    eig = eig_calculation(denominator_individual_loglikelihoods, log_numerator)

    return eig


def run_experiment(design):
    return ((design > true_thresholds[0]) & (design < true_thresholds[1])).astype(int)


def get_points_near_line(slope, X, intercept=0):
    dists = onp.abs(-slope * X[:, 0] + X[:, 1] - intercept) / onp.sqrt(slope ** 2 + 1)
    observed_idx = onp.where(dists <= slice_radius)[0]
    return observed_idx
    # return dists <= slice_radius


# variational_model = ApproximateModel()

model = Model(prior_mean=prior_mean, prior_stddev=onp.sqrt(prior_variance))


limits = [-10, 10]
grid_size = 50
radius = 5
center = onp.array([0, 0])
x1s = onp.linspace(*limits, num=grid_size)
x2s = onp.linspace(*limits, num=grid_size)
X1, X2 = onp.meshgrid(x1s, x2s)
X = onp.vstack([X1.ravel(), X2.ravel()]).T
norms = onp.linalg.norm(X, axis=1)
Y = (norms < radius).astype(int)

# Randomly flip some of the tumor cells to be healthy
tumor_idx = onp.where(Y == 1)[0]
flip_list = onp.random.binomial(n=1, p=0.8, size=len(tumor_idx))
Y[tumor_idx] = flip_list

plt.figure(figsize=(7, 5))
for c in [0, 1]:
    curr_idx = onp.where(Y == c)[0]
    plt.scatter(X[curr_idx, 0], X[curr_idx, 1], label="Healthy" if c == 0 else "Tumor")
plt.axis("off")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("./out/border_finding_data.png")
plt.show()
import ipdb

ipdb.set_trace()

slice_radius = 0.25

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
    curr_idx = get_points_near_line(slope=dd[1], X=X, intercept=dd[0])
    curr_X = X[curr_idx]
    Xs_for_designs.append(curr_X)

observed_idx = onp.random.choice(onp.arange(len(X)), size=20, replace=False).tolist()
# observed_idx = []

for iternum in range(25):

    variational_model = ApproximateModel(key=PRNGKey(iternum))

    if len(observed_idx) == 0:
        fitted_params = jnp.zeros(4) - 1
    else:
        fitted_params = fit_variational_model(
            X=X[onp.unique(onp.array(observed_idx))],
            Y=Y[onp.unique(onp.array(observed_idx))],
            model_object=model,
        )

    radius_mean = jnp.exp(fitted_params[0] + 0.5 * jnp.exp(fitted_params[2]) ** 2)
    print(radius_mean)

    eigs = []
    for dd in tqdm(Xs_for_designs):

        eigs.append(
            nmc(
                design=dd,
                variational_params=fitted_params,
            )
        )

    print(onp.array(eigs))
    print(onp.max(eigs))
    # plt.close()
    # plt.plot(onp.array(eigs))
    # plt.show()

    plt.figure(figsize=(7, 5))

    # for ii in range(len(Xs_for_designs)):
    #     plt.scatter(
    #         Xs_for_designs[ii][:, 0],
    #         Xs_for_designs[ii][:, 1],
    #         c=onp.repeat(eigs[ii], len(Xs_for_designs[ii])),
    #         vmin=onp.min(eigs),
    #         vmax=onp.max(eigs),
    #     )
    # plt.show()

    if len(observed_idx) == 0:
        for c in [0, 1]:
            curr_idx = onp.where(Y == c)
            plt.scatter(
                X[curr_idx, 0],
                X[curr_idx, 1],
                label="Tumor" if c == 1 else "Healthy",
                alpha=0.5,
            )
    else:
        for c in [0, 1]:
            curr_idx = onp.where(Y[onp.array(observed_idx)] == c)
            plt.scatter(
                X[onp.array(observed_idx)][curr_idx, 0],
                X[onp.array(observed_idx)][curr_idx, 1],
                label="Tumor" if c == 1 else "Healthy",
                alpha=0.5,
            )

    # plt.scatter(X[idx_for_designs, 0], X[idx_for_designs, 1], c=eigs)

    plt.colorbar()
    best_design_idx = onp.argmax(eigs)
    best_design = candidate_designs[best_design_idx]
    curr_observed_idx = get_points_near_line(
        slope=best_design[1], X=X, intercept=best_design[0]
    )
    observed_idx.extend(curr_observed_idx.tolist())

    # observed_idx.append(idx_for_designs[best_design_idx])

    # plt.plot(
    #     onp.array([limits[0], limits[1]]),
    #     onp.array([best_design[0] + limits[0] * best_design[1], best_design[0] + limits[1] * best_design[1]]),
    #     color="red",
    #     linewidth=5,
    #     linestyle="--",
    # )

    # import ipdb; ipdb.set_trace()
    plt.scatter(
        Xs_for_designs[best_design_idx][:, 0],
        Xs_for_designs[best_design_idx][:, 1],
        color="red",
    )

    plt.xlim(limits)
    plt.ylim(limits)
    plt.show()


import ipdb

ipdb.set_trace()
