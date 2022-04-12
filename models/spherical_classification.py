import numpy as onp
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial

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




def expit(x):
    return 1 / (1 + jnp.exp(-x))


def unpack_variational_params(params):
    means, stddevs = params[:4], jnp.exp(params[4:])
    return means, stddevs


def unpack_model_params(params):
    radius, slope, center = params[0], params[1], params[2:]
    return radius, slope, center


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

    def predict(self, X, radius, slope, center):
        logits = jnp.linalg.norm(X - center, axis=1) - radius
        logits *= slope
        preds = expit(-logits)
        return preds

    def log_density(self, X, Y, params):
        radius, slope, center = unpack_model_params(params)
        transformed_params = self.predict(X, radius, slope, center)

        log_prior_radius = lognormal_logpdf(
            radius, mean=self.prior_mean, stddev=self.prior_stddev
        )
        log_prior_slope = norm.logpdf(
            slope, loc=self.prior_mean, scale=self.prior_stddev
        )
        log_prior_center1 = norm.logpdf(
            center[0], loc=self.prior_mean, scale=self.prior_stddev
        )
        log_prior_center2 = norm.logpdf(
            center[1], loc=self.prior_mean, scale=self.prior_stddev
        )
        log_prior_center = log_prior_center1 + log_prior_center2

        log_likelihood = jnp.sum(bernoulli.logpmf(k=Y, p=transformed_params))
        return log_likelihood + log_prior_radius + log_prior_slope + log_prior_center


class ApproximateModel:
    def __init__(self, key=PRNGKey(4)):
        self.key = key

    # @partial(jit, static_argnums=(0,))
    def sample(self, params, size):
        means, stddevs = unpack_variational_params(params)
        std_normal_samples = random.normal(key=self.key, shape=(size, 4))
        samples = std_normal_samples * stddevs + means
        return jnp.hstack([jnp.exp(samples[:, 0]).reshape(-1, 1), samples[:, 1:]])
        return samples

    def log_density(self, responses, params):
        means, stddevs = unpack_variational_params(params)

        radius, slope, center = unpack_model_params(responses)

        log_dens_radius = lognormal_logpdf(radius, mean=means[0], stddev=stddevs[0])
        log_dens_slope = norm.logpdf(slope, loc=means[1], scale=stddevs[1])
        log_dens_center = norm.logpdf(center, loc=means[2:], scale=stddevs[2:]).sum()

        return log_dens_radius + log_dens_slope + log_dens_center


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
    params = jnp.concatenate(
        [
            jnp.ones(4) * 0.5,
            -4 + 1e-4 * random.normal(variational_model.key, shape=(4,)),
        ]
    )

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
        # if jnp.isnan(get_params(opt_state)).sum():
        #     import ipdb; ipdb.set_trace()
        if step_num % print_every == 0:
            print(
                "Step: {:<15} ELBO: {}".format(
                    step_num, onp.round(-1 * onp.asarray(curr_mll), 2)
                )
            )
            # print(get_params(opt_state))
            

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

