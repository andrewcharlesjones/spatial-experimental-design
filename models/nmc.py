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








@jit
def eig_calculation(denominator_loglikelihoods, log_numerator):

    n_inner_samples = denominator_loglikelihoods.shape[1]
    log_denominator_mean = logsumexp(denominator_loglikelihoods, axis=1) - jnp.log(
        n_inner_samples
    )

    eig = jnp.mean(log_numerator - log_denominator_mean)
    return eig


@jit
def sample_bernoulli(p, key):
    return random.bernoulli(key=key, p=p)


@jit
def bernoulli_logpmf(p, x):
    return bernoulli.logpmf(p=p, k=x)




class EIGComputer:

    def __init__(
        self, 
        X_full, 
        model,
        variational_model,
    ):

        self.X_full = X_full
        self.model = model
        self.variational_model = variational_model

    @partial(jit, static_argnums=(0,))
    def compute_eig(
        self,
        X,
        variational_params,
        n_outer_samples=100,
        n_inner_samples=50,
        mask=None,
        key_int=0
    ):
    
        # predict_one = lambda x, params: self.model.predict(X=x.reshape(1, -1), radius=params[0], slope=params[1], center=params[2:])
        # self.predict_one = jit(predict_one)

        
        # Sample
        # param_samples = self.variational_model.sample(
        #     params=self.variational_params, size=n_inner_samples
        # )
        # param_samples_for_y = self.variational_model.sample(
        #     params=self.variational_params, size=n_outer_samples
        # )
        param_samples, param_samples_for_y = self.sample_params(
            variational_params=variational_params, n_outer_samples=n_outer_samples, n_inner_samples=n_inner_samples
        )

        # Run NMC
        eig = self.nmc(
            X,
            variational_params,
            param_samples,
            param_samples_for_y,
            mask=mask,
            key_int=key_int
        )

        return eig

    # @partial(jit, static_argnums=(0,))
    def sample_params(self, variational_params, n_outer_samples, n_inner_samples):

        # Sample parameters from current model
        #   (either the prior or posterior conditioned on observed data so far)
        param_samples = self.variational_model.sample(
            params=variational_params, size=n_inner_samples
        )
        param_samples_for_y = self.variational_model.sample(
            params=variational_params, size=n_outer_samples
        )
        return param_samples, param_samples_for_y

    @partial(jit, static_argnums=(0,))
    def predict_one(self, x, sample):
        return self.model.predict(
            X=x.reshape(1, -1), radius=sample[0], slope=sample[1], center=sample[2:]
        )


    @partial(jit, static_argnums=(0,))
    def nmc(
        self,
        X,
        variational_params,
        param_samples,
        param_samples_for_y,
        mask=None,
        key_int=0,
    ):

        logit_p = vmap(lambda s: vmap(lambda x: self.predict_one(x.reshape(1, -1), s))(X))(
            param_samples
        ).squeeze(axis=-1)

        logit_p_for_y = vmap(lambda s: vmap(lambda x: self.predict_one(x.reshape(1, -1), s))(X))(
            param_samples_for_y
        ).squeeze(axis=-1)

        # Sample imaginary data
        y_samples = vmap(lambda p: sample_bernoulli(p=p, key=PRNGKey(key_int)))(logit_p_for_y).astype(int)

        # Evaluate likelihood of synthetic data
        log_numerator_individual = vmap(lambda lp, y: bernoulli_logpmf(p=lp, x=y))(
            logit_p_for_y, y_samples
        )
        # import ipdb; ipdb.set_trace()

        if mask is not None:
            log_numerator_individual = log_numerator_individual * mask.reshape(1, -1)

        log_numerator = log_numerator_individual.sum(1)

        denominator_individual_loglikelihoods = vmap(
            lambda y: bernoulli_logpmf(p=logit_p, x=y)
        )(y_samples)

        if mask is not None:
            denominator_individual_loglikelihoods = (
                denominator_individual_loglikelihoods * mask.reshape(1, 1, -1)
            )
        denominator_loglikelihoods = jnp.sum(denominator_individual_loglikelihoods, axis=2)

        eig = eig_calculation(denominator_loglikelihoods, log_numerator)

        return eig




