from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, jit


class ThomsonSamplingState(NamedTuple):
    alphas: Array
    betas: Array


@jit
def select_action(key: Array, state: ThomsonSamplingState) -> Array:
    expected_rewards = jax.random.beta(key, state.alphas, state.betas)

    return jnp.argmax(expected_rewards)


@jit
def update_state(state: ThomsonSamplingState, action: int, reward: float) -> ThomsonSamplingState:
    alpha, beta = state.alphas, state.betas

    alpha_update = alpha[action] + reward
    beta_update = beta[action] + (1.0 - reward)

    alpha = alpha.at[action].set(alpha_update)
    beta = beta.at[action].set(beta_update)

    return ThomsonSamplingState(alpha, beta)
