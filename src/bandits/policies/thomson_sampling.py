from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, jit


class ThomsonSamplingState(NamedTuple):
    alphas: Array
    betas: Array
    discount: float


@jit
def select_action(key: Array, state: ThomsonSamplingState) -> Array:
    expected_rewards = jax.random.beta(key, state.alphas, state.betas)

    return jnp.argmax(expected_rewards)


@jit
def update_state(state: ThomsonSamplingState, action: int, reward: float) -> ThomsonSamplingState:
    alphas = state.alphas * state.discount
    betas = state.betas * state.discount

    alphas_update = alphas[action] + reward
    betas_update = betas[action] + (1.0 - reward)

    alphas = alphas.at[action].set(alphas_update)
    betas = betas.at[action].set(betas_update)

    return ThomsonSamplingState(alphas, betas, state.discount)
