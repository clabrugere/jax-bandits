from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, jit


class EpsilonGreedyState(NamedTuple):
    epsilon: float
    epsilon_decay: float
    values: Array
    counts: Array


@jit
def select_action(key: Array, state: EpsilonGreedyState) -> Array:
    key, subkey = jax.random.split(key)
    num_actions = state.values.shape[0]

    return jax.lax.cond(
        jax.random.uniform(key) < state.epsilon,
        lambda: jax.random.randint(subkey, shape=(), minval=0, maxval=num_actions),
        lambda: jnp.argmax(state.values),
    )


@jit
def update_state(state: EpsilonGreedyState, action: int, reward: float) -> EpsilonGreedyState:
    counts, values = state.counts, state.values

    counts_update = counts[action] + 1
    values_update = values[action] + (reward - values[action]) / counts_update

    counts = counts.at[action].set(counts_update)
    values = values.at[action].set(values_update)

    return EpsilonGreedyState(state.epsilon * state.epsilon_decay, state.epsilon_decay, values, counts)
