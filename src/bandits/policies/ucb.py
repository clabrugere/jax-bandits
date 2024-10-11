from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, jit


class UCBState(NamedTuple):
    c: float
    steps: int
    values: Array
    counts: Array


@jit
def select_action(key: Array, state: UCBState) -> Array:
    num_actions = state.counts.shape[0]

    def select_random():
        unselected_actions = jnp.where(state.counts == 0, True, False)
        return jax.random.choice(key, jnp.arange(num_actions), p=unselected_actions.astype(jnp.float32))

    def select_ucb():
        return jnp.argmax(state.values + state.c * jnp.sqrt(jnp.log(state.steps) / state.counts))

    return jax.lax.cond(
        state.steps < num_actions,
        select_random,
        select_ucb,
    )


@jit
def update_state(state: UCBState, action: int, reward: float) -> UCBState:
    counts = state.counts
    values = state.values

    counts_update = counts[action] + 1
    values_update = values[action] + (reward - values[action]) / counts_update

    counts = counts.at[action].set(counts_update)
    values = values.at[action].set(values_update)

    return UCBState(state.c, state.steps + 1, values, counts)
