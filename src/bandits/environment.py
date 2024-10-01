from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, jit

# TODO: generate non stationary rewards


# TODO: generate rewards within the method
@partial(jit, static_argnames=("select_action_fn", "update_state_fn"))
def simulate(
    keys: Array,
    true_rewards: Array,
    select_action_fn: Callable,
    update_state_fn: Callable,
    init_state: NamedTuple,
) -> tuple[NamedTuple, Array]:
    def step(carry: tuple[NamedTuple, Array], key: Array) -> tuple[NamedTuple, Array]:
        bandit_state, true_rewards = carry
        key_action, key_action_reward = jax.random.split(key)

        action = select_action_fn(key_action, bandit_state)
        reward = jax.random.bernoulli(key_action_reward, true_rewards[action]).astype(jnp.float32)
        bandit_state = update_state_fn(bandit_state, action, reward)

        return (bandit_state, true_rewards), reward

    # key_rewards, key = jax.random.split(key)
    # keys_scan = jax.random_split(key, num_steps)
    # true_rewards = jax.random.uniform(key_rewards, num_actions)

    (state, _), rewards = jax.lax.scan(step, (init_state, true_rewards), keys)

    return state, rewards


# TODO: pass only one key and split it in the method
@partial(jit, static_argnames=("select_action_fn", "update_state_fn"))
def simulate_multiple(
    keys: Array,
    true_rewards: Array,
    select_action_fn: Callable,
    update_state_fn: Callable,
    init_state: NamedTuple,
) -> tuple[NamedTuple, Array]:
    return jax.vmap(simulate, in_axes=(0, None, None, None, None))(
        keys,
        true_rewards,
        select_action_fn,
        update_state_fn,
        init_state,
    )
