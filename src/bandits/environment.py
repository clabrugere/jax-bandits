from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, jit

SelectActionFn = Callable[[Array, NamedTuple], Array]
UpdateStateFn = Callable[[NamedTuple, int, float], NamedTuple]


# TODO: update signature to allow passing a function generating rewards instead
@partial(jit, static_argnames=("select_action_fn", "update_state_fn", "num_actions", "num_steps"))
def simulate(
    key: Array,
    select_action_fn: SelectActionFn,
    update_state_fn: UpdateStateFn,
    init_state: NamedTuple,
    num_actions: int,
    num_steps: int,
) -> tuple[NamedTuple, Array]:
    def step(bandit_state: Array, xs: tuple[Array, Array]) -> tuple[NamedTuple, Array]:
        key_action, rewards = xs

        action = select_action_fn(key_action, bandit_state)
        bandit_state = update_state_fn(bandit_state, action, rewards[action])

        return bandit_state, rewards[action]

    key_rewards, key_probas, key = jax.random.split(key, 3)
    keys_scan = jax.random.split(key, num_steps)

    reward_probas = jax.random.uniform(key_probas, (num_actions,))
    true_rewards = jax.random.bernoulli(key_rewards, reward_probas, (num_steps, num_actions)).astype(jnp.float32)

    state, rewards = jax.lax.scan(step, init_state, (keys_scan, true_rewards))

    return state, rewards


@partial(jit, static_argnames=("select_action_fn", "update_state_fn", "num_actions", "num_steps", "num_iter"))
def simulate_multiple(
    key: Array,
    select_action_fn: SelectActionFn,
    update_state_fn: UpdateStateFn,
    init_state: NamedTuple,
    num_actions: int,
    num_steps: int,
    num_iter: int,
) -> tuple[NamedTuple, Array]:
    keys = jax.random.split(key, num_iter)
    return jax.vmap(simulate, in_axes=(0, None, None, None, None, None))(
        keys,
        select_action_fn,
        update_state_fn,
        init_state,
        num_actions,
        num_steps,
    )
