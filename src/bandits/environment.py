from functools import partial
from typing import Callable, NamedTuple

import jax
from jax import Array, jit

SelectActionFn = Callable[[Array, NamedTuple], Array]
UpdateStateFn = Callable[[NamedTuple, int, float], NamedTuple]
RewardsGenerationFn = Callable[..., tuple[Array, Array]]


@partial(
    jit,
    static_argnames=(
        "select_action_fn",
        "update_state_fn",
        "num_actions",
        "num_steps",
        "rewards_generator_fn",
        "rewards_generator_fn_args",
    ),
)
def simulate(
    key: Array,
    select_action_fn: Callable,
    update_state_fn: Callable,
    init_state: NamedTuple,
    num_actions: int,
    num_steps: int,
    rewards_generator_fn: RewardsGenerationFn,
    rewards_generator_fn_args: tuple,
) -> tuple[NamedTuple, Array]:
    def step(bandit_state: Array, xs: tuple[Array, Array]) -> tuple[NamedTuple, Array]:
        key_action, rewards = xs

        action = select_action_fn(key_action, bandit_state)
        bandit_state = update_state_fn(bandit_state, action, rewards[action])

        return bandit_state, rewards[action]

    key_rewards, key = jax.random.split(key, 2)
    keys_scan = jax.random.split(key, num_steps)

    true_rewards, _ = rewards_generator_fn(key_rewards, num_steps, num_actions, *rewards_generator_fn_args)
    state, rewards = jax.lax.scan(step, init_state, (keys_scan, true_rewards))

    return state, rewards


@partial(
    jit,
    static_argnames=(
        "select_action_fn",
        "update_state_fn",
        "num_actions",
        "num_steps",
        "rewards_generator_fn",
        "rewards_generator_fn_args",
        "num_iter",
    ),
)
def simulate_multiple(
    key: Array,
    select_action_fn: Callable,
    update_state_fn: Callable,
    init_state: NamedTuple,
    num_actions: int,
    num_steps: int,
    num_iter: int,
    rewards_generator_fn: Callable,
    rewards_generator_fn_args: tuple,
) -> tuple[NamedTuple, Array]:
    keys = jax.random.split(key, num_iter)
    return jax.vmap(simulate, in_axes=(0, None, None, None, None, None, None, None))(
        keys,
        select_action_fn,
        update_state_fn,
        init_state,
        num_actions,
        num_steps,
        rewards_generator_fn,
        rewards_generator_fn_args,
    )
