from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, jit

from bandits.data import DatasetGenerationFn, batched_rewards

SelectActionFn = Callable[[Array, NamedTuple], Array]
UpdateStateFn = Callable[[NamedTuple, int, float], NamedTuple]


class Policy(NamedTuple):
    select_action_fn: SelectActionFn
    update_state_fn: UpdateStateFn
    init_state: NamedTuple


@partial(jit, static_argnames=("select_action_fn", "update_state_fn", "num_steps"))
def simulate(
    key: Array,
    true_rewards: Array,
    select_action_fn: SelectActionFn,
    update_state_fn: UpdateStateFn,
    init_state: Array,
    num_steps: int,
) -> tuple[NamedTuple, Array]:
    def step(bandit_state: Array, xs: tuple[Array, Array]) -> tuple[NamedTuple, Array]:
        key_action, rewards = xs

        action = select_action_fn(key_action, bandit_state)
        bandit_state = update_state_fn(bandit_state, action, rewards[action])

        return bandit_state, rewards[action]

    keys = jax.random.split(key, num_steps)
    state, rewards = jax.lax.scan(step, init_state, (keys, true_rewards))

    return state, rewards


@partial(jit, static_argnames=("select_action_fn", "update_state_fn", "num_steps", "num_iter"))
def simulate_multiple_iterations(
    key: Array,
    true_rewards: Array,
    select_action_fn: SelectActionFn,
    update_state_fn: UpdateStateFn,
    init_state: Array,
    num_steps: int,
    num_iter: int,
) -> tuple[NamedTuple, Array]:
    keys = jax.random.split(key, num_iter)
    return jax.vmap(simulate, in_axes=(0, None, None, None, None, None))(
        keys,
        true_rewards,
        select_action_fn,
        update_state_fn,
        init_state,
        num_steps,
    )


@partial(jit, static_argnames=("select_action_fn", "update_state_fn", "num_steps", "num_iter"))
def simulate_multiple_datasets(
    key: Array,
    batched_true_rewards: Array,
    select_action_fn: SelectActionFn,
    update_state_fn: UpdateStateFn,
    init_state: Array,
    num_steps: int,
    num_iter: int,
) -> tuple[NamedTuple, Array]:
    return jax.vmap(simulate_multiple_iterations, in_axes=(None, 0, None, None, None, None, None))(
        key,
        batched_true_rewards,
        select_action_fn,
        update_state_fn,
        init_state,
        num_steps,
        num_iter,
    )


def benchmark_policies(
    key: Array,
    policies: list[Policy],
    num_actions: int,
    num_steps: int,
    num_iter: int,
    num_datasets: int,
    dataset_generator_fn: DatasetGenerationFn,
    dataset_generator_fn_args: tuple,
) -> tuple[list[NamedTuple], Array]:
    key_datasets, key_iters = jax.random.split(key, 2)

    batched_true_rewards, _ = batched_rewards(
        key_datasets,
        dataset_generator_fn,
        dataset_generator_fn_args,
        num_steps,
        num_actions,
        num_datasets,
    )

    states, rewards = [], []
    for policy in policies:
        s, r = simulate_multiple_datasets(
            key_iters,
            batched_true_rewards,
            policy.select_action_fn,
            policy.update_state_fn,
            policy.init_state,
            num_steps,
            num_iter,
        )
        states.append(s)
        rewards.append(r)

    return states, jnp.stack(rewards, axis=0)
