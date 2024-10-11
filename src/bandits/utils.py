import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array


def plot_rewards(rewards: Array, names: list[str]) -> None:
    num_policies, num_datasets, num_iterations, num_steps = rewards.shape

    stderr = rewards.reshape(num_policies, num_datasets * num_iterations, num_steps).std(axis=1)
    stderr /= jnp.sqrt(num_datasets * num_iterations)

    step_inds = jnp.arange(1, num_steps + 1)
    rewards = rewards.mean(axis=(1, 2)).cumsum(axis=1) / step_inds

    _, ax = plt.subplots()
    for i in range(num_policies):
        ax.plot(rewards[i, :], label=names[i])
        ax.fill_between(step_inds - 1, rewards[i, :] + stderr[i], rewards[i, :] - stderr[i], alpha=0.2)

    ax.set_xlabel("step")
    ax.set_ylabel("average reward")
    ax.spines[["right", "top"]].set_visible(False)

    plt.legend()
    plt.tight_layout()


def plot_reward_probs(p: Array) -> None:
    assert p.ndim == 2

    _, ax = plt.subplots()
    for a in range(p.shape[1]):
        ax.plot(p[:, a], label=f"action {a}")

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("step")
    ax.set_ylabel("$P(reward = 1)$")
    ax.spines[["right", "top"]].set_visible(False)

    plt.legend()
    plt.tight_layout()
