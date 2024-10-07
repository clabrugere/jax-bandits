import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array


def plot_rewards(rewards: list[Array], names: list[str]) -> None:
    num_steps = rewards[0].shape[1]
    step_inds = jnp.arange(1, num_steps + 1)

    _, ax = plt.subplots()
    for reward, name in zip(rewards, names):
        ax.plot(jnp.cumsum(jnp.mean(reward, axis=0)) / step_inds, label=name)

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
