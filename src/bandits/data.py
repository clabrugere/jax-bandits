from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array, jit
from jax.scipy.special import gammaln

InterpolationFn = Callable[[Array, Array, Array], Array]


@jit
def linear_interpolation(x0: Array, x1: Array, t: Array) -> Array:
    return (1 - t) * x0 + t * x1


@jit
def cosine_interpolation(x0: Array, x1: Array, t: Array) -> Array:
    cos_t = (1 - jnp.cos(t * jnp.pi)) / 2.0
    return (1 - cos_t) * x0 + cos_t * x1


@jit
def exponential_interpolation(x0: Array, x1: Array, t: Array, base: float = 2.0) -> Array:
    t_exp = jnp.pow(t, base)
    return (1 - t_exp) * x0 + t_exp * x1


@jit
def bezier_interpolation(control_points: Array, t: Array) -> Array:
    def binomial_coeff(n: int, k: int) -> Array:
        return jnp.exp(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))

    def bezier_term(t: Array, degree: int, i: int) -> Array:
        return (1 - t) ** (degree - i) * t**i

    degree = control_points.shape[0] - 1
    binom_coeffs = jax.vmap(binomial_coeff, in_axes=(None, 0))(degree, jnp.arange(degree + 1))
    weighted_control_points = binom_coeffs[:, None] * control_points

    # Compute (1-t)^(m-i) * t^i for all control points
    terms = jax.vmap(bezier_term, in_axes=(None, None, 0))(t, degree, jnp.arange(degree + 1))

    return jnp.dot(terms.T, weighted_control_points)


@partial(jit, static_argnames=("num_steps", "num_actions"))
def stationary_rewards(key: Array, num_steps: int, num_actions: int) -> tuple[Array, Array]:
    """Generates a random array of shape (num_steps, num_actions) or stationary rewards by sampling a probability vector
    from the uniform distribution and using it to sample num_steps rewards from the Bernoulli distribution.
    """
    key_p, key_r = jax.random.split(key, 2)
    p = jax.random.uniform(key_p, (num_actions,))
    rewards = jax.random.bernoulli(key_r, p, (num_steps, num_actions)).astype(jnp.float32)

    return rewards, p


@partial(jit, static_argnames=("num_steps", "num_actions", "interpolation_fn"))
def biregime_rewards(
    key: Array,
    num_steps: int,
    num_actions: int,
    interpolation_fn: InterpolationFn,
) -> tuple[Array, Array]:
    """Generates a random array of shape (num_steps, num_actions) of non stationary rewards with 2 regimes.
    The method samples two probability vectors p_init and p_final from the uniform distribution and creates num_steps-2
    probability vectors in between by interpolating using interpolation_fn. Finally it samples num_steps rewards from
    the Bernoulli distribution.
    """
    key_p_init, key_p_final, key_r = jax.random.split(key, 3)

    p_init = jax.random.uniform(key_p_init, (num_actions,))
    p_final = jax.random.uniform(key_p_final, (num_actions,))

    t = jnp.expand_dims(jnp.linspace(0.0, 1.0, num_steps), -1)
    p = interpolation_fn(p_init, p_final, t)
    rewards = jax.random.bernoulli(key_r, p, (num_steps, num_actions)).astype(jnp.float32)

    return rewards, p


@partial(jit, static_argnames=("num_steps", "num_actions", "num_regimes"))
def multiregime_rewards(key: Array, num_steps: int, num_actions: int, num_regimes: int) -> tuple[Array, Array]:
    """Generates a random array of shape (num_steps, num_actions) of non stationary rewards with num_regimes regimes.
    The method samples num_regimes probability vectors [p0...pr] and creates num_steps - num_regimes probability vectors
    in between by using Bézier interpolation. It then samples num_steps rewards from the Bernoulli distribution.
    """
    key_p, key_r = jax.random.split(key, 2)

    control_points = jax.random.uniform(key_p, (num_regimes, num_actions))
    t = jnp.linspace(0.0, 1.0, num_steps)
    p = bezier_interpolation(control_points, t)
    rewards = jax.random.bernoulli(key_r, p, (num_steps, num_actions)).astype(jnp.float32)

    return rewards, p
