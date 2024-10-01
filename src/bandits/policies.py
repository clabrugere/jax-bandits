from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from numpy.typing import NDArray


class Policy(ABC):
    def __init__(self, num_actions: int, name: str, *args, **kwargs) -> None:
        assert num_actions > 1, f"must have at least two actions, got {num_actions}"
        self.num_actions = num_actions
        self.name = name
        self.reset()

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def __call__(self, *args, **kwargs) -> int: ...

    @abstractmethod
    def update(self, action: int, reward: float, *args, **kwargs) -> None: ...


class EpsilonGreedy(Policy):
    def __init__(self, num_actions: int, epsilon: float, decay: float = 1.0, name: str = "Greedy") -> None:
        assert 0.0 < epsilon < 1.0, f"epsilon must be in (0, 1), got {epsilon}"
        assert 0.0 < epsilon <= 1.0, f"decay must be in (0, 1], got {decay}"

        self.epsilon = epsilon
        self.decay = decay
        super().__init__(num_actions, f"{epsilon:.2f}-{name}")

    def reset(self) -> None:
        self.counts = np.zeros(self.num_actions, dtype=np.int32)
        self.values = np.zeros(self.num_actions, dtype=np.float32)

    def __call__(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return self.values.argmax()

    def update(self, action: int, reward: float):
        self.epsilon *= self.decay
        self.counts[action] += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]


class UCB(Policy):
    def __init__(self, num_actions: int, c: float = 1.0, name: str = "UCB") -> None:
        assert c > 0, f"c must be strictly positive, got {c}"
        self.c = c
        super().__init__(num_actions, f"{name}{c:.0f}")

    def reset(self) -> None:
        self.steps = 0
        self.counts = np.zeros(self.num_actions, dtype=np.int32)
        self.values = np.zeros(self.num_actions, dtype=np.float32)

    def __call__(self) -> int:
        if self.steps < self.num_actions:
            return self.steps

        return np.argmax(self.values + self.c * np.sqrt(2 * np.log(self.steps) / self.counts))

    def update(self, action: int, reward: float) -> None:
        self.steps += 1
        self.counts[action] += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]


class ThomsonSampling(Policy, ABC):
    def __init__(
        self,
        num_actions: int,
        distribution: Callable,
        parameter_priors: tuple[NDArray, ...],
        discount: float = 1.0,
        name: str = "ThomsonSampling",
    ) -> None:
        assert 0 < discount <= 1.0, f"discount must be in (0, 1], got {discount}"

        self.distribution = distribution
        self.parameter_priors = parameter_priors
        self.discount = discount
        super().__init__(num_actions, name)

    def reset(self) -> None:
        self.parameters = tuple(p.copy() for p in self.parameter_priors)

    def __call__(self) -> int:
        action_probas = self.distribution(*self.parameters)
        action = np.argmax(action_probas)

        return action


class BernouillyThomsonSampling(ThomsonSampling):
    def __init__(
        self,
        num_actions: int,
        alpha_priors: NDArray,
        beta_priors: NDArray,
        discount: float = 1.0,
        name: str = "BernouilliThomsonSampling",
    ) -> None:
        assert (
            (num_actions,) == alpha_priors.shape == beta_priors.shape
        ), f"shape mismatch, got {(num_actions,)}, {alpha_priors.shape}, {beta_priors.shape}"

        super().__init__(num_actions, np.random.beta, (alpha_priors, beta_priors), discount, name)

    def update(self, action: int, reward: float) -> None:
        alpha, beta = self.parameters

        alpha *= self.discount
        beta *= self.discount

        alpha[action] += reward
        beta[action] += 1.0 - reward

        self.parameters = (alpha, beta)
