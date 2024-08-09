import numpy as np
from abc import abstractmethod
from src.plotting_utils import plot_trajectories
from tqdm import tqdm_notebook as tqdm
import warnings


# class SamplingResult:
#     def __init__(
#             self,
#             markov_state: np.ndarray,
#             payoff: np.ndarray,
#             discount_factor: np.ndarray
#     ):
#         self.markov_state = markov_state
#         self.payoff = payoff
#         self.discount_factor = discount_factor


class AbstractSampler:
    def __init__(
            self,
            cnt_trajectories: int,
            cnt_times: int | None = None,
            t: float | None = None,
            time_grid: np.ndarray | None = None,
            seed: int | None = None,
            *args,
            **kwargs
    ):
        if (cnt_times is None or t is None) and time_grid is None:
            raise ValueError("You must specify time and cnt_times or time_grid")
        if time_grid is None:
            self.time_grid = np.linspace(0, t, cnt_times)
        else:
            self.time_grid = time_grid
        self.time_deltas = np.diff(self.time_grid)
        self.cnt_trajectories = cnt_trajectories
        self.cnt_times = len(self.time_grid)
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.markov_state = None
        self.payoff = None
        self.discount_factor = None

    @abstractmethod
    def sample(self) -> None:
        raise NotImplementedError()

    def plot(
            self,
            cnt: int,
            plot_mean: False = False,
            y: str = "payoff"
    ):
        if "markov_state" in y:
            if self.markov_state.shape[2] != 1:
                warnings.warn("We cannot plot >=2d processes", UserWarning)
            plot_trajectories(self.time_grid, self.markov_state[:, :, 0], cnt, "Markov state", "Markov State",
                              plot_mean)
        if "payoff" in y:
            plot_trajectories(self.time_grid, self.payoff, cnt, "payoff", "Payoff", plot_mean)
        if "discount_factor" in y:
            plot_trajectories(self.time_grid, self.discount_factor, cnt, "discount factor",
                              "Discount Factor", plot_mean)


class WienerRainbowPutOptionSampler(AbstractSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigmas = kwargs.get("sigmas")
        self.strike = kwargs.get("strike")
        self.dim = len(self.sigmas)

    def sample(self) -> None:
        normals = self.random_state.normal(0, 1, (self.dim, self.cnt_trajectories, self.cnt_times - 1))
        self.markov_state = np.zeros((self.dim, self.cnt_trajectories, self.cnt_times), dtype=float)
        for i in range(self.dim):
            for j in tqdm(range(self.cnt_trajectories)):
                self.markov_state[i][j] = np.zeros(len(self.time_deltas) + 1, dtype=float)
                self.markov_state[i][j][1:] = self.sigmas[i] * np.cumsum(normals[i][j] * np.sqrt(self.time_deltas))
        self.payoff = np.clip(self.strike - np.min(self.markov_state, axis=0), 0, 1e20)
        self.discount_factor = np.ones_like(self.payoff)
        self.markov_state = np.transpose(self.markov_state, (1, 2, 0))


class GeometricBrownianMotionPutSampler(AbstractSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = kwargs.get("sigma")
        self.strike = kwargs.get("strike")
        self.mu = kwargs.get("mu")
        self.asset0 = kwargs.get("asset0")
        self.t = kwargs.get('t', 1)  # Make sure 't' is initialized

    def sample(self) -> None:
        normals = self.random_state.normal(0, 1, (self.cnt_trajectories, self.cnt_times - 1))
        self.markov_state = np.zeros((self.cnt_trajectories, self.cnt_times, 1), dtype=float)
        self.markov_state[:, 0, 0] = self.asset0
        for t in tqdm(range(1, self.cnt_times)):
            dt = self.time_deltas[t - 1]
            self.markov_state[:, t, 0] = self.markov_state[:, t - 1, 0] * np.exp(
                (self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * normals[:, t - 1]
            )
        self.payoff = np.clip(self.strike - self.markov_state[:, :, 0], 0, 1e20)
        self.discount_factor = np.repeat(
            np.exp(- self.mu * self.time_grid).reshape((1, -1)), self.cnt_trajectories, axis=0
        )
