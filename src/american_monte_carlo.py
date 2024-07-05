import numpy as np
from matplotlib import pyplot as plt
from src.samplers import AbstractSampler
from tqdm import tqdm_notebook as tqdm
from sklearn.base import TransformerMixin
from IPython.display import display, clear_output


def plot_progress(sampler, bar, price_history, lower_bound, upper_bound):
    clear_output(wait=True)
    display(bar.container)
    plt.ticklabel_format(style='plain', useOffset=False)
    plt.plot(sampler.time_grid, price_history, label='price')
    plt.plot(sampler.time_grid, lower_bound, label='lower bound')
    plt.plot(sampler.time_grid, upper_bound, label="upper bound")
    plt.legend()
    plt.title("$Option_t$")
    plt.xlabel("$t$")
    plt.ylabel("price")
    plt.grid()
    plt.show()


class AmericanMonteCarlo:
    def __init__(
            self,
            sampler: AbstractSampler,
            basis_functions_transformer: TransformerMixin,
            regularization_alpha=1e-4
    ):
        self.sampler = sampler
        self.regularization_alpha = regularization_alpha
        self.basis_functions_transformer = basis_functions_transformer
        self.price_history = None
        self.option_price = None
        self.result = {}

    def price(self, test=False, quiet=False):
        self.sampler.sample()
        if not quiet:
            self.sampler.plot(cnt=10, plot_mean=True, y="payoff, discount_factor, markov_state")
        discounted_payoff = self.sampler.payoff * self.sampler.discount_factor  # may be incorrect for a general case
        self.option_price = discounted_payoff[:, -1].copy()
        weights = [None] * self.sampler.cnt_times
        self.price_history = [None] * (self.sampler.cnt_times - 1) + [self.option_price.mean()]

        lower_bound = np.zeros(self.sampler.cnt_times)
        upper_bound = np.zeros(self.sampler.cnt_times)
        for i in range(self.sampler.cnt_times):
            lower_bound[i] = discounted_payoff[:, i:].mean(axis=0).max()
            upper_bound[i] = discounted_payoff[:, i:].max(axis=1).mean()

        bar = tqdm(range(self.sampler.cnt_times - 2, -1, -1))
        for time_index in bar:
            if time_index == 0:
                continuation_value = np.ones(self.sampler.cnt_trajectories) * np.mean(self.option_price)
                in_the_money_indices = np.arange(self.sampler.cnt_trajectories, dtype=int)
            else:
                in_the_money_indices = np.where(discounted_payoff[:, time_index] > 1e-9)[0]
                if (len(in_the_money_indices) / self.sampler.cnt_trajectories < 1e-3 or
                        len(in_the_money_indices) < 2 or test and weights[time_index] is None):
                    self.price_history[time_index] = self.option_price.mean()
                    continue
                features = self.sampler.markov_state[in_the_money_indices, time_index]
                transformed = self.basis_functions_transformer.fit_transform(features)
                if not test:
                    regularization = np.eye(transformed.shape[1], dtype=float) * self.regularization_alpha
                    inv = np.linalg.pinv((transformed.T @ transformed + regularization), rcond=1e-4)
                    weights[time_index] = inv @ transformed.T @ self.option_price[in_the_money_indices]
                continuation_value = transformed @ weights[time_index]

            indicator = discounted_payoff[in_the_money_indices, time_index] > continuation_value.reshape(-1)
            self.option_price[in_the_money_indices] = \
                (indicator * discounted_payoff[in_the_money_indices, time_index].copy() +
                 ~indicator * self.option_price[in_the_money_indices])
            self.price_history[time_index] = self.option_price.mean()
            if not quiet and time_index % 10 == 0:
                plot_progress(self.sampler, bar, self.price_history, lower_bound, upper_bound)

        if not quiet:
            self.sampler.plot(cnt=10, plot_mean=True, y="payoff, discount_factor, markov_state")

        key = "test" if test else "train"
        self.result[key] = {
            "price": float(self.option_price.mean()),
            "upper_bound": float(discounted_payoff.max(axis=1).mean()),
            "lower_bound": float(discounted_payoff.mean(axis=0).max()),
            "std": float(self.option_price.std())
        }

        return self.result
