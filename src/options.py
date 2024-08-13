from datetime import date
from enum import Enum
from typing import Optional
import numpy as np

QtyType = int
PriceType = float
TickerType = str


class OptionType(Enum):
    CALL = 1
    PUT = 2


class Stock:
    def __init__(self, ticker: TickerType, spot_price: PriceType):
        self.ticker = ticker
        self.spot_price = spot_price


class Option:
    def __init__(self,
                 option_type: OptionType,
                 underlying: Stock,
                 dividend_yield: Optional[PriceType],
                 qty: QtyType,
                 strike_price: PriceType,
                 expiration_date: date):
        self.option_type = option_type

        self.underlying = underlying
        self.dividend_yield = dividend_yield or 0

        self.qty = qty
        self.strike_price = strike_price
        self.expiration_date = expiration_date

    def is_call(self) -> bool:
        return self.option_type == OptionType.CALL

    def is_put(self) -> bool:
        return not self.is_call()

    def simple_payoff(self, spot_price: PriceType) -> PriceType:
        if self.is_call():
            return max(spot_price - self.strike_price, 0)
        return max(self.strike_price - spot_price, 0)

    def payoff(self, spot_price: PriceType, time_to_maturity_in_years: float = 0) -> PriceType:
        raise NotImplementedError()


class EuropeanOption(Option):
    def payoff(self, spot_price: PriceType, time_to_maturity_in_years: float = 0) -> PriceType:
        return 0 if time_to_maturity_in_years > 1e-8 else super().simple_payoff(spot_price)


class AmericanOption(Option):
    def payoff(self, spot_price: PriceType, time_to_maturity_in_years: float = 0) -> PriceType:
        return super().simple_payoff(spot_price)


class GeometricBrownianMotion:
    @staticmethod
    def simulate(init: float,
                 mu: float,
                 sigma: float,
                 dt: float,
                 num_steps: int,
                 num_paths: int,
                 seed: int | None = None) -> np.ndarray:

        random_state = np.random.RandomState(seed)

        multiplicative_factor = np.exp((mu - 0.5 * (sigma ** 2)) * dt)
        noise_factor = sigma * np.sqrt(dt)

        noise = np.exp(noise_factor * random_state.normal(0, 1, (num_steps, num_paths)))

        paths = np.empty((num_steps + 1, num_paths))
        paths[0, :] = init

        for t in range(1, num_steps + 1):
            paths[t, :] = paths[t - 1, :] * multiplicative_factor * noise[t - 1, :]

        return paths

    @staticmethod
    def estimate_parameters(series: np.ndarray, dt: float) -> tuple[float, float]:
        log_series_diff = np.diff(np.log(series), axis=0)

        observed_mean = np.mean(log_series_diff)
        observed_var = np.var(log_series_diff)

        estimated_mu = (observed_mean + observed_var / 2) / dt
        estimated_sigma = np.sqrt(observed_var / dt)

        return estimated_mu, estimated_sigma
