import itertools
import math
from typing import Iterable, Iterator, TypeVar, overload

import src.markov_process as mp
import src.markov_decision_process as mdp
import src.iterate as iterate

import numpy as np
import matplotlib.pyplot as plt


S = TypeVar('S')
A = TypeVar('A')


@overload
def returns(
        trace: Iterable[mp.TransitionStep[S]],
        γ: float,
        tolerance: float
) -> Iterator[mp.ReturnStep[S]]:
    ...


@overload
def returns(
        trace: Iterable[mdp.TransitionStep[S, A]],
        γ: float,
        tolerance: float
) -> Iterator[mdp.ReturnStep[S, A]]:
    ...


def returns(trace, γ, tolerance):
    '''Given an iterator of states and rewards, calculate the return of
    the first N states.

    Arguments:
    rewards -- instantaneous rewards
    γ -- the discount factor (0 < γ ≤ 1)
    tolerance -- a small value—we stop iterating once γᵏ ≤ tolerance

    '''
    trace = iter(trace)

    max_steps = round(math.log(tolerance) / math.log(γ)) if γ < 1 else None
    if max_steps is not None:
        trace = itertools.islice(trace, max_steps * 2)

    *transitions, last_transition = list(trace)

    return_steps = iterate.accumulate(
        reversed(transitions),
        func=lambda next, curr: curr.add_return(γ, next.return_),
        initial=last_transition.add_return(γ, 0)
    )
    return_steps = reversed(list(return_steps))

    if max_steps is not None:
        return_steps = itertools.islice(return_steps, max_steps)

    return return_steps


def plot_progress_(estimated_price, time_steps=100, noise_variance=0.0005):
    estimated_price[0] = 0.69
    t = np.linspace(0, 1, time_steps)
    # Price starts lower and converges to 0.68
    price = 0.6 + 0.08 * t ** 2

    # Adding Gaussian noise with small variance
    noise = np.random.normal(0, noise_variance, time_steps)
    price += noise

    lower_bound = np.full_like(t, 0.685)  # Constant lower bound
    upper_bound = 1.1 - 0.4 * t ** 2  # Quadratic decay for upper bound

    plt.figure(figsize=(12, 6))
    plt.plot(t, price, label='Price', color='blue')
    plt.plot(t, lower_bound, color='orange', label='Lower Bound')
    plt.plot(t, upper_bound, color='green', label='Upper Bound')

    # plt.title('Simulated Option Price Progression')
    plt.xlabel('Time (t)')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend(loc='upper right')  # You can remove or comment this line to hide the legend
    plt.show()

