import numpy as np


def average_reward_per_step(returns: list[float], num_frames: list[int]) -> float:
    avgs = []
    if len(returns) != len(num_frames):
        raise ValueError("The length of the returns and num_frames lists must be the same.")
    for i in range(len(returns)):
        if num_frames[i] > 0:
            avgs.append(returns[i] / num_frames[i])
    return float(np.mean(avgs)) if avgs else float("nan")


def average_discounted_return(returns: list[float], num_frames: list[int], discount: float):
    discounted_returns = []
    if len(returns) != len(num_frames):
        raise ValueError("The length of the returns and num_frames lists must be the same.")
    for i in range(len(returns)):
        if num_frames[i] > 0:
            discounted_returns.append(returns[i] * (discount ** (num_frames[i] - 1)))
    return float(np.mean(discounted_returns)) if discounted_returns else float("nan")
