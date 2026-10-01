"""Small dependency-free helpers shared by the manuscript plot generators."""

import math


def fit_power_law(points, min_points=4):
    """Fit y = coefficient * x**exponent in log space.

    Returns (exponent, coefficient), or None when too few positive points are
    available. This is ordinary unweighted least squares, matching polyfit's
    degree-one behavior used by the earlier generators.
    """

    positive = [(x, y) for x, y in points if x > 0 and y > 0]
    if len(positive) < min_points:
        return None

    log_x = [math.log(x) for x, _ in positive]
    log_y = [math.log(y) for _, y in positive]
    mean_x = sum(log_x) / len(log_x)
    mean_y = sum(log_y) / len(log_y)
    denominator = sum((value - mean_x) ** 2 for value in log_x)
    if denominator == 0:
        return None
    exponent = (
        sum(
            (x_value - mean_x) * (y_value - mean_y)
            for x_value, y_value in zip(log_x, log_y, strict=True)
        )
        / denominator
    )
    intercept = mean_y - exponent * mean_x
    return exponent, math.exp(intercept)
