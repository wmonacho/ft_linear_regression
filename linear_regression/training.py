"""Training of the linear model with gradient descent."""

from linear_regression.dataset import Data
from linear_regression.maths import normalize
from linear_regression.model import estimate_price


def train_model(
    data: tuple[Data, ...], learning_rate: float = 0.01, iterations: int = 1000
) -> tuple[float, float]:
    """
    Train the model on normalized data, then convert θ back to the original scale.

    Returns:
        The learned parameters (theta0, theta1).
    """
    x = tuple(d.kilometers for d in data)
    y = tuple(d.price for d in data)

    x_norm, x_mean, x_std = normalize(x)
    y_norm, y_mean, y_std = normalize(y)

    theta0_norm, theta1_norm = _gradient_descent(x_norm, y_norm, learning_rate, iterations)

    theta1 = theta1_norm * y_std / x_std
    theta0 = y_mean + theta0_norm * y_std - theta1 * x_mean
    return theta0, theta1


def _gradient_descent(
    x: tuple[float, ...], y: tuple[float, ...], learning_rate: float, iterations: int
) -> tuple[float, float]:
    """Run gradient descent with the subject's formulas, starting from θ0 = θ1 = 0."""
    theta0, theta1 = 0.0, 0.0
    sample_count = len(x)
    for _ in range(iterations):
        sum_errors = 0.0
        sum_errors_km = 0.0
        for i in range(sample_count):
            error = estimate_price(x[i], theta0, theta1) - y[i]
            sum_errors += error
            sum_errors_km += error * x[i]
        tmp_theta0 = learning_rate * sum_errors / sample_count
        tmp_theta1 = learning_rate * sum_errors_km / sample_count
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1
    return theta0, theta1