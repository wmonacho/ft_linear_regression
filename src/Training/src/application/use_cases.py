import numpy as np
from Training.src.domain.data import Data
from Training.src.application.math import math_operations


def train_model(data: tuple[Data, ...], learning_rate: float = 0.01, iterations: int = 1000) -> Tuple[float, float]:
    """
    Trains a linear regression model using gradient descent.

    Args:
        data (Tuple[Data, ...]): A tuple of Data objects containing training data.
        learning_rate (float): The learning rate for gradient descent.
        iterations (int): The number of iterations for training.

    Returns:
        Tuple[float, float]: The learned parameters (theta_0, theta_1) of the linear model.
    """
    x = tuple([d.kilometers for d in data])
    y = tuple([d.price for d in data])
    
    x_norm, x_mean, x_std = math_operations.normalize(x)
    y_norm, y_mean, y_std = math_operations.normalize(y)
    
    theta_0 = 0.0
    theta_1 = 0.0
    n = len(data)

    theta0_norm, theta1_norm = _gradient_descent(
        x_norm, y_norm, learning_rate, iterations
    )

    theta_1 = theta1_norm * y_std / x_std
    theta_0 = y_mean + theta0_norm * y_std - theta_1 * x_mean
    
    return theta_0, theta_1