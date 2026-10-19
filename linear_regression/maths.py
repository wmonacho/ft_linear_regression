"""Math helpers for the linear regression model."""


def mean(arr: tuple[float, ...]) -> float:
    """Calculate the mean of a list of numbers."""
    if not arr:
        raise ValueError("The input array cannot be empty.")
    return sum(arr) / len(arr)


def standard_deviation(arr: tuple[float, ...]) -> float:
    """Calculate the standard deviation of a list of numbers."""
    if len(arr) < 2:
        raise ValueError("At least two data points are required to calculate standard deviation.")
    mean_value = mean(arr)
    variance = sum((x - mean_value) ** 2 for x in arr) / (len(arr) - 1)
    return variance ** 0.5


def normalize(arr: tuple[float, ...]) -> tuple[tuple[float, ...], float, float]:
    """Normalize a list of numbers using z-score normalization."""
    mean_value = mean(arr)
    std_dev = standard_deviation(arr)
    if std_dev == 0:
        return tuple(0.0 for _ in arr), mean_value, 1.0
    normalized = tuple((x - mean_value) / std_dev for x in arr)
    return normalized, mean_value, std_dev


def _check_same_length(actual: tuple[float, ...], predicted: tuple[float, ...]) -> None:
    """Check if two tuples have the same length."""
    if len(actual) != len(predicted):
        raise ValueError("The actual and predicted values must have the same length.")


def mean_absolute_error(actual: tuple[float, ...], predicted: tuple[float, ...]) -> float:
    """Calculate the mean absolute error between true and predicted values."""
    _check_same_length(actual, predicted)
    absolute_errors = []
    for real_price, predicted_price in zip(actual, predicted):
        absolute_errors.append(abs(real_price - predicted_price))
    return mean(tuple(absolute_errors))


def root_mean_squared_error(actual: tuple[float, ...], predicted: tuple[float, ...]) -> float:
    """Calculate the root mean squared error between true and predicted values."""
    _check_same_length(actual, predicted)
    squared_errors = []
    for real_price, predicted_price in zip(actual, predicted):
        squared_errors.append((real_price - predicted_price) ** 2)
    mean_squared_error = mean(tuple(squared_errors))
    return mean_squared_error ** 0.5


def r_squared(actual: tuple[float, ...], predicted: tuple[float, ...]) -> float:
    """
    Calculate the R-squared value between true and predicted values.

    Raises:
        ValueError: If all true values are identical (R² is undefined).
    """
    _check_same_length(actual, predicted)
    mean_actual = mean(actual)
    total_variance = sum((x - mean_actual) ** 2 for x in actual)
    residual_variance = sum(
        (real_price - predicted_price) ** 2
        for real_price, predicted_price in zip(actual, predicted)
    )
    if total_variance == 0:
        raise ValueError("R² is undefined when all true values are identical.")
    return 1 - residual_variance / total_variance
