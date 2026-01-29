""" math functions for linear regression model """

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

