"""Linear hypothesis shared by both programs, and persistence of its parameters."""

import json
from pathlib import Path

THETAS_FILE = Path("thetas.json")


def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
    """Hypothesis from the subject: estimatePrice(mileage) = θ0 + θ1 * mileage."""
    return theta0 + theta1 * mileage


def save_thetas(theta0: float, theta1: float, path: Path = THETAS_FILE) -> None:
    """Save θ0 and θ1 so the prediction program can use them."""
    

    with open(path, "w") as f:
        json.dump({"theta0": theta0, "theta1": theta1}, f)


def load_thetas(path: Path = THETAS_FILE) -> tuple[float, float]:
    """Load θ0 and θ1, defaulting to (0, 0) before any training."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return (data["theta0"], data["theta1"])
    except (FileNotFoundError, json.JSONDecodeError):
        return (0.0, 0.0)

