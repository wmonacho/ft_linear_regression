"""Linear hypothesis shared by both programs, and persistence of its parameters."""

import json
import math
from pathlib import Path

THETAS_FILE = Path("thetas.json")


def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
    """Hypothesis from the subject: estimatePrice(mileage) = θ0 + θ1 * mileage."""
    return theta0 + theta1 * mileage


def save_thetas(theta0: float, theta1: float, path: Path = THETAS_FILE) -> None:
    """
    Save θ0 and θ1 so the prediction program can use them.

    Raises:
        ValueError: If the file cannot be written.
    """
    try:
        with open(path, "w") as f:
            json.dump({"theta0": theta0, "theta1": theta1}, f)
    except OSError as e:
        raise ValueError(f"Cannot write {path}: {e}") from e


def load_thetas(path: Path = THETAS_FILE) -> tuple[float, float]:
    """
    Load θ0 and θ1, defaulting to (0, 0) before any training.

    Raises:
        ValueError: If the file exists but cannot be read or is invalid.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return 0.0, 0.0
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Cannot read {path}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Invalid {path}: expected a JSON object.")
    return _read_theta(data, "theta0", path), _read_theta(data, "theta1", path)


def _read_theta(data: dict[str, object], key: str, path: Path) -> float:
    if key not in data:
        raise ValueError(f"Invalid {path}: missing '{key}'.")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid {path}: '{key}' must be a finite number.")
    return float(value)
