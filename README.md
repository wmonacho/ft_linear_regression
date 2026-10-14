# ft_linear_regression

Predict the price of a car from its mileage with a simple linear regression trained by gradient descent (42 project).

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## Installation

```sh
uv sync
```

## Usage

Train the model on a dataset (CSV with `km` and `price` columns):

```sh
uv run train.py data/data.csv
```

Predict a price (prompts for a mileage):

```sh
uv run predict.py
```

Before any training, θ0 = θ1 = 0, so every prediction is 0.

Or use the interactive menu (train, predict, show parameters, reset):

```sh
uv run menu.py
```

## How it works

- **Hypothesis:** `estimatePrice(mileage) = θ0 + θ1 × mileage`
- **Training:** gradient descent starting from θ0 = θ1 = 0. At each iteration, `tmpθ0` and `tmpθ1` are computed with the subject's formulas, then both θ are updated simultaneously.
- **Normalization:** mileages and prices are z-score normalized before the descent (large mileages would make it diverge), then θ0 and θ1 are converted back to the original scale.
- **Persistence:** `train.py` saves θ0 and θ1 to `thetas.json`, which `predict.py` reads. The file is generated, so it is not versioned.

## Project structure

| File | Role |
|---|---|
| `train.py` | Training program |
| `predict.py` | Prediction program |
| `menu.py` | Interactive launcher for both programs |
| `linear_regression/model.py` | Hypothesis, saving and loading of θ0/θ1 |
| `linear_regression/training.py` | Gradient descent and denormalization |
| `linear_regression/maths.py` | Mean, standard deviation, normalization |
| `linear_regression/dataset.py` | CSV loading and validation |
| `linear_regression/cli.py` | Terminal messages and argument parsing |

## Use of AI

<!-- TODO: describe what was generated or assisted by AI, as required by the subject. -->
