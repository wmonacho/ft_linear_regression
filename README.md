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

### Bonus

Plot the dataset and the regression line:

```sh
uv run plot.py data/data.csv
```

Measure the precision of the model on a dataset (MAE, RMSE, R²):

```sh
uv run precision.py data/data.csv
```

### Menu

Or use the interactive menu (train, predict, plot, precision, show parameters, reset):

```sh
uv run menu.py
```

## How it works

- **Hypothesis:** `estimatePrice(mileage) = θ0 + θ1 × mileage`
- **Training:** gradient descent starting from θ0 = θ1 = 0. At each iteration, `tmpθ0` and `tmpθ1` are computed with the subject's formulas, then both θ are updated simultaneously.
- **Normalization:** mileages and prices are z-score normalized before the descent (large mileages would make it diverge), then θ0 and θ1 are converted back to the original scale.
- **Precision:** MAE (average error in €), RMSE (same, but penalizes large errors) and R² (share of the price variation explained by the mileage, 1 = perfect). On `data/data.csv`: MAE ≈ 558 €, RMSE ≈ 668 €, R² ≈ 0.733.
- **Persistence:** `train.py` saves θ0 and θ1 to `thetas.json`, which `predict.py` reads. The file is generated, so it is not versioned.

## Project structure

| File | Role |
|---|---|
| `train.py` | Training program |
| `predict.py` | Prediction program |
| `plot.py` | Bonus: plot of the dataset and the regression line |
| `precision.py` | Bonus: precision of the model (MAE, RMSE, R²) |
| `menu.py` | Interactive launcher for all programs |
| `linear_regression/model.py` | Hypothesis, saving and loading of θ0/θ1 |
| `linear_regression/training.py` | Gradient descent and denormalization |
| `linear_regression/maths.py` | Mean, standard deviation, normalization, precision metrics |
| `linear_regression/dataset.py` | CSV loading and validation |
| `linear_regression/cli.py` | Terminal messages and argument parsing |

## Use of AI

Claude (Anthropic) was used as an assistant on this project:

- **Input validation:** wrote the validation of `thetas.json` (`load_thetas`) and of the CSV content (`load_dataset`).
- **Interactive menu:** `menu.py`.
- **Bonus:** `plot.py`, `precision.py`, and guidance on the precision metrics (MAE, RMSE, R²).
- **Documentation:** this README, docstrings and comments in the code.
- **Project management:** breakdown of the work into GitHub issues (tickets).
- **Learning:** explanations of the subject, gradient descent and normalization, and code reviews.

The core logic was written by me: the hypothesis (`estimate_price`), the gradient descent, the normalization/denormalization, saving the parameters and the prediction program.

