"""Train and evaluate a simple linear regression model for student scores."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "student_scores.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"


def load_dataset(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the CSV dataset and validate the required columns."""
    frame = pd.read_csv(csv_path)
    required_columns = {"Hours", "Scores"}
    missing_columns = required_columns.difference(frame.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}"
        )
    if frame[["Hours", "Scores"]].isnull().any().any():
        raise ValueError("The Hours and Scores columns must not contain missing values.")

    return frame


def train_and_evaluate(frame: pd.DataFrame):
    """Train the model and return the model, test data, predictions and metrics."""
    X = frame[["Hours"]]
    y = frame["Scores"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    results = {
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_test, predictions)),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    return model, X_test, y_test, predictions, results


def save_regression_plot(frame: pd.DataFrame, model: LinearRegression) -> Path:
    """Save a scatter plot and the fitted regression line."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.scatter(frame["Hours"], frame["Scores"], label="Actual data")
    line_x = np.linspace(frame["Hours"].min(), frame["Hours"].max(), 100)
    line_y = model.predict(pd.DataFrame({"Hours": line_x}))
    plt.plot(line_x, line_y, label="Regression line")
    plt.xlabel("Hours Studied")
    plt.ylabel("Scores")
    plt.title("Linear Regression: Study Hours vs Student Scores")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "regression_plot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def main() -> None:
    """Run the complete CSV extraction and linear regression workflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_dataset()

    print("FIRST FIVE RECORDS")
    print(frame.head().to_string(index=False))
    print("\nDATASET INFORMATION")
    print(frame.info())
    print("\nMISSING VALUES")
    print(frame.isnull().sum().to_string())
    print("\nDESCRIPTIVE STATISTICS")
    print(frame.describe().to_string())

    model, X_test, y_test, predictions, results = train_and_evaluate(frame)

    print("\nLINEAR REGRESSION RESULTS")
    print(f"Slope: {results['slope']:.4f}")
    print(f"Intercept: {results['intercept']:.4f}")
    print(
        f"Regression equation: Score = {results['intercept']:.3f} + "
        f"({results['slope']:.3f} x Hours)"
    )
    print(f"MAE: {results['mae']:.4f}")
    print(f"MSE: {results['mse']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"R-squared: {results['r2']:.4f}")

    comparison = pd.DataFrame({
        "Hours": X_test["Hours"].to_numpy(),
        "Actual_Score": y_test.to_numpy(),
        "Predicted_Score": np.round(predictions, 2),
    }).sort_values("Hours")
    comparison_path = OUTPUT_DIR / "predictions.csv"
    comparison.to_csv(comparison_path, index=False)

    summary = (
        "LINEAR REGRESSION RESULTS\n"
        f"Observations: {len(frame)}\n"
        f"Training observations: {results['train_rows']}\n"
        f"Testing observations: {results['test_rows']}\n\n"
        f"Regression equation: Score = {results['intercept']:.3f} + "
        f"({results['slope']:.3f} x Hours)\n"
        f"MAE: {results['mae']:.4f}\n"
        f"MSE: {results['mse']:.4f}\n"
        f"RMSE: {results['rmse']:.4f}\n"
        f"R-squared: {results['r2']:.4f} ({results['r2'] * 100:.2f}%)\n\n"
        "Caution: the test set contains only five observations, so interpret "
        "the evaluation metrics cautiously.\n"
    )
    (OUTPUT_DIR / "model_metrics.txt").write_text(summary, encoding="utf-8")

    plot_path = save_regression_plot(frame, model)
    print("\nACTUAL VS PREDICTED SCORES")
    print(comparison.to_string(index=False))
    print(f"\nSaved predictions to: {comparison_path.relative_to(ROOT_DIR)}")
    print(f"Saved metrics to: {(OUTPUT_DIR / 'model_metrics.txt').relative_to(ROOT_DIR)}")
    print(f"Saved plot to: {plot_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
