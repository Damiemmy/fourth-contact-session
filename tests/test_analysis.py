from pathlib import Path

import pandas as pd
import pytest

from src.linear_regression import load_dataset, train_and_evaluate

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_has_expected_columns_and_rows():
    frame = load_dataset(ROOT / "data" / "student_scores.csv")
    assert {"Hours", "Scores"}.issubset(frame.columns)
    assert len(frame) == 25


def test_model_returns_expected_metrics():
    frame = load_dataset(ROOT / "data" / "student_scores.csv")
    model, X_test, y_test, predictions, results = train_and_evaluate(frame)

    assert len(X_test) == 5
    assert len(y_test) == 5
    assert len(predictions) == 5
    assert results["slope"] > 0
    assert results["r2"] > 0.9


def test_missing_required_column_raises_error(tmp_path):
    invalid_csv = tmp_path / "invalid.csv"
    pd.DataFrame({"Hours": [1, 2], "Marks": [10, 20]}).to_csv(
        invalid_csv, index=False
    )

    with pytest.raises(ValueError, match="Scores"):
        load_dataset(invalid_csv)
