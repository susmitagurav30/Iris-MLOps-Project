
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/iris_model.pkl")

FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_model_prediction():
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=FEATURES
    )

    prediction = model.predict(sample)
    assert prediction[0] in [0, 1, 2]


def test_model_prediction_count():
    model = joblib.load(MODEL_PATH)

    samples = pd.DataFrame([
        [5.1, 3.5, 1.4, 0.2],
        [6.0, 2.9, 4.5, 1.5],
        [6.5, 3.0, 5.8, 2.2]
    ], columns=FEATURES)

    predictions = model.predict(samples)
    assert len(predictions) == 3
