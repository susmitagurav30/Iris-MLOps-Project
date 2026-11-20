
import pandas as pd
import joblib
import json
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_model():

    # 1. Load dataset
    df = pd.read_csv("data/iris.csv")

    # 2. Separate features and target
    X = df.drop(columns=["target", "species"])
    y = df["target"]

    # 3. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4. Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    # 5. Configure MLflow
    # Configure SQLite database for MLflow
    mlflow.set_tracking_uri("file:./mlruns")

    # Create or select experiment
    mlflow.set_experiment("Iris-Classification")
    # 6. Start MLflow experiment
    with mlflow.start_run(run_name="RandomForest-Experiment-1"):

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # 7. Log parameters
        mlflow.log_param("algorithm", "Random Forest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("random_state", 42)

        # 8. Log evaluation metrics
        mlflow.log_metric("accuracy", accuracy)

        # 9. Save trained model
        Path("models").mkdir(exist_ok=True)

        joblib.dump(model, "models/iris_model.pkl")

        # 10. Log model in MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            name="iris_model",
            skops_trusted_types=[
            "sklearn.tree._tree.Tree"
            ]
        )

        # 11. Save metrics
        metrics = {
            "algorithm": "Random Forest",
            "accuracy": float(accuracy),
            "training_samples": len(X_train),
            "testing_samples": len(X_test)
        }

        with open("metrics.json", "w") as file:
            json.dump(metrics, file, indent=4)

        # Display results
        print("MLflow experiment completed!")
        print("Experiment: Iris-Classification")
        print("Accuracy:", accuracy * 100, "%")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nModel saved successfully!")


if __name__ == "__main__":
    train_model()
