
import pandas as pd
from sklearn.datasets import load_iris
from pathlib import Path


def prepare_dataset():

    # Step 1: Load Iris dataset
    iris = load_iris()

    # Step 2: Convert dataset into DataFrame
    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    # Step 3: Add target column
    df["target"] = iris.target

    # Step 4: Add flower species names
    df["species"] = df["target"].map(
        dict(enumerate(iris.target_names))
    )

    # Step 5: Create data folder
    Path("data").mkdir(exist_ok=True)

    # Step 6: Save dataset as CSV
    df.to_csv("data/iris.csv", index=False)

    # Step 7: Display dataset information
    print("Dataset created successfully!")
    print("Dataset shape:", df.shape)

    print("\nFirst five records:")
    print(df.head())

    print("\nSpecies distribution:")
    print(df["species"].value_counts())


if __name__ == "__main__":
    prepare_dataset()
