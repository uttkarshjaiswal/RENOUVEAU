
import pickle
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Keep the command-line workflow's paths stable regardless of the launch folder.
SCRIPT_FOLDER = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_FOLDER / "data" / "signs.csv"
MODEL_FILE = SCRIPT_FOLDER / "sign_model.pkl"


def train_model(data_file=DATA_FILE, model_file=MODEL_FILE):
    data_file = Path(data_file)
    model_file = Path(model_file)
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found: {data_file}")

    print("Loading collected sign data...")
    df = pd.read_csv(data_file)
    if df.empty:
        raise ValueError("signs.csv is empty. Collect data first.")

    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    X = df.drop(columns=["label"])
    y = df["label"].astype(str)
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isnull().any().any() or not X.notna().all().all():
        raise ValueError(
            "Dataset contains missing or non-numeric features. "
            "Check your signs.csv file."
        )

    if len(X.columns) != 63:
        raise ValueError(
            f"Expected 63 landmark features, found {len(X.columns)}. "
            "Check that your CSV contains 21 hand landmarks with x, y, z."
        )

    if y.nunique() < 2:
        raise ValueError("At least two different sign labels are required.")

    counts = y.value_counts()
    too_small = counts[counts < 2]
    if not too_small.empty:
        raise ValueError(
            "Every label needs at least 2 samples for training/testing:\n"
            f"{too_small.to_string()}\n"
            "Collect more samples for these labels and try again."
        )

    print(f"Total samples: {len(df)}")
    print(f"Features per sample: {X.shape[1]}")
    print("\nSamples per sign:")
    print(y.value_counts())

    test_size = max(0.20, y.nunique() / len(y))
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    print("\nTraining SignBridge model...")
    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nTest accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    model_data = {
        "model": model,
        "feature_count": X.shape[1],
        "labels": sorted(y.unique().tolist())
    }

    model_file.parent.mkdir(parents=True, exist_ok=True)
    with model_file.open("wb") as file:
        pickle.dump(model_data, file)

    print(f"\nSUCCESS! Model saved to: {model_file}")
    print("Recognized labels:", model_data["labels"])
    return model_data["labels"]


def main():
    try:
        train_model()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")

if __name__ == "__main__":
    main()