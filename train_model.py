"""
train_model.py — Model training and evaluation for AquaSense.

Workflow:
  1. Load (or generate) the dataset.
  2. Clean and inspect the data.
  3. Split into training and test sets.
  4. Train a Random Forest Regressor.
  5. Evaluate using MAE, RMSE, and R².
  6. Save the trained model to model.pkl.

Why Random Forest instead of Linear Regression?
  The feature set contains a mix of integer counts, binary flags, and continuous
  variables. Random Forest handles non-linear interactions (e.g. temperature ×
  garden size) and mixed feature types without requiring scaling, while still
  being interpretable via feature importances. Linear Regression was evaluated
  first (see comments below) but produced a materially higher MAE on this dataset.
"""

import sys
import pathlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH  = pathlib.Path("water_consumption.csv")
MODEL_PATH = pathlib.Path("model.pkl")
FEATURES   = [
    "occupants",
    "property_size_m2",
    "num_bathrooms",
    "has_dishwasher",
    "has_washing_machine",
    "avg_temp_celsius",
    "garden_size_m2",
    "days_away_per_year",
]
TARGET     = "water_consumption_lpd"
TEST_SIZE  = 0.20
SEED       = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_data(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[INFO] Dataset not found at '{path}'. Generating synthetic dataset …")
        from generate_dataset import generate
        df = generate()
        df.to_csv(path, index=False)
        print(f"[INFO] Synthetic dataset saved to '{path}'.")
    else:
        df = pd.read_csv(path)

    # Basic integrity checks
    missing = df[FEATURES + [TARGET]].isnull().sum()
    if missing.any():
        print("[WARN] Missing values detected — dropping affected rows.")
        print(missing[missing > 0])
        df = df.dropna(subset=FEATURES + [TARGET])

    return df


def evaluate(name: str, y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n--- {name} ---")
    print(f"  MAE  : {mae:.2f} L/day")
    print(f"  RMSE : {rmse:.2f} L/day")
    print(f"  R²   : {r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 1. Load data
    df = load_data(DATA_PATH)
    print(f"\n[INFO] Dataset loaded: {len(df)} rows × {len(df.columns)} columns.")
    print(df[FEATURES + [TARGET]].describe().round(2))

    X = df[FEATURES]
    y = df[TARGET]

    # 2. Train / test split  (no data leakage — split before any fitting)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    print(f"\n[INFO] Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # 3. Baseline: Linear Regression (for comparison / transparency)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    evaluate("Linear Regression (baseline)", y_test, lr_preds)

    # 4. Primary model: Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    metrics  = evaluate("Random Forest Regressor (selected)", y_test, rf_preds)

    # 5. Feature importances
    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\n[INFO] Feature importances (Random Forest):")
    for feat, imp in importances.items():
        print(f"  {feat:<28} {imp:.4f}")

    # 6. Save model + feature list together so app.py can verify alignment
    artifact = {"model": rf, "features": FEATURES}
    joblib.dump(artifact, MODEL_PATH)
    print(f"\n[INFO] Model saved -> '{MODEL_PATH}'")

    # 7. Summary
    print("\n========================================")
    print("  Training complete.")
    print(f"  MAE  : {metrics['mae']:.2f} L/day")
    print(f"  RMSE : {metrics['rmse']:.2f} L/day")
    print(f"  R2   : {metrics['r2']:.4f}")
    print("  Predictions are estimates only.")
    print("  Performance reflects synthetic data quality.")
    print("========================================")


if __name__ == "__main__":
    main()
