from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "house_prices_clean.csv"
)

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("\nHOUSE PRICE PREDICTION - MODEL TRAINING")
print("=" * 55)


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

print(f"Records: {len(df):,}")


# --------------------------------------------------
# Features and target
# --------------------------------------------------

target = "price"

X = df.drop(columns=[target])
y = df[target]


# House ID is only an identifier
X = X.drop(columns=["house_id"])


categorical_features = [
    "location"
]

numeric_features = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "parking_spaces",
    "floor",
    "total_floors",
    "distance_to_city_km",
]


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features,
        ),
    ]
)


# --------------------------------------------------
# Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)


print(f"Training records: {len(X_train):,}")
print(f"Testing records: {len(X_test):,}")


# --------------------------------------------------
# Models
# --------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=250,
        max_depth=18,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=4,
        random_state=42,
    ),
}


results = []


# --------------------------------------------------
# Train and evaluate
# --------------------------------------------------

for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    results.append(
        {
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        }
    )

    print(
        f"MAE : ₹{mae:,.2f}"
    )

    print(
        f"RMSE: ₹{rmse:,.2f}"
    )

    print(
        f"R²  : {r2:.4f}"
    )


# --------------------------------------------------
# Results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2",
    ascending=False,
)

results_file = (
    OUTPUT_DIR
    / "model_comparison.csv"
)

results_df.to_csv(
    results_file,
    index=False,
)


# --------------------------------------------------
# Select best model
# --------------------------------------------------

best_model_name = results_df.iloc[0]["model"]

print("\n")
print("=" * 55)
print("MODEL COMPARISON")
print("=" * 55)

print(
    results_df.to_string(
        index=False
    )
)

print("\nBest model:")
print(best_model_name)


# --------------------------------------------------
# Retrain best model
# --------------------------------------------------

best_model = models[
    best_model_name
]

best_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            best_model,
        ),
    ]
)

best_pipeline.fit(
    X_train,
    y_train,
)


# --------------------------------------------------
# Save model
# --------------------------------------------------

model_file = (
    MODEL_DIR
    / "house_price_model.pkl"
)

joblib.dump(
    best_pipeline,
    model_file,
)


# --------------------------------------------------
# Save model metadata
# --------------------------------------------------

metadata = {
    "best_model": best_model_name,
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test)),
    "features": numeric_features
    + categorical_features,
    "target": target,
    "test_size": 0.20,
    "random_state": 42,
}


metadata_file = (
    MODEL_DIR
    / "model_metadata.json"
)

with open(
    metadata_file,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metadata,
        file,
        indent=4,
    )


print("\nMODEL TRAINING COMPLETE")
print("=" * 55)

print(
    f"Saved model: {model_file}"
)

print(
    f"Saved comparison: {results_file}"
)

print(
    f"Saved metadata: {metadata_file}"
)

print("\nNEXT MODULE READY: STREAMLIT APPLICATION")