from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "raw" / "house_prices.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "house_prices_clean.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


print("\nHOUSE PRICE DATA PREPROCESSING")
print("=" * 45)


# Load dataset
df = pd.read_csv(INPUT_FILE)

print(f"Raw records: {len(df):,}")
print(f"Raw columns: {len(df.columns)}")


# Remove duplicate records
duplicates = df.duplicated().sum()
df = df.drop_duplicates()

print(f"Duplicate records removed: {duplicates}")


# Check required columns
required_columns = [
    "house_id",
    "location",
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "parking_spaces",
    "floor",
    "total_floors",
    "distance_to_city_km",
    "price",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# Convert numerical columns
numeric_columns = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "parking_spaces",
    "floor",
    "total_floors",
    "distance_to_city_km",
    "price",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Remove invalid numerical values
df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

before_missing = len(df)

df = df.dropna(
    subset=numeric_columns + ["location"]
)

removed_missing = before_missing - len(df)

# Business validation
df = df[df["area_sqft"] > 0]
df = df[df["bedrooms"] > 0]
df = df[df["bathrooms"] > 0]
df = df[df["price"] > 0]
df = df[df["age_years"] >= 0]
df = df[df["parking_spaces"] >= 0]
df = df[df["distance_to_city_km"] >= 0]
df = df[df["floor"] > 0]
df = df[df["total_floors"] > 0]

# Floor cannot exceed total floors
df = df[
    df["floor"] <= df["total_floors"]
]


# Reset index
df = df.reset_index(drop=True)


# Save cleaned dataset
df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nPREPROCESSING COMPLETE")
print("=" * 45)
print(f"Clean records: {len(df):,}")
print(f"Final columns: {len(df.columns)}")
print(f"Missing/invalid rows removed: {removed_missing}")

print("\nLocation distribution:")
print(df["location"].value_counts())

print("\nPrice statistics:")
print(
    f"Average price: ₹{df['price'].mean():,.0f}"
)
print(
    f"Minimum price: ₹{df['price'].min():,.0f}"
)
print(
    f"Maximum price: ₹{df['price'].max():,.0f}"
)

print("\nOutput:")
print(OUTPUT_FILE)

print("\nNEXT MODULE READY: MODEL TRAINING")