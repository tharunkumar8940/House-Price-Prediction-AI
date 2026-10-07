from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_FILE = RAW_DIR / "house_prices.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

n = 3000

locations = [
    "Chennai",
    "Coimbatore",
    "Bangalore",
    "Hyderabad",
    "Pune",
    "Mumbai",
    "Delhi",
    "Kochi",
]

data = pd.DataFrame({
    "house_id": range(1, n + 1),
    "location": np.random.choice(locations, n),
    "area_sqft": np.random.randint(600, 4001, n),
    "bedrooms": np.random.randint(1, 6, n),
    "bathrooms": np.random.randint(1, 5, n),
    "age_years": np.random.randint(0, 31, n),
    "parking_spaces": np.random.randint(0, 4, n),
    "floor": np.random.randint(1, 21, n),
    "total_floors": np.random.randint(1, 26, n),
    "distance_to_city_km": np.round(
        np.random.uniform(1, 30, n), 2
    ),
})

location_multiplier = {
    "Mumbai": 1.80,
    "Bangalore": 1.55,
    "Delhi": 1.45,
    "Hyderabad": 1.25,
    "Pune": 1.20,
    "Chennai": 1.15,
    "Kochi": 1.00,
    "Coimbatore": 0.90,
}

data["location_multiplier"] = data["location"].map(
    location_multiplier
)

base_price = (
    data["area_sqft"] * 6500
    * data["location_multiplier"]
)

bedroom_value = data["bedrooms"] * 250000
bathroom_value = data["bathrooms"] * 150000
parking_value = data["parking_spaces"] * 100000
floor_value = data["floor"] * 25000
age_penalty = data["age_years"] * 45000
distance_penalty = data["distance_to_city_km"] * 45000

noise = np.random.normal(0, 500000, n)

data["price"] = (
    base_price
    + bedroom_value
    + bathroom_value
    + parking_value
    + floor_value
    - age_penalty
    - distance_penalty
    + noise
)

data["price"] = data["price"].clip(lower=1000000)

data = data.drop(columns=["location_multiplier"])

data.to_csv(OUTPUT_FILE, index=False)

print("\nHOUSE PRICE DATASET GENERATED")
print("=" * 45)
print(f"File: {OUTPUT_FILE}")
print(f"Rows: {len(data):,}")
print(f"Columns: {len(data.columns)}")
print("\nLocations:")
print(data["location"].value_counts())

print("\nPrice range:")
print(f"Minimum: ₹{data['price'].min():,.0f}")
print(f"Maximum: ₹{data['price'].max():,.0f}")

print("\nDataset created successfully.")