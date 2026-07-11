"""
Generates a realistic synthetic 'Used Car Price Prediction' dataset.
Mimics real-world Indian used-car listing data (like Cardekho/Quikr datasets)
so the project can demonstrate regression, categorical encoding and
feature engineering meaningfully.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

N = 5000

brands_models = {
    "Maruti":   ["Swift", "Baleno", "Alto", "Dzire", "WagonR", "Ertiga"],
    "Hyundai":  ["i10", "i20", "Creta", "Venue", "Verna"],
    "Honda":    ["City", "Amaze", "Jazz", "WR-V"],
    "Toyota":   ["Innova", "Fortuner", "Glanza", "Etios"],
    "Tata":     ["Nexon", "Tiago", "Altroz", "Harrier"],
    "Mahindra": ["XUV500", "Scorpio", "Bolero", "XUV300"],
    "Ford":     ["EcoSport", "Figo", "Endeavour"],
    "Volkswagen": ["Polo", "Vento"],
    "BMW":      ["3 Series", "5 Series", "X1"],
    "Audi":     ["A4", "A6"],
}

fuel_types = ["Petrol", "Diesel", "CNG", "Electric"]
transmissions = ["Manual", "Automatic"]
owner_types = ["First", "Second", "Third", "Fourth & Above"]
locations = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Pune", "Hyderabad",
             "Kolkata", "Lucknow", "Jaipur", "Ahmedabad"]

# base price (in lakhs INR) & segment info per brand (rough realism)
brand_base_price = {
    "Maruti": 6, "Hyundai": 7.5, "Honda": 8.5, "Toyota": 11,
    "Tata": 7, "Mahindra": 9, "Ford": 8, "Volkswagen": 9,
    "BMW": 35, "Audi": 40
}

rows = []
current_year = 2026

for _ in range(N):
    brand = np.random.choice(list(brands_models.keys()),
                              p=[0.22, 0.18, 0.12, 0.08, 0.13, 0.1, 0.06, 0.05, 0.03, 0.03])
    model = np.random.choice(brands_models[brand])
    year = np.random.randint(2008, 2025)
    age = current_year - year

    fuel = np.random.choice(fuel_types, p=[0.5, 0.35, 0.1, 0.05])
    transmission = np.random.choice(transmissions, p=[0.72, 0.28])
    owner = np.random.choice(owner_types, p=[0.55, 0.28, 0.12, 0.05])
    location = np.random.choice(locations)

    # km driven correlates with age, with noise
    km_driven = max(500, int(np.random.normal(age * 11000, 8000)))

    engine_cc = int(np.random.choice([796, 998, 1197, 1248, 1498, 1997, 2494, 2996]))
    mileage_kmpl = round(max(8, np.random.normal(18 - engine_cc / 500, 2)), 1)
    power_bhp = round(engine_cc / 15 + np.random.normal(0, 5), 1)
    seats = int(np.random.choice([4, 5, 6, 7], p=[0.05, 0.75, 0.05, 0.15]))

    base = brand_base_price[brand]

    # ---- price formula (ground truth signal + noise), in Lakhs INR ----
    price = base
    price -= age * (base * 0.065)                 # depreciation with age
    price -= (km_driven / 100000) * (base * 0.12)  # depreciation with usage
    price += (power_bhp / 100) * 2.0               # more power -> higher price
    price += 1.5 if transmission == "Automatic" else 0
    price -= {"First": 0, "Second": 0.5, "Third": 1.0, "Fourth & Above": 1.6}[owner]
    price += {"Petrol": 0, "Diesel": 0.4, "CNG": -0.3, "Electric": 3.0}[fuel]
    price += np.random.normal(0, base * 0.08)      # random market noise
    price = max(0.8, round(price, 2))

    rows.append([
        brand, model, year, km_driven, fuel, transmission, owner,
        mileage_kmpl, engine_cc, power_bhp, seats, location, price
    ])

df = pd.DataFrame(rows, columns=[
    "brand", "model", "year", "km_driven", "fuel_type", "transmission",
    "owner_type", "mileage_kmpl", "engine_cc", "power_bhp", "seats",
    "location", "selling_price_lakhs"
])

# introduce a few realistic missing values (like real scraped data)
for col in ["mileage_kmpl", "power_bhp", "seats"]:
    idx = np.random.choice(df.index, size=int(0.02 * N), replace=False)
    df.loc[idx, col] = np.nan

df.to_csv("/home/claude/car_price_prediction/data/used_cars.csv", index=False)
print("Saved dataset:", df.shape)
print(df.head())
