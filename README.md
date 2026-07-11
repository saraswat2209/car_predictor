# ResaleIQ — Used Car Price Prediction

A mini-project that estimates the resale value of used cars using machine
learning regression, built with a Flask backend and a web frontend.

## What this project demonstrates
- **Regression**: Linear Regression, Ridge, Decision Tree, Random Forest,
  Gradient Boosting — trained and compared.
- **Categorical encoding**: Label Encoding for brand, model, fuel type,
  transmission, owner type, and location.
- **Feature engineering**: car age, km driven per year, power-to-engine
  ratio, brand average price (target-style encoding), luxury flag, and
  log-transformed mileage.

## Project structure
```
car_price_prediction/
├── data/
│   ├── generate_data.py     # creates the synthetic dataset
│   └── used_cars.csv        # generated dataset
├── model/
│   ├── train_model.py       # full ML pipeline
│   ├── best_model.pkl       # trained Gradient Boosting model
│   ├── scaler.pkl, label_encoders.pkl, brand_avg_price.pkl
│   ├── feature_cols.json, model_meta.json
│   └── model_comparison.csv
├── backend/
│   └── app.py                # Flask API + serves the frontend
├── frontend/
│   ├── templates/index.html
│   └── static/css/style.css, static/js/script.js
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt

# 1. (optional) regenerate data & retrain
python data/generate_data.py
python model/train_model.py

# 2. start the app
cd backend
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## Results
Gradient Boosting Regressor was selected as the best model:

| Model              | R² Score | MAE (Lakhs) | RMSE (Lakhs) |
|---------------------|----------|-------------|--------------|
| Gradient Boosting    | 0.9647   | 0.579       | 0.901        |
| Random Forest        | 0.9447   | 0.740       | 1.128        |
| Decision Tree        | 0.9157   | 0.893       | 1.392        |
| Linear Regression    | 0.7071   | 1.338       | 2.595        |
| Ridge Regression     | 0.7071   | 1.338       | 2.595        |

## Note on data
This project uses a **synthetically generated dataset** (`generate_data.py`)
designed to mimic real-world used-car listings (à la Cardekho/Quikr), since
it's built for offline academic demonstration. To use it with a real
dataset, replace `data/used_cars.csv` with real data using the same column
names and re-run `train_model.py`.
