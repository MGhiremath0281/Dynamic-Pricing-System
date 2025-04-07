import pandas as pd
import joblib
import os

MODEL_PATH = os.path.join("models", "saved", "popularity_model.pkl")
DATA_PATH = os.path.join("data", "popularity_clean.csv")

def get_popular_products(n=10):
    try:
        df = pd.read_csv(DATA_PATH)
        model = joblib.load(MODEL_PATH)
        popular_products = model.head(n)  # or however your model output is structured
        return popular_products.to_dict(orient="records")
    except Exception as e:
        print("Error fetching popular products:", e)
        return []
