import pandas as pd
import joblib
import os

MODEL_PATH = os.path.join("models", "saved", "recommendation_model.pkl")
DATA_PATH = os.path.join("data", "recommendation_clean.csv")

def get_recommendations(product_line):
    try:
        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)

        if product_line not in df['Product line'].unique():
            return []

        recommended = df[df['Product line'] == product_line].sort_values(by='Rating', ascending=False).head(5)
        return recommended[['Product line', 'Rating', 'Payment']].to_dict(orient='records')
    except Exception as e:
        print("Error:", e)
        return []
