import pandas as pd
import joblib
import os

MODEL_PATH = os.path.join("models", "saved", "segmentation_model.pkl")
DATA_PATH = os.path.join("data", "segmentation_clean.csv")

def segment_customer(customer_id):
    try:
        df = pd.read_csv(DATA_PATH)
        model = joblib.load(MODEL_PATH)

        customer_data = df[df['CustomerID'] == customer_id]

        if customer_data.empty:
            return {"error": "Customer not found."}

        # Drop ID column for prediction
        features = customer_data.drop(columns=['CustomerID'])
        cluster = model.predict(features)[0]

        return {
            "CustomerID": customer_id,
            "Segment": f"Cluster {cluster}"
        }

    except Exception as e:
        print("Error segmenting customer:", e)
        return {"error": "Segmentation failed."}
