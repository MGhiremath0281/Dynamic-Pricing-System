# models/train_popularity.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.utils.helpers import get_logger
from config import POPULARITY_CLEAN, POPULARITY_MODEL

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

logger = get_logger("train_popularity")

def train():
    logger.info("📥 Loading dataset...")
    df = pd.read_csv(POPULARITY_CLEAN)

    logger.info(f"📊 Columns: {df.columns.tolist()}")
    logger.info(f"🔢 Dataset shape: {df.shape}")

    # ✅ Encode 'Sub Category' (categorical)
    le = LabelEncoder()
    df['Sub Category'] = le.fit_transform(df['Sub Category'])

    # ✅ Features and label
    feature_cols = ['Sub Category', 'Price', 'Has_Discount', 'Rating_Score', 'Num_Reviews']
    X = df[feature_cols]
    y = df['Popular']

    # ✅ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ✅ Evaluate
    y_pred = model.predict(X_test)
    logger.info("📈 Model evaluation:\n" + classification_report(y_test, y_pred))

    # ✅ Save model
    joblib.dump(model, POPULARITY_MODEL)
    logger.info(f"✅ Model saved to: {POPULARITY_MODEL}")

if __name__ == "__main__":
    logger.info("🚀 Starting training...")
    train()
