import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.utils.helpers import get_logger
from config import SEGMENTATION_CLEAN, SEGMENTATION_MODEL

import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = get_logger("train_segmentation")

def train():
    logger.info("📥 Loading dataset...")
    df = pd.read_csv(SEGMENTATION_CLEAN)

    logger.info(f"🔢 Dataset shape: {df.shape}")
    logger.info(f"📊 Columns: {df.columns.tolist()}")

    logger.info("🧼 Cleaning and selecting useful features...")

    # ✅ Convert 'Time' to hour for behavior analysis
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')  # Avoids warning
    df['Hour'] = df['Time'].dt.hour

    # ✅ Selecting relevant features
    features = df[['Unit price', 'Quantity', 'Total', 'gross income', 'Rating', 'Hour']]

    logger.info("⚖️ Scaling features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    logger.info("🔍 Training KMeans model...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(scaled_features)

    logger.info("📌 Cluster centers:\n" + str(kmeans.cluster_centers_))

    # ✅ Save model and scaler
    with open(SEGMENTATION_MODEL, "wb") as f:
        pickle.dump({
            "model": kmeans,
            "scaler": scaler,
            "features": features.columns.tolist()
        }, f)

    logger.info(f"✅ Segmentation model saved to: {SEGMENTATION_MODEL}")

if __name__ == "__main__":
    logger.info("🚀 Starting segmentation model training...")
    train()
