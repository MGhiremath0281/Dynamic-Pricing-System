# models/train_recommendation.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.utils.helpers import get_logger
from config import RECOMMENDATION_CLEAN, RECOMMENDATION_MODEL

import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = get_logger("train_recommendation")

def get_top_n_recommendations(similarity_df, item, n=5):
    if item not in similarity_df.columns:
        return []
    similar_items = similarity_df[item].sort_values(ascending=False)
    return similar_items.iloc[1:n+1].index.tolist()

def evaluate_model(basket, similarity_df, top_n=5):
    logger.info(f"🧪 Evaluating Top-{top_n} recommendations on a subset of users...")

    hit_count = 0
    total = 0
    test_users = np.random.choice(basket.index, size=min(50, len(basket)), replace=False)

    for user in test_users:
        user_items = basket.loc[user]
        bought_items = user_items[user_items > 0].index.tolist()

        if len(bought_items) < 2:
            continue  # Not enough items to evaluate

        test_item = np.random.choice(bought_items)
        bought_items.remove(test_item)

        recommended = []
        for item in bought_items:
            recommended.extend(get_top_n_recommendations(similarity_df, item, n=top_n))

        recommended = set(recommended)
        total += 1
        if test_item in recommended:
            hit_count += 1

    hit_rate = hit_count / total if total else 0
    logger.info(f"🎯 Hit Rate@{top_n}: {hit_rate:.2f}")

def train():
    logger.info("📥 Loading dataset...")
    df = pd.read_csv(RECOMMENDATION_CLEAN)

    logger.info(f"📊 Columns: {df.columns.tolist()}")
    logger.info(f"🔢 Dataset shape: {df.shape}")

    logger.info("🔧 Creating user-item interaction matrix...")
    basket = pd.crosstab(df['Member_number'], df['itemDescription'])

    logger.info("📐 Computing item-item similarity...")
    similarity_matrix = cosine_similarity(basket.T)
    similarity_df = pd.DataFrame(similarity_matrix, index=basket.columns, columns=basket.columns)

    logger.info("📊 Evaluating model...")
    evaluate_model(basket, similarity_df, top_n=5)

    model = {
        "similarity_matrix": similarity_df,
        "basket": basket
    }

    # ✅ Save model
    with open(RECOMMENDATION_MODEL, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"✅ Recommendation model saved to: {RECOMMENDATION_MODEL}")

if __name__ == "__main__":
    logger.info("🚀 Starting recommendation model training...")
    train()
