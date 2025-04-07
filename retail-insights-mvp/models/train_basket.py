# models/train_basket.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.utils.helpers import get_logger
from config import BASKET_CLEAN, BASKET_MODEL

import pandas as pd
import pickle
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

logger = get_logger("train_basket")

def train():
    logger.info("📥 Loading dataset...")
    df = pd.read_csv(BASKET_CLEAN)

    logger.info(f"📊 Columns: {df.columns.tolist()}")
    logger.info(f"🔢 Dataset shape: {df.shape}")

    logger.info("🧺 Building transaction list...")
    transactions = df.groupby("TransactionID")['item'].apply(list).tolist()

    logger.info("📐 Converting transactions to one-hot encoded format...")
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    logger.info("🔍 Running Apriori algorithm...")
    frequent_itemsets = apriori(df_encoded, min_support=0.02, use_colnames=True)

    logger.info("📏 Generating association rules...")
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

    logger.info(f"✅ Found {len(frequent_itemsets)} frequent itemsets and {len(rules)} rules")

    model = {
        "frequent_itemsets": frequent_itemsets,
        "rules": rules,
        "encoder": te
    }

    # ✅ Save model
    with open(BASKET_MODEL, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"✅ Basket analysis model saved to: {BASKET_MODEL}")

if __name__ == "__main__":
    logger.info("🚀 Starting basket model training...")
    train()
