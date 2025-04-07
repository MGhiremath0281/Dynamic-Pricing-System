import os

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Data and model directories
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved")
LOG_FILE = os.path.join(BASE_DIR, "logs", "app.log")

# Popularity model paths
POPULARITY_CLEAN = os.path.join(DATA_DIR, "Popularity_clean.csv")
POPULARITY_MODEL = os.path.join(MODEL_DIR, "popularity_model.pkl")

# Recommendation model paths
RECOMMENDATION_CLEAN = os.path.join(DATA_DIR, "recommendation_data.csv")
RECOMMENDATION_MODEL = os.path.join(MODEL_DIR, "recommendation_model.pkl")

# Basket model paths
BASKET_CLEAN = os.path.join(DATA_DIR, "market_basket_dataset.csv")
BASKET_MODEL = os.path.join(MODEL_DIR, "basket_model.pkl")

SEGMENTATION_CLEAN = os.path.join(DATA_DIR, "segmentation_cleaned.csv")
SEGMENTATION_MODEL = os.path.join(MODEL_DIR, "segmentation_model.pkl")
