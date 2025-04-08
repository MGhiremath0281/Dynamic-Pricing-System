import sys
import io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import logging

# Set default encoding to utf-8 for stdout (to avoid UnicodeEncodeError)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Initialize logger
logger = logging.getLogger('train_popularity')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# File paths
DATA_PATH = "data/Popularity_clean.csv"  # Update with actual path
POPULARITY_MODEL = "models/saved/popularity_model.pkl"

def train():
    logger.info("🚀 Starting training...")

    # Loading dataset
    logger.info("📥 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Display dataset info
    logger.info(f"📊 Columns: {df.columns.tolist()}")
    logger.info(f"🔢 Dataset shape: {df.shape}")

    # Clean data (if necessary)
    logger.info("🧼 Cleaning data...")

    # Label encoding for 'Customer Segment' column if exists
    if 'Customer Segment' in df.columns:
        logger.info("🎯 Encoding 'Customer Segment'...")
        le = LabelEncoder()
        df['Customer Segment'] = le.fit_transform(df['Customer Segment'])
    
    # Feature selection (adjust based on your dataset)
    X = df[['Price (INR)', 'Has_Discount', 'Rating_score', 'Num_reviews']]  # Example features
    y = df['Popularity']  # Assuming 'Popularity' is the target column

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model initialization and training
    logger.info("🛠️ Training the model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Model evaluation
    logger.info("📈 Model evaluation:")
    eval_report = classification_report(y_test, y_pred)
    logger.info(eval_report)

    # Save model
    logger.info(f"✅ Model saved to: {POPULARITY_MODEL}")
    with open(POPULARITY_MODEL, 'wb') as f:
        pickle.dump(model, f)

    # Save evaluation report to file
    with open("model_evaluation.txt", 'w') as f:
        f.write(eval_report)

if __name__ == "__main__":
    train()
