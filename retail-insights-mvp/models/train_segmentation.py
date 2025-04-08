import pandas as pd
import logging
import sys
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

# Configure logging to handle UTF-8
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger()

# Constants for file paths (adjust as needed)
SEGMENTATION_CLEAN = "data\\segmentation_cleaned.csv"  # Replace with your actual cleaned dataset path
SEGMENTATION_MODEL = "models/saved/segmentation_model.pkl"  # Where the model will be saved

def train():
    try:
        logger.info("Starting segmentation model training...")

        # Step 1: Load the dataset
        logger.info("Loading dataset...")
        df = pd.read_csv(SEGMENTATION_CLEAN)

        # Step 2: Log basic information about the dataset
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")

        # Step 3: Clean and select useful features
        logger.info("Cleaning and selecting useful features...")

        # Select relevant features (adjust based on your dataset)
        features = df[['Price (INR)', 'Quantity Sold', 'Sales (Target)', 'Customer Segment']]  # Example features

        # Step 3.1: Label encode 'Customer Segment'
        logger.info("Encoding 'Customer Segment' column...")
        le = LabelEncoder()
        features['Customer Segment'] = le.fit_transform(features['Customer Segment'])

        # Step 4: Scale the features
        logger.info("Scaling features...")
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Step 5: Train the model (using KMeans for segmentation)
        logger.info("Training KMeans model...")
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(scaled_features)

        logger.info("Cluster centers:\n" + str(kmeans.cluster_centers_))

        # Step 6: Save the trained model and scaler to the specified path
        logger.info(f"Saving the trained model and scaler to {SEGMENTATION_MODEL}...")

        # Save the model, scaler, and selected features
        with open(SEGMENTATION_MODEL, "wb") as f:
            pickle.dump({
                "model": kmeans,
                "scaler": scaler,
                "features": features.columns.tolist(),
                "label_encoder": le  # Saving the label encoder for future use
            }, f)

        logger.info("Model and scaler saved successfully!")

    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    train()
