import pandas as pd
import logging
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

# Configure logging to handle UTF-8
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger()

# Constants for file paths (adjust as needed)
SALES_PREDICTION_CLEAN = "data\\sales_prediction_dataset.csv"  # Replace with your actual cleaned dataset path
SALES_PREDICTION_MODEL = "models/saved/sales_prediction_model.pkl"  # Where the model will be saved

def train():
    try:
        logger.info("Starting sales prediction model training...")

        # Step 1: Load the dataset
        logger.info("Loading dataset...")
        df = pd.read_csv(SALES_PREDICTION_CLEAN)

        # Step 2: Log basic information about the dataset
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")

        # Step 3: Clean and select useful features for prediction
        logger.info("Cleaning and selecting useful features...")

        # Extract month from 'Date' column for additional feature
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month

        # Select relevant features (adjust based on your dataset)
        features = df[['Price (INR)', 'Quantity Sold', 'Discount (%)', 'Promotions', 
                       'Store Location', 'Day of the Week', 'Seasonality', 'Month']]  # Include 'Month' derived from Date

        # Step 3.1: Label encode categorical columns
        logger.info("Encoding categorical columns...")
        le_promotions = LabelEncoder()
        features['Promotions'] = le_promotions.fit_transform(features['Promotions'])

        le_store_location = LabelEncoder()
        features['Store Location'] = le_store_location.fit_transform(features['Store Location'])

        le_day_of_week = LabelEncoder()
        features['Day of the Week'] = le_day_of_week.fit_transform(features['Day of the Week'])

        le_seasonality = LabelEncoder()
        features['Seasonality'] = le_seasonality.fit_transform(features['Seasonality'])

        # Target variable
        target = df['Sales (Target)']

        # Step 4: Split the data into training and testing sets
        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

        # Step 5: Scale the features (optional but recommended for regression models)
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Step 6: Train the model (using RandomForestRegressor for sales prediction)
        logger.info("Training RandomForestRegressor model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Step 7: Evaluate the model
        logger.info("Evaluating model performance...")
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Mean Squared Error (MSE): {mse}")

        # Step 8: Save the trained model, scaler, and label encoders to the specified path
        logger.info(f"Saving the trained model, scaler, and label encoders to {SALES_PREDICTION_MODEL}...")
        with open(SALES_PREDICTION_MODEL, "wb") as f:
            pickle.dump({
                "model": model,
                "scaler": scaler,
                "features": features.columns.tolist(),
                "label_encoders": {
                    'Promotions': le_promotions,
                    'Store Location': le_store_location,
                    'Day of the Week': le_day_of_week,
                    'Seasonality': le_seasonality
                }
            }, f)

        logger.info("Model, scaler, and label encoders saved successfully!")

    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    train()
