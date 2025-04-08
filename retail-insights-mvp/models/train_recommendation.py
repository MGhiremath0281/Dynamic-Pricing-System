import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Path to your dataset (use raw string or double backslashes)
RECOMMENDATION_CLEAN = 'data\\recommendation_dataset.csv'  # Update path as needed

# Load data function
def load_data():
    try:
        df = pd.read_csv(RECOMMENDATION_CLEAN)
        return df
    except FileNotFoundError:
        print(f"Error: The file at {RECOMMENDATION_CLEAN} was not found.")
        raise

# Train the recommendation model
def train():
    # Load the dataset
    print("🚀 Starting recommendation model training...")
    df = load_data()

    # Check the structure of the dataset (just for debugging)
    print("📊 Columns:", df.columns)
    print("🔢 Dataset shape:", df.shape)

    # Clean the data if necessary (example: drop NA, encode categorical variables)
    print("🧼 Cleaning data...")
    df = df.dropna()  # Example: remove rows with missing values

    # Features and target variable (adjusted based on the dataset)
    X = df.drop(columns=['Final Score'])  # Use 'Final Score' as the target
    y = df['Final Score']  # 'Final Score' is the target variable for prediction

    # Preprocessing pipeline for categorical variables
    # OneHotEncoder will be used for the 'Category' column (you can add more columns if necessary)
    categorical_features = ['Category']
    numeric_features = [col for col in X.columns if col not in categorical_features]

    # Column transformer: applies OneHotEncoder to 'Category' and leaves other columns unchanged
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features),
            ('num', 'passthrough', numeric_features)
        ])

    # Create a pipeline that applies preprocessing and then fits the model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))  # Changed to Regressor
    ])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    print("🛠️ Training the model...")
    model.fit(X_train, y_train)

    # Evaluate the model
    print("📈 Model evaluation:")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error for regression problems
    print(f"Mean Squared Error (MSE): {mse}")

    # Save the model to a .pkl file
    model_path = 'models/saved/recommendation_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to: {model_path}")

# Start training
if __name__ == "__main__":
    train()
