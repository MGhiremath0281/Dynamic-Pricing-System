import joblib
from flask import Flask, request, jsonify, render_template
import pandas as pd
import os

# Set the path for models
MODEL_PATH = os.path.join(os.getcwd(), 'models', 'saved')

# Load pre-trained models and scalers
popularity_model = joblib.load(os.path.join(MODEL_PATH, 'popularity_model.pkl'))
popularity_scaler = joblib.load(os.path.join(MODEL_PATH, 'popularity_scaler.pkl'))
recommendation_model = joblib.load(os.path.join(MODEL_PATH, 'recommendation_model.pkl'))
recommendation_scaler = joblib.load(os.path.join(MODEL_PATH, 'recomend_scaler.pkl'))  # Corrected file name
segmentation_model = joblib.load(os.path.join(MODEL_PATH, 'segmentation_model.pkl'))
segmentation_scaler = joblib.load(os.path.join(MODEL_PATH, 'segment_scaler.pkl'))

# Initialize Flask app
app = Flask(__name__, template_folder='app/templates')  # Specify the templates folder explicitly

@app.route('/')
def home():
    return render_template('home.html')

# Route for popularity prediction
@app.route('/predict_popularity', methods=['GET', 'POST'])
def predict_popularity():
    if request.method == 'POST':
        try:
            # Get form data
            data = {
                'Sub Category': int(request.form['Sub Category']),
                'Price (INR)': float(request.form['Price (INR)']),
                'Has_Discount': int(request.form['Has_Discount']),
                'Rating_score': float(request.form['Rating_score']),
                'Num_reviews': int(request.form['Num_reviews'])
            }

            # Preprocess input data
            df = pd.DataFrame([data])
            X_scaled = popularity_scaler.transform(df)

            # Make prediction
            prediction = popularity_model.predict(X_scaled)

            return render_template('popularity.html', prediction=prediction[0])

        except Exception as e:
            return render_template('popularity.html', error=str(e))

    return render_template('popularity.html')

# Route for recommendation prediction
@app.route('/predict_recommendation', methods=['GET', 'POST'])
def predict_recommendation():
    if request.method == 'POST':
        try:
            # Get form data
            data = {
                'User ID': int(request.form['User ID']),
                'Item ID': int(request.form['Item ID']),
                'Category': request.form['Category'],
                'Price (INR)': float(request.form['Price (INR)']),
                'CF Score': float(request.form['CF Score']),
                'CBF Score': float(request.form['CBF Score'])
            }

            # Preprocess input data
            df = pd.DataFrame([data])
            # One-hot encode the 'Category' column
            df = pd.get_dummies(df, columns=['Category'], drop_first=True)

            # Ensure all necessary columns are in the data
            required_columns = ['User ID', 'Item ID', 'Price (INR)', 'CF Score', 'CBF Score', 'Category_Dairy', 'Category_Fruits', 'Category_Snacks', 'Category_Vegetables']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[required_columns]

            # Scale the features
            X_scaled = recommendation_scaler.transform(df)

            # Make prediction
            prediction = recommendation_model.predict(X_scaled)

            return render_template('recommendation.html', prediction=prediction[0])

        except Exception as e:
            return render_template('recommendation.html', error=str(e))

    return render_template('recommendation.html')

# Route for segmentation prediction
@app.route('/predict_segmentation', methods=['GET', 'POST'])
def predict_segmentation():
    if request.method == 'POST':
        try:
            # Get form data
            data = {
                'Price (INR)': float(request.form['Price (INR)']),
                'Quantity Sold': int(request.form['Quantity Sold']),
                'Sales (Target)': int(request.form['Sales (Target)'])
            }

            # Preprocess input data
            df = pd.DataFrame([data])
            scaled_features = segmentation_scaler.transform(df)

            # Make prediction
            segment = segmentation_model.predict(scaled_features)

            return render_template('segmentation.html', segment=segment[0])

        except Exception as e:
            return render_template('segmentation.html', error=str(e))

    return render_template('segmentation.html')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
