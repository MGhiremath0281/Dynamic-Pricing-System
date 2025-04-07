import pandas as pd
import joblib
import os

# Define the model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/saved/popularity_model.pkl')

# Load the trained model
popularity_model = joblib.load(MODEL_PATH)

def predict_popularity(sub_category, price, has_discount, rating_score, num_reviews):
    """
    Predicts whether a product will be popular based on the input features.
    
    Parameters:
        sub_category (str): The sub-category of the product.
        price (float): The price of the product.
        has_discount (int): 0 or 1 indicating if the product has a discount.
        rating_score (float): Product rating out of 5.
        num_reviews (float): Number of reviews received.

    Returns:
        int: 0 (Not Popular) or 1 (Popular)
    """

    # Create a dataframe with the input
    input_data = pd.DataFrame([{
        'Sub Category': sub_category,
        'Price': float(price),
        'Has_Discount': int(has_discount),
        'Rating_Score': float(rating_score),
        'Num_Reviews': float(num_reviews)
    }])

    # One-hot encode or handle categorical encoding as used during training
    # Assuming model was trained using pandas.get_dummies on 'Sub Category'
    # So we align it here using the same logic
    model_features = popularity_model.feature_names_in_  # this gives us training feature names
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

    # Predict
    prediction = popularity_model.predict(input_encoded)

    return int(prediction[0])
