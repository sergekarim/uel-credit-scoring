# predict_api.py
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

from functions.prediction_reasons import generate_reasons
from functions.features_engineering import calculate_derived_features

from flask import Flask, request, jsonify
app = Flask(__name__)
# Load your saved models and label encoder at startup
try:
    model, label_encoder = joblib.load("results/models/credit_model.pkl")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    model, label_encoder = None, None


def generate_prediction_reasons(derived_features, predicted_grade, confidence):
    """Generate explicit reasons for the credit score prediction"""
    reasons = generate_reasons(derived_features, predicted_grade, confidence)
    return reasons


@app.route('/predict', methods=['POST'])
def predict_credit_grade():
    if not model or not label_encoder:
        return jsonify({
            'error': 'Model not loaded. Please ensure credit_model.pkl exists.'
        }), 500

    try:
        # Get JSON data from request
        client_data = request.get_json()

        if not client_data:
            return jsonify({
                'error': 'No JSON data provided'
            }), 400

        # Validate required columns
        required_cols = []
        for i in range(1, 13):
            required_cols.extend([
                f"Sales_M{i}", f"Purchases_M{i}",
                f"Decl_Sales_M{i}", f"Decl_Purchases_M{i}"
            ])

        missing_cols = [col for col in required_cols if col not in client_data]
        if missing_cols:
            return jsonify({
                'error': f'Missing required columns: {missing_cols}'
            }), 400

        # Calculate derived features
        derived_features = calculate_derived_features(client_data)

        # Combine original and derived features
        all_features = {**client_data, **derived_features}

        # Handle any infinite or NaN values
        for key, value in all_features.items():
            if np.isnan(value) or np.isinf(value):
                all_features[key] = 0

        # Convert to DataFrame for models prediction
        client_df = pd.DataFrame([all_features])

        # Ensure columns are in the same order as training features
        if hasattr(model, 'feature_names_in_'):
            feature_columns = model.feature_names_in_
            # Only use columns that exist in both the models and our dataframe
            available_features = [col for col in feature_columns if col in client_df.columns]
            client_df = client_df[available_features]

        # Make prediction
        prediction_encoded = model.predict(client_df)[0]  # Get single prediction
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]

        # Get prediction probabilities if available
        confidence_score = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(client_df)[0]  # Get single prediction probabilities
            confidence_score = float(max(proba))  # Highest probability as confidence score

        # Generate prediction reasons
        prediction_reasons = generate_prediction_reasons(derived_features, prediction, confidence_score or 0.5)

        # Format result
        result = {
            'status': 'success',
            'predicted_grade': prediction,
            'reasons': prediction_reasons,
            'derived_features': {
                'Compliance_Sales': derived_features['Compliance_Sales'],
                'Compliance_Purchases': derived_features['Compliance_Purchases'],
                'Sales_Stability': derived_features['Sales_Stability'],
                'Purchases_Stability': derived_features['Purchases_Stability'],
                'Purchase_to_Sales_Ratio': derived_features['Purchase_to_Sales_Ratio']
            }
        }

        if confidence_score:
            result['confidence'] = confidence_score

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'label_encoder_loaded': label_encoder is not None
    })


@app.route('/models-info', methods=['GET'])
def model_info():
    """Get information about the loaded models"""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    info = {
        'model_type': type(model).__name__,
        'classes': label_encoder.classes_.tolist() if label_encoder else None
    }

    if hasattr(model, 'feature_names_in_'):
        info['required_features'] = model.feature_names_in_.tolist()

    return jsonify(info)

if __name__ == '__main__':
    print("Starting Credit Scoring API...")
    print("Endpoints:")
    print("  POST /predict - Predict credit grades")
    print("  GET /health - Health check")
    print("  GET /models-info - Model information")

    app.run(debug=True, host='0.0.0.0', port=5001)