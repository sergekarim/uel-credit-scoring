from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

from feature_engineering import calculate_derived_features

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
    reasons = []

    # Analyze Compliance_Sales
    compliance_sales = derived_features['Compliance_Sales']
    if compliance_sales >= 0.95:
        reasons.append("Excellent sales compliance (declared vs actual sales match closely)")
    elif compliance_sales >= 0.85:
        reasons.append("Good sales compliance with minor discrepancies")
    elif compliance_sales >= 0.70:
        reasons.append("Moderate sales compliance issues detected")
    else:
        reasons.append("Poor sales compliance - significant underreporting of sales")

    # Analyze Compliance_Purchases
    compliance_purchases = derived_features['Compliance_Purchases']
    if compliance_purchases >= 0.95:
        reasons.append("Excellent purchase compliance (declared vs actual purchases align well)")
    elif compliance_purchases >= 0.85:
        reasons.append("Good purchase compliance with acceptable variance")
    elif compliance_purchases >= 0.70:
        reasons.append("Moderate purchase compliance concerns")
    else:
        reasons.append("Poor purchase compliance - substantial underreporting detected")

    # Analyze Sales_Stability
    sales_stability = derived_features['Sales_Stability']
    if sales_stability <= 0.15:
        reasons.append("Highly stable sales pattern indicating consistent business performance")
    elif sales_stability <= 0.30:
        reasons.append("Moderately stable sales with acceptable variation")
    elif sales_stability <= 0.50:
        reasons.append("Somewhat volatile sales pattern")
    else:
        reasons.append("Highly volatile sales indicating business instability")

    # Analyze Purchases_Stability
    purchases_stability = derived_features['Purchases_Stability']
    if purchases_stability <= 0.20:
        reasons.append("Consistent purchasing behavior")
    elif purchases_stability <= 0.40:
        reasons.append("Moderately consistent purchasing pattern")
    else:
        reasons.append("Irregular purchasing behavior detected")

    # Analyze Purchase_to_Sales_Ratio
    purchase_ratio = derived_features['Purchase_to_Sales_Ratio']
    if 0.40 <= purchase_ratio <= 0.70:
        reasons.append("Healthy purchase-to-sales ratio indicating normal business operations")
    elif purchase_ratio < 0.40:
        reasons.append("Low purchase-to-sales ratio - may indicate high margins or service-based business")
    elif purchase_ratio > 0.80:
        reasons.append("High purchase-to-sales ratio - low margins or potential operational issues")
    else:
        reasons.append("Purchase-to-sales ratio within acceptable range")

    # Add confidence-based reasoning
    if confidence >= 0.90:
        reasons.append("High models confidence in this prediction")
    elif confidence >= 0.75:
        reasons.append("Good models confidence in this prediction")
    elif confidence >= 0.60:
        reasons.append("Moderate models confidence - some uncertainty in prediction")
    else:
        reasons.append("Low models confidence - prediction should be reviewed manually")

    # Add overall assessment based on actual credit grade patterns
    grade_upper = predicted_grade.upper()
    if grade_upper in ['AAA', 'AA', 'A']:
        reasons.insert(0, "Excellent credit profile with minimal risk")
    elif grade_upper in ['BBB', 'BB', 'B']:
        reasons.insert(0, "Good to moderate credit profile with manageable risk")
    elif grade_upper in ['CCC', 'CC', 'C']:
        reasons.insert(0, "Below investment grade with elevated credit risk")
    elif grade_upper in ['D', 'DD', 'DDD']:
        reasons.insert(0, "High credit risk profile requiring significant attention")
    else:
        # For any other grades, provide a neutral assessment
        reasons.insert(0, f"Credit grade {predicted_grade} assessed based on financial metrics")

    return reasons
    """Calculate derived features from the raw monthly data for a single client"""
    # Extract monthly data
    sales_values = [client_data[f"Sales_M{i + 1}"] for i in range(12)]
    purchases_values = [client_data[f"Purchases_M{i + 1}"] for i in range(12)]
    decl_sales_values = [client_data[f"Decl_Sales_M{i + 1}"] for i in range(12)]
    decl_purchases_values = [client_data[f"Decl_Purchases_M{i + 1}"] for i in range(12)]

    # Calculate derived features
    sales_sum = sum(sales_values)
    purchases_sum = sum(purchases_values)
    decl_sales_sum = sum(decl_sales_values)
    decl_purchases_sum = sum(decl_purchases_values)

    derived_features = {}

    # Compliance ratios (avoid division by zero)
    derived_features["Compliance_Sales"] = decl_sales_sum / sales_sum if sales_sum != 0 else 0
    derived_features["Compliance_Purchases"] = decl_purchases_sum / purchases_sum if purchases_sum != 0 else 0

    # Stability (coefficient of variation)
    sales_mean = np.mean(sales_values)
    purchases_mean = np.mean(purchases_values)

    derived_features["Sales_Stability"] = np.std(sales_values) / sales_mean if sales_mean != 0 else 0
    derived_features["Purchases_Stability"] = np.std(purchases_values) / purchases_mean if purchases_mean != 0 else 0

    # Purchase to sales ratio
    derived_features["Purchase_to_Sales_Ratio"] = purchases_sum / sales_sum if sales_sum != 0 else 0

    return derived_features


@app.route('/predict', methods=['POST'])
def predict_credit_grade():
    """
    Endpoint to predict credit grade for a single client
    Expects JSON payload with client data (no array wrapper)
    """
    if not model or not label_encoder:
        return jsonify({
            'error': 'Model not loaded. Please ensure credit_scoring_model.pkl exists.'
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

    app.run(debug=True, host='0.0.0.0', port=5000)