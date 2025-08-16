import pandas as pd

def generate_reasons(derived_features, predicted_grade, confidence):
    reasons = []


    # # Add confidence-based reasoning
    if confidence >= 0.90:
        reasons.append("High models confidence in this prediction")
    elif confidence >= 0.75:
        reasons.append("Good models confidence in this prediction")
    elif confidence >= 0.60:
        reasons.append("Moderate models confidence - some uncertainty in prediction")
    else:
        reasons.append("Low models confidence - prediction should be reviewed manually")

    # Compliance
    if derived_features['Compliance_Sales'] >= 0.95:
        reasons.append("Excellent sales compliance (declared vs actual match closely)")
    elif derived_features['Compliance_Sales'] >= 0.85:
        reasons.append("Good sales compliance with minor discrepancies")
    else:
        reasons.append("Moderate/poor sales compliance detected")

    if derived_features['Compliance_Purchases'] >= 0.95:
        reasons.append("Excellent purchase compliance")
    elif derived_features['Compliance_Purchases'] >= 0.85:
        reasons.append("Good purchase compliance with minor variance")
    else:
        reasons.append("Moderate/poor purchase compliance detected")

        # --- Stability ---
    if derived_features['Sales_Stability'] <= 0.15:
        reasons.append("Highly stable sales pattern")
    elif derived_features['Sales_Stability'] <= 0.3:
        reasons.append("Moderately stable sales pattern")
    else:
        reasons.append("Volatile sales pattern")

    if derived_features['Purchases_Stability'] <= 0.2:
        reasons.append("Consistent purchasing behavior")
    elif derived_features['Purchases_Stability'] <= 0.4:
        reasons.append("Moderately consistent purchasing pattern")
    else:
        reasons.append("Irregular purchasing behavior detected")

    # --- Purchase-to-Sales Ratio ---
    ratio = derived_features['Purchase_to_Sales_Ratio']
    if 0.4 <= ratio <= 0.7:
        reasons.append("Healthy purchase-to-sales ratio")
    elif ratio < 0.4:
        reasons.append("Low purchase-to-sales ratio - may indicate high margins")
    else:
        reasons.append("High purchase-to-sales ratio - potential operational issues")

    # Growth
    if derived_features['Sales_Growth'] > 1.2:
        reasons.append("Sales show strong positive growth")
    elif derived_features['Sales_Growth'] < 0.8:
        reasons.append("Sales are declining")

    if derived_features['Purchases_Growth'] > 1.2:
        reasons.append("Purchases show strong growth")
    elif derived_features['Purchases_Growth'] < 0.8:
        reasons.append("Purchases are declining")

        # --- Seasonality & Peak ---
    if derived_features['Sales_Seasonality'] > 0.4:
        reasons.append("High sales variability / seasonality detected")


    # --- Profitability ---
    if derived_features['Profitability'] > 0.3:
        reasons.append("Healthy profitability proxy")
    else:
        reasons.append("Low profitability - review cost structure")

    # --- Outlier Detection ---
    if derived_features['Sales_Max_to_Mean'] > 2.0:
        reasons.append("Sales have large spikes compared to average")

    # --- Interaction & Rolling Features ---
    if derived_features['Compliance_Adjusted'] < 0.7:
        reasons.append("Adjusted compliance indicates some risk")
    if derived_features['Q4_to_Annual_Sales'] > 0.4:
        reasons.append("High concentration of sales in Q4")


    # Keep original grade and confidence reasoning
    grade_upper = predicted_grade.upper()
    if grade_upper in ['AAA', 'AA', 'A']:
        reasons.insert(0, "Excellent credit profile with minimal risk")
    elif grade_upper in ['BBB', 'BB', 'B']:
        reasons.insert(0, "Good to moderate credit profile with manageable risk")
    elif grade_upper in ['CCC', 'CC', 'C']:
        reasons.insert(0, "Below investment grade with elevated credit risk")
    else:
        reasons.insert(0, f"High credit risk profile ({predicted_grade})")

    return reasons
