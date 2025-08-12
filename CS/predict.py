import pandas as pd
import joblib

# Load your saved model and label encoder
model, label_encoder = joblib.load("credit_scoring_model.pkl")

# Load new client data from CSV
new_clients_df = pd.read_csv("new_clients_data.csv")

# Optional: if you didn't include aggregate features in the CSV,
# compute them here exactly as before
new_clients_df["Compliance_Sales"] = new_clients_df[[f"Decl_Sales_M{i+1}" for i in range(12)]].sum(axis=1) / new_clients_df[[f"Sales_M{i+1}" for i in range(12)]].sum(axis=1)
new_clients_df["Compliance_Purchases"] = new_clients_df[[f"Decl_Purchases_M{i+1}" for i in range(12)]].sum(axis=1) / new_clients_df[[f"Purchases_M{i+1}" for i in range(12)]].sum(axis=1)
new_clients_df["Sales_Stability"] = new_clients_df[[f"Sales_M{i+1}" for i in range(12)]].std(axis=1) / new_clients_df[[f"Sales_M{i+1}" for i in range(12)]].mean(axis=1)
new_clients_df["Purchases_Stability"] = new_clients_df[[f"Purchases_M{i+1}" for i in range(12)]].std(axis=1) / new_clients_df[[f"Purchases_M{i+1}" for i in range(12)]].mean(axis=1)
new_clients_df["Purchase_to_Sales_Ratio"] = new_clients_df[[f"Purchases_M{i+1}" for i in range(12)]].sum(axis=1) / new_clients_df[[f"Sales_M{i+1}" for i in range(12)]].sum(axis=1)

# Make sure columns are in the same order as training features
feature_columns = model.feature_names_in_  # sklearn attribute, or you can save feature list yourself
new_clients_df = new_clients_df[feature_columns]

# Predict
predictions_encoded = model.predict(new_clients_df)
predictions = label_encoder.inverse_transform(predictions_encoded)

# Show results
for i, grade in enumerate(predictions):
    print(f"Client {i+1}: Predicted Credit Grade = {grade}")
