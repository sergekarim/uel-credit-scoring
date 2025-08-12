# credit_scoring_notebook.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib


# --- Step 1: Data Generation ---

def generate_client_data(client_id, grade_target):
    np.random.seed(client_id)  # reproducible per client

    # We'll generate sales, purchases, declared sales/purchases
    # to satisfy the specific grade condition.

    if grade_target == "A":
        # Very low volatility, very high declaration ratio
        base_sales = np.random.uniform(4000, 6000)
        sales = base_sales + np.random.normal(0, base_sales * 0.05, 12)  # low volatility
        purchases = np.random.uniform(3000, 5000, 12)
        declared_sales = sales * np.random.uniform(0.96, 1.0, 12)  # high declaration ratio >0.95
        declared_purchases = purchases * np.random.uniform(0.85, 1.0, 12)
    elif grade_target == "B":
        # Low-medium volatility, high declaration ratio (>0.9)
        base_sales = np.random.uniform(3000, 7000)
        sales = base_sales + np.random.normal(0, base_sales * 0.15, 12)  # moderate volatility <0.25
        purchases = np.random.uniform(2000, 6000, 12)
        declared_sales = sales * np.random.uniform(0.91, 0.95, 12)  # declaration ratio >0.9
        declared_purchases = purchases * np.random.uniform(0.85, 1.0, 12)
    elif grade_target == "C":
        # Medium volatility, medium declaration ratio (>0.85)
        base_sales = np.random.uniform(2000, 8000)
        sales = base_sales + np.random.normal(0, base_sales * 0.3, 12)  # volatility <0.4
        purchases = np.random.uniform(1500, 7000, 12)
        declared_sales = sales * np.random.uniform(0.86, 0.9, 12)  # declaration ratio >0.85
        declared_purchases = purchases * np.random.uniform(0.85, 1.0, 12)
    else:  # grade_target == "D"
        # High volatility OR low declaration ratio
        sales = np.random.uniform(1000, 10000, 12)
        purchases = np.random.uniform(800, 9000, 12)
        declared_sales = sales * np.random.uniform(0.7, 0.85, 12)  # declaration ratio <= 0.85
        declared_purchases = purchases * np.random.uniform(0.7, 0.85, 12)

    # Calculate actual metrics (should satisfy the conditions approximately)
    sales_volatility = np.std(sales) / np.mean(sales)
    declaration_ratio = declared_sales.sum() / sales.sum()

    # Double check grade to be consistent (optional)
    if sales_volatility < 0.15 and declaration_ratio > 0.95:
        grade = "A"
    elif sales_volatility < 0.25 and declaration_ratio > 0.9:
        grade = "B"
    elif sales_volatility < 0.4 and declaration_ratio > 0.85:
        grade = "C"
    else:
        grade = "D"

    return {
        "ClientID": f"C{client_id:03d}",
        **{f"Sales_M{i + 1}": sales[i] for i in range(12)},
        **{f"Purchases_M{i + 1}": purchases[i] for i in range(12)},
        **{f"Decl_Sales_M{i + 1}": declared_sales[i] for i in range(12)},
        **{f"Decl_Purchases_M{i + 1}": declared_purchases[i] for i in range(12)},
        "CreditGrade": grade
    }


# Generate dataset 500 record per category
grades = ["A", "B", "C", "D"]
records_per_grade = 500  # 2000 total = 500 per grade
data = []

for i, grade in enumerate(grades):
    for j in range(records_per_grade):
        client_id = i * records_per_grade + j + 1
        data.append(generate_client_data(client_id, grade))

df = pd.DataFrame(data)

# Save to CSV (optional)
df.to_csv("credit_data.csv", index=False)

print(df["CreditGrade"].value_counts())
print("Sample data:")
print(df.head())

# --- Step 2: Machine Learning ---

# Prepare features and target
X = df.drop(columns=["ClientID", "CreditGrade"])
y = df["CreditGrade"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train Random Forest classifier
model = RandomForestClassifier(n_estimators=2000, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation metrics
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(
    y_test, y_pred,
    labels=label_encoder.transform(label_encoder.classes_),
    target_names=label_encoder.classes_))
# print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model and label encoder for future use
joblib.dump((model, label_encoder), "credit_scoring_model.pkl")

# --- Step 3: Feature Importance ---

importances = model.feature_importances_
features = X.columns
feat_imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance",
                                                                                         ascending=False)

print("\nTop 10 Important Features:")
print(feat_imp_df.head(10))

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x="Importance", y="Feature", data=feat_imp_df.head(15))
plt.title("Top 15 Feature Importances in Credit Scoring Model")
plt.tight_layout()
plt.show()

# --- Step 4: Credit Grade Distribution ---

plt.figure(figsize=(6, 4))
sns.countplot(x="CreditGrade", data=df, order=sorted(df["CreditGrade"].unique()))
plt.title("Distribution of Credit Grades")
plt.show()

# --- Step 5: Pairplot of Top Features ---

top_features = feat_imp_df.head(5)["Feature"].tolist() + ["CreditGrade"]
sns.pairplot(df[top_features], hue="CreditGrade", palette="Set2")
plt.suptitle("Pairplot of Top Features by Credit Grade", y=1.02)
plt.show()

# --- Step 6: Example Prediction for New Client ---

new_client = X_test.iloc[0:1]
pred_grade = label_encoder.inverse_transform(model.predict(new_client))[0]
print("Predicted Grade for sample client:", pred_grade)
