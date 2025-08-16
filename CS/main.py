# credit_scoring_notebook.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from feature_engineering import add_features



# --- Step 1: Data Generation ---

def generate_client_data(client_id, grade_target):
    import numpy as np
    np.random.seed(client_id)  # reproducible per client

    # Generate data based on initial target grade
    if grade_target == "A":
        base_sales = np.random.uniform(4000, 6000)
        sales = base_sales + np.random.normal(0, base_sales * 0.05, 12)
        purchases = np.random.uniform(3000, 5000, 12)
        declared_sales = sales * np.random.uniform(0.96, 1.0, 12)
        declared_purchases = purchases * np.random.uniform(0.85, 1.0, 12)
    elif grade_target == "B":
        base_sales = np.random.uniform(3000, 7000)
        sales = base_sales + np.random.normal(0, base_sales * 0.15, 12)
        purchases = np.random.uniform(2000, 6000, 12)
        declared_sales = sales * np.random.uniform(0.91, 0.95, 12)
        declared_purchases = purchases * np.random.uniform(0.85, 1.0, 12)
    elif grade_target == "C":
        base_sales = np.random.uniform(2000, 8000)
        sales = base_sales + np.random.normal(0, base_sales * 0.3, 12)
        purchases = np.random.uniform(1500, 7000, 12)
        declared_sales = sales * np.random.uniform(0.86, 0.9, 12)
        declared_purchases = purchases * np.random.uniform(0.85, 1.0, 12)
    else:  # D
        sales = np.random.uniform(1000, 10000, 12)
        purchases = np.random.uniform(800, 9000, 12)
        declared_sales = sales * np.random.uniform(0.7, 0.85, 12)
        declared_purchases = purchases * np.random.uniform(0.7, 0.85, 12)

    # Calculate metrics
    sales_volatility = np.std(sales) / np.mean(sales)
    declaration_ratio = declared_sales.sum() / sales.sum()

    # Count problematic months
    zero_sales_months = np.sum(sales == 0)
    zero_declared_months = np.sum(declared_sales == 0)
    high_mismatch_months = np.sum(np.abs(declared_sales - sales) > (sales * 0.2))  # >20% difference

    # Determine grade
    if (sales_volatility < 0.15 and declaration_ratio > 0.95 and
        zero_sales_months < 2 and zero_declared_months < 2 and high_mismatch_months < 2):
        grade = "A"
    elif (sales_volatility < 0.25 and declaration_ratio > 0.9 and
          zero_sales_months < 2 and zero_declared_months < 2 and high_mismatch_months < 2):
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

# Generate data 500 record per category
grades = ["A", "B", "C", "D"]
records_per_grade = 500  # 2000 total = 500 per grade
data = []

for i, grade in enumerate(grades):
    for j in range(records_per_grade):
        client_id = i * records_per_grade + j + 1
        data.append(generate_client_data(client_id, grade))

df = pd.DataFrame(data)

# add aggregate features:
df = add_features(df)

# Save to CSV (optional)
df.to_csv("results/data/credit_data.csv", index=False)

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

# Train/test split (already done in your code)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Parameter grid for tuning Random Forest
param_dist = {
    "n_estimators": [500, 1000, 1500, 2000],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced"]
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# RandomizedSearch (faster than GridSearch)
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=10,               # test only 30 random combos instead of 216
    cv=cv,
    scoring="f1_macro",      # optimize across all grades equally
    random_state=42,
    verbose=2,
    n_jobs=-1
)

# Train
random_search.fit(X_train, y_train)

# Best model
model = random_search.best_estimator_
print("Best Parameters:", random_search.best_params_)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation metrics
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(
    y_test, y_pred,
    labels=label_encoder.transform(label_encoder.classes_),
    target_names=label_encoder.classes_
))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Confidence scores (probabilities)
y_proba = model.predict_proba(X_test)
roc_score = roc_auc_score(y_test, y_proba, multi_class="ovr")
print("\nROC-AUC Score:", roc_score)

# Save model + encoder
joblib.dump((model, label_encoder), "results/models/credit_model.pkl")

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
plt.savefig(f"results/plots/top_features.png", dpi=300, bbox_inches='tight')
plt.show()

# --- Step 4: Credit Grade Distribution ---


plt.figure(figsize=(6, 4))
sns.countplot(x="CreditGrade", data=df, order=sorted(df["CreditGrade"].unique()))
plt.title("Distribution of Credit Grades")
plt.savefig(f"results/plots/distribution_of_credit_grades.png", dpi=300, bbox_inches='tight')
plt.show()

# --- Step 5: Pairplot of Top Features ---
top_features = feat_imp_df.head(5)["Feature"].tolist() + ["CreditGrade"]
sns.pairplot(df[top_features], hue="CreditGrade", palette="Set2")
plt.suptitle("Pairplot of Top Features by Credit Grade", y=1.02)
plt.savefig(f"results/plots/pairplot_of_top_features.png", dpi=300, bbox_inches='tight')
plt.show()

# --- Step 6: Example Prediction for New Client ---
new_client = X_test.iloc[0:1]
pred_grade = label_encoder.inverse_transform(model.predict(new_client))[0]
print("Predicted Grade for sample client:", pred_grade)
