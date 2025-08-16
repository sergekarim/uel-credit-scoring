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


file_path = "results/data/credit_data.csv"
try:
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully: {df.shape[0]} records, {df.shape[1]} columns")
except FileNotFoundError:
    print(f"File not found: {file_path}")
    df = pd.DataFrame()  # empty DataFrame as fallback

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
joblib.dump((model, label_encoder), "results/models/credit_model-new.pkl")

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
