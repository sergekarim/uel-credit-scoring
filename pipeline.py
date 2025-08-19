# pipeline.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)

from functions.features_engineering import add_features


# -------------------------------------------------------------------
# Pipeline Steps
# -------------------------------------------------------------------

def load_data(file_path):
    """Load dataset and show basic info"""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully: {df.shape[0]} records, {df.shape[1]} columns")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        df = pd.DataFrame()
    return df


def explore_data(df):
    """Show dataset distribution and summary"""
    print("\nClass Distribution (CreditGrade):")
    print(df["CreditGrade"].value_counts())
    print("\nSample Data:")
    print(df.head())
    print("\n--- Dataset Overview ---")
    print(df.info())
    print("\n--- Summary Statistics ---")
    print(df.describe(include="all"))


def preprocess_data(df):
    """Prepare features, labels, and split dataset"""

    # Step 1: Add engineered features first
    df = add_features(df)
    df.to_csv("results/data/preprocessed_data-new.csv", index=False)

    # Step 2: Split features and target
    X = df.drop(columns=["ClientID", "CreditGrade"])
    y = df["CreditGrade"]

    # Step 3: Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Step 4: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    return X_train, X_test, y_train, y_test, label_encoder,df


def train_model(X_train, y_train):
    """Train Random Forest with RandomizedSearchCV"""
    param_dist = {
        "n_estimators": [500, 1000, 1500, 2000],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=10,
        cv=cv,
        scoring="f1_macro",
        random_state=42,
        verbose=2,
        n_jobs=-1
    )

    print("\nTraining model with RandomizedSearchCV...")
    random_search.fit(X_train, y_train)
    print("\nBest Parameters:", random_search.best_params_)

    return random_search.best_estimator_


def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate model with accuracy, reports, confusion, ROC"""
    y_pred = model.predict(X_test)

    print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(
        y_test, y_pred,
        labels=label_encoder.transform(label_encoder.classes_),
        target_names=label_encoder.classes_
    ))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Probabilities for ROC-AUC
    y_proba = model.predict_proba(X_test)
    roc_score = roc_auc_score(y_test, y_proba, multi_class="ovr")
    print("\nROC-AUC Score:", roc_score)


def save_model(model, label_encoder, path="results/models/credit_model-new.pkl"):
    """Save trained model + encoder"""
    joblib.dump((model, label_encoder), path)
    print(f"\nModel saved successfully at {path}")


def feature_importance(model, X, top_n=10):
    """Plot and return feature importance"""
    importances = model.feature_importances_
    features = X.columns

    # Safety check
    if len(features) != len(importances):
        raise ValueError(
            f"Feature length ({len(features)}) and importance length ({len(importances)}) do not match"
        )

    feat_imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\nTop 10 Important Features:")
    print(feat_imp_df.head(10))

    # Plot top features
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Importance", y="Feature", data=feat_imp_df.head(top_n))
    plt.title(f"Top {top_n} Feature Importances in Credit Scoring Model")
    plt.tight_layout()
    plt.savefig("results/plots/top_features.png", dpi=300, bbox_inches="tight")
    plt.show()

    return feat_imp_df


def visualize_data(df):
    """Data distribution + pairplot visualization"""
    plt.figure(figsize=(6, 4))
    sns.countplot(x="CreditGrade", data=df, order=sorted(df["CreditGrade"].unique()))
    plt.title("Distribution of Credit Grades")
    plt.savefig("results/plots/distribution_of_credit_grades.png", dpi=300, bbox_inches="tight")
    plt.show()

def visualize_top_features(feat_imp_df):
    top_features = feat_imp_df.head(5)["Feature"].tolist() + ["CreditGrade"]
    sns.pairplot(df[top_features], hue="CreditGrade", palette="Set2")
    plt.suptitle("Pairplot of Top Features by Credit Grade", y=1.02)
    plt.savefig("results/plots/pairplot_of_top_features.png", dpi=300, bbox_inches="tight")
    plt.show()


def example_prediction(model, X_test, label_encoder):
    """Make a single prediction"""
    new_client = X_test.iloc[0:1]
    pred_grade = label_encoder.inverse_transform(model.predict(new_client))[0]
    print("\nPredicted Grade for sample client:", pred_grade)


# -------------------------------------------------------------------
# Run the pipeline
# -------------------------------------------------------------------
if __name__ == "__main__":
    file_path = "dataset/credit_data-new.csv"
    df = load_data(file_path)

    if df.empty:
        raise SystemExit("Dataset not available, stopping pipeline.")

    explore_data(df)
    visualize_data(df)

    X_train, X_test, y_train, y_test, label_encoder,df = preprocess_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, label_encoder)
    save_model(model, label_encoder)

    feat_imp_df = feature_importance(model, X_train)
    visualize_top_features(feat_imp_df)
    example_prediction(model, X_test, label_encoder)
