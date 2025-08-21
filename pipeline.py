# pipeline.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)

from functions.features_engineering import add_features
from functions.plot_confusion_matrix import plot_confusion_matrix


# -------------------------------------------------------------------
# Pipeline Steps
# -------------------------------------------------------------------

def load_validate_data(file_path):
    """Load dataset and show basic info"""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully: {df.shape[0]} records, {df.shape[1]} columns")

        # Data type validation
        df = validate_data_types(df)
        print("Data types validated successfully")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        df = pd.DataFrame()
    return df


def validate_data_types(df):
    """Validate and correct expected data types"""
    print("Validating data types...")

    # Check numerical columns (Sales and Purchases)
    numerical_cols = ([f"Sales_M{i + 1}" for i in range(12)] +
                      [f"Purchases_M{i + 1}" for i in range(12)] +
                      [f"Decl_Sales_M{i + 1}" for i in range(12)] +
                      [f"Decl_Purchases_M{i + 1}" for i in range(12)])

    for col in numerical_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"Converting {col} to numeric")
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # Check categorical columns
    categorical_cols = ['ClientID', 'Business_Size', 'CreditGrade']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('object')

    # Check for any conversion issues
    if df.isnull().sum().sum() > 0:
        print("Warning: Some values were converted to NaN during type validation")
        print("Columns with NaN values:")
        print(df.isnull().sum()[df.isnull().sum() > 0])

    return df

def explore_data(df):
    """Show dataset distribution and summary"""
    print("\nClass Distribution (CreditGrade):")
    print(df["CreditGrade"].value_counts())

    print("\nBusiness Size Distribution:")
    print(df["Business_Size"].value_counts())

    print("\nSample Data (First 5 rows):")
    display_cols = ['ClientID', 'Business_Size', 'Sales_M1', 'Purchases_M1',
                    'Sales_M2', 'Purchases_M2', 'CreditGrade']
    print(df[display_cols].head())

    print("\n--- Dataset Overview ---")
    print(df.info())
    print("\n--- Summary Statistics ---")
    print(df.describe(include="all"))


def preprocess_data(df):
    """Prepare features, labels, and split dataset"""
    # Step 1: Add engineered features first
    df = add_features(df)
    df.to_csv("results/data/preprocessed_data.csv", index=False)

    # Step 2: Split features and target
    X = df.drop(columns=["ClientID","Business_Size", "CreditGrade"])
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

    cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=10,
        cv=cv,
        scoring="f1_macro",
        random_state=42,
        verbose=2,
        n_jobs=-1,
        return_train_score=True
    )


    print("\nTraining model with RandomizedSearchCV...")
    start_time = time.time()
    random_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print("\nBest Parameters:", random_search.best_params_)

    # 1. Overfitting Detection
    print(f"\n1. OVERFITTING ANALYSIS:")
    cv_results = random_search.cv_results_
    best_idx = random_search.best_index_

    train_score = cv_results['mean_train_score'][best_idx]
    val_score = cv_results['mean_test_score'][best_idx]
    gap = train_score - val_score

    print(f"   Train F1-Score: {train_score:.4f}")
    print(f"   Validation F1-Score: {val_score:.4f}")
    print(f"   Train-Val Gap: {gap:.4f}")

    if gap > 0.1:
        print(f"WARNING: Potential overfitting detected (gap > 0.1)")
    else:
        print(f"Good generalization (gap ≤ 0.1)")

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

    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    # Plot and save confusion matrix
    plot_confusion_matrix(y_test, y_pred, label_encoder)
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Probabilities for ROC-AUC
    y_proba = model.predict_proba(X_test)
    roc_score = roc_auc_score(y_test, y_proba, multi_class="ovr")
    print("\nROC-AUC Score:", roc_score)


def save_model(model, label_encoder, path="results/models/credit_model.pkl"):
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

    # Credit Grade pie chart
    plt.figure(figsize=(6, 6))
    credit_counts = df['CreditGrade'].value_counts()
    labels = [f'{label}\n({count})' for label, count in credit_counts.items()]
    plt.pie(credit_counts, labels=labels, autopct='%1.1f%%')
    plt.title('Credit Grade Distribution')
    plt.savefig('results/plots/credit_grade_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Business Size pie chart
    plt.figure(figsize=(6, 6))
    size_counts = df['Business_Size'].value_counts()
    labels = [f'{label}\n({count})' for label, count in size_counts.items()]
    plt.pie(size_counts, labels=labels, autopct='%1.1f%%')
    plt.title('Business Size Distribution')
    plt.savefig('results/plots/business_size_distribution.png', dpi=300, bbox_inches='tight')
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
    file_path = "dataset/credit_data.csv"
    df = load_validate_data(file_path)

    if df.empty:
        raise SystemExit("Dataset not available, stopping pipeline.")

    explore_data(df)
    visualize_data(df)

    X_train, X_test, y_train, y_test, label_encoder,df = preprocess_data(df)
    model = train_model(X_train, y_train)
    save_model(model, label_encoder)
    evaluate_model(model, X_test, y_test, label_encoder)

    feat_imp_df = feature_importance(model, X_train)
    visualize_top_features(feat_imp_df)
    example_prediction(model, X_test, label_encoder)
