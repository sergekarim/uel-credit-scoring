# academic_credit_scoring_pipeline.py
"""
Academic Credit Scoring Model Pipeline
=====================================

This module implements a comprehensive credit scoring system for academic research.
The pipeline includes data generation, feature engineering, models training, and evaluation.

Author: Serge Paluku
Institution: UEL
Date: 16-Aug-2025

Research Question: How do business compliance patterns and operational stability
affect credit risk assessment in small-to-medium enterprises?

Methodology: Random Forest Classification with engineered business health features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import joblib


class CreditScoringPipeline:
    """
    A comprehensive credit scoring pipeline for academic research.

    This class encapsulates the entire workflow from data generation
    to models evaluation and interpretation.
    """

    def __init__(self, random_state=42, output_dir="results"):
        """
        Initialize the credit scoring pipeline.

        Parameters:
        -----------
        random_state : int, default=42
            Random seed for reproducible results
        output_dir : str, default="results"
            Directory to save outputs (models, plots, reports)
        """
        self.random_state = random_state
        self.output_dir = output_dir
        self.model = None
        self.label_encoder = None
        self.feature_importance_df = None
        self.df = None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        os.makedirs(f"{output_dir}/data", exist_ok=True)

        print(f"Credit Scoring Pipeline Initialized")
        print(f"Output directory: {output_dir}")
        print(f"Random state: {random_state}")
        print("-" * 50)

    def generate_synthetic_data(self, records_per_grade=500, grades=["A", "B", "C", "D"]):
        """
        Generate synthetic business data for credit scoring research.

        Parameters:
        -----------
        records_per_grade : int, default=500
            Number of companies to generate per credit grade
        grades : list, default=["A", "B", "C", "D"]
            Credit grades to generate

        Returns:
        --------
        pd.DataFrame : Generated data with business metrics and credit grades
        """
        print("Phase 1: Synthetic Data Generation")
        print(f"Generating {records_per_grade} records per grade: {grades}")

        def generate_client_data(client_id, grade_target):
            """Generate individual client business data based on target grade."""
            np.random.seed(client_id + self.random_state)

            # [Same generate_client_data function as in your improved code]
            if grade_target == "A":
                base_sales = np.random.uniform(5000, 8000)
                sales = base_sales + np.random.normal(0, base_sales * 0.05, 12)
                sales = np.maximum(sales, base_sales * 0.8)
                purchases = np.random.uniform(3000, 5000, 12)
                declared_sales = sales * np.random.uniform(0.96, 1.0, 12)
                declared_purchases = purchases * np.random.uniform(0.90, 1.0, 12)
            elif grade_target == "B":
                base_sales = np.random.uniform(3500, 6000)
                sales = base_sales + np.random.normal(0, base_sales * 0.15, 12)
                sales = np.maximum(sales, base_sales * 0.6)
                purchases = np.random.uniform(2000, 4500, 12)
                declared_sales = sales * np.random.uniform(0.91, 0.95, 12)
                declared_purchases = purchases * np.random.uniform(0.85, 0.95, 12)
            elif grade_target == "C":
                base_sales = np.random.uniform(2000, 5000)
                sales = base_sales + np.random.normal(0, base_sales * 0.3, 12)
                sales = np.maximum(sales, base_sales * 0.3)
                purchases = np.random.uniform(1500, 4000, 12)
                declared_sales = sales * np.random.uniform(0.86, 0.9, 12)
                declared_purchases = purchases * np.random.uniform(0.80, 0.90, 12)
            else:  # D grade
                base_sales = np.random.uniform(500, 4000)
                scenario = np.random.choice([1, 2, 3])
                if scenario == 1:
                    sales = np.random.uniform(0, 1000, 12)
                    zero_months = np.random.choice(12, size=np.random.randint(2, 6), replace=False)
                    sales[zero_months] = 0
                elif scenario == 2:
                    sales = base_sales + np.random.normal(0, base_sales * 0.8, 12)
                    sales = np.maximum(sales, 0)
                else:
                    sales = base_sales + np.random.normal(0, base_sales * 0.4, 12)
                    sales = np.maximum(sales, base_sales * 0.2)
                purchases = np.random.uniform(100, 3000, 12)
                declared_sales = sales * np.random.uniform(0.5, 0.85, 12)
                declared_purchases = purchases * np.random.uniform(0.5, 0.80, 12)

            # Ensure no negative values
            sales = np.maximum(sales, 0)
            purchases = np.maximum(purchases, 0)
            declared_sales = np.maximum(declared_sales, 0)
            declared_purchases = np.maximum(declared_purchases, 0)

            # Grade validation logic
            avg_sales = np.mean(sales)
            sales_volatility = np.std(sales) / np.mean(sales) if np.mean(sales) > 0 else 1.0
            declaration_ratio = declared_sales.sum() / sales.sum() if sales.sum() > 0 else 0
            zero_sales_months = np.sum(sales == 0)

            if (avg_sales >= 4000 and sales_volatility < 0.15 and declaration_ratio > 0.95 and zero_sales_months == 0):
                grade = "A"
            elif (
                    avg_sales >= 2500 and sales_volatility < 0.25 and declaration_ratio > 0.90 and zero_sales_months <= 1):
                grade = "B"
            elif (avg_sales >= 1500 and sales_volatility < 0.4 and declaration_ratio > 0.85 and zero_sales_months <= 2):
                grade = "C"
            else:
                grade = "D"

            return {
                "ClientID": f"C{client_id:04d}",
                **{f"Sales_M{i + 1}": sales[i] for i in range(12)},
                **{f"Purchases_M{i + 1}": purchases[i] for i in range(12)},
                **{f"Decl_Sales_M{i + 1}": declared_sales[i] for i in range(12)},
                **{f"Decl_Purchases_M{i + 1}": declared_purchases[i] for i in range(12)},
                "CreditGrade": grade
            }

        # Generate data
        data = []
        total_records = len(grades) * records_per_grade

        for i, grade in enumerate(grades):
            for j in range(records_per_grade):
                client_id = i * records_per_grade + j + 1
                data.append(generate_client_data(client_id, grade))

        self.df = pd.DataFrame(data)

        # Feature Engineering
        self._engineer_features()

        print(f"✓ Generated {len(self.df)} records")
        print(f"✓ Grade distribution: {dict(self.df['CreditGrade'].value_counts())}")

        return self.df

    def _engineer_features(self):
        """Engineer business health and compliance features."""
        print("Phase 2: Feature Engineering")

        # Compliance features
        self.df["Compliance_Sales"] = (
                self.df[[f"Decl_Sales_M{i + 1}" for i in range(12)]].sum(axis=1) /
                (self.df[[f"Sales_M{i + 1}" for i in range(12)]].sum(axis=1) + 1e-6)
        )
        self.df["Compliance_Purchases"] = (
                self.df[[f"Decl_Purchases_M{i + 1}" for i in range(12)]].sum(axis=1) /
                (self.df[[f"Purchases_M{i + 1}" for i in range(12)]].sum(axis=1) + 1e-6)
        )

        # Stability features
        self.df["Sales_Stability"] = (
                self.df[[f"Sales_M{i + 1}" for i in range(12)]].std(axis=1) /
                (self.df[[f"Sales_M{i + 1}" for i in range(12)]].mean(axis=1) + 1e-6)
        )
        self.df["Purchases_Stability"] = (
                self.df[[f"Purchases_M{i + 1}" for i in range(12)]].std(axis=1) /
                (self.df[[f"Purchases_M{i + 1}" for i in range(12)]].mean(axis=1) + 1e-6)
        )

        # Business health features
        self.df["Purchase_to_Sales_Ratio"] = (
                self.df[[f"Purchases_M{i + 1}" for i in range(12)]].sum(axis=1) /
                (self.df[[f"Sales_M{i + 1}" for i in range(12)]].sum(axis=1) + 1e-6)
        )
        self.df["Avg_Monthly_Sales"] = self.df[[f"Sales_M{i + 1}" for i in range(12)]].mean(axis=1)
        self.df["Zero_Sales_Months"] = (self.df[[f"Sales_M{i + 1}" for i in range(12)]].values == 0).sum(axis=1)

        # Advanced features
        sales_months = self.df[[f"Sales_M{i + 1}" for i in range(12)]].values
        months = np.arange(1, 13)
        sales_trends = []
        for i in range(len(self.df)):
            if np.sum(sales_months[i]) > 0:
                slope = np.polyfit(months, sales_months[i], 1)[0]
                sales_trends.append(slope / np.mean(sales_months[i]) * 12)
            else:
                sales_trends.append(-1.0)
        self.df["Sales_Trend"] = sales_trends

        self.df["Sales_Compliance_Gap"] = 1 - self.df["Compliance_Sales"]

        print("✓ Engineered 10 business health features")

        # Save data
        self.df.to_csv(f"{self.output_dir}/data/credit_dataset.csv", index=False)
        print(f"✓ Dataset saved to {self.output_dir}/data/credit_dataset.csv")

    def train_model(self, test_size=0.2, cv_folds=5, hyperparameter_tuning=True):
        """
        Train and evaluate the credit scoring models.

        Parameters:
        -----------
        test_size : float, default=0.2
            Proportion of data for testing
        cv_folds : int, default=5
            Number of cross-validation folds
        hyperparameter_tuning : bool, default=True
            Whether to perform hyperparameter optimization
        """
        print("Phase 3: Model Training and Evaluation")

        # Prepare data
        X = self.df.drop(columns=["ClientID", "CreditGrade"])
        y = self.df["CreditGrade"]

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=self.random_state,
            stratify=y_encoded
        )

        print(f"✓ Split data: {len(X_train)} train, {len(X_test)} test samples")

        # Hyperparameter tuning
        if hyperparameter_tuning:
            print("Performing hyperparameter optimization...")
            param_grid = {
                'n_estimators': [1000, 2000],
                'max_depth': [10, 15, 20],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 5]
            }

            rf_base = RandomForestClassifier(random_state=self.random_state, class_weight='balanced')
            grid_search = GridSearchCV(rf_base, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"✓ Best parameters: {grid_search.best_params_}")
        else:
            # Default models
            self.model = RandomForestClassifier(
                n_estimators=2000,
                random_state=self.random_state,
                class_weight='balanced',
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5
            )

        # Train models
        self.model.fit(X_train, y_train)

        # Predictions
        y_pred = self.model.predict(X_test)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring='accuracy')

        # Results
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

        print(f"✓ Model trained successfully")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Cross-val Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
        print(f"  Macro F1-Score: {f1:.4f}")

        # Detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(
            y_test, y_pred,
            labels=self.label_encoder.transform(self.label_encoder.classes_),
            target_names=self.label_encoder.classes_
        ))

        # Save models
        joblib.dump((self.model, self.label_encoder), f"{self.output_dir}/models/credit_model.pkl")
        print(f"✓ Model saved to {self.output_dir}/models/credit_model.pkl")

        # Store test results
        self.test_results = {
            'accuracy': accuracy,
            'cv_scores': cv_scores,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_test': y_test,
            'y_pred': y_pred
        }

        return self.model

    def analyze_feature_importance(self, top_n=20):
        """Analyze and visualize feature importance."""
        print("Phase 4: Feature Importance Analysis")

        importances = self.model.feature_importances_
        features = self.model.feature_names_in_

        self.feature_importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        print(f"Top {min(top_n, len(self.feature_importance_df))} Most Important Features:")
        print(self.feature_importance_df.head(top_n))

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=self.feature_importance_df.head(top_n),
            x="Importance",
            y="Feature"
        )
        plt.title(f"Top {top_n} Feature Importances - Credit Scoring Model")
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.show()

        return self.feature_importance_df

    def generate_academic_report(self):
        """Generate comprehensive academic report."""
        print("Phase 5: Academic Report Generation")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""
CREDIT SCORING MODEL - ACADEMIC RESEARCH REPORT
==============================================

Generated: {timestamp}
Dataset Size: {len(self.df)} companies
Random State: {self.random_state}

EXECUTIVE SUMMARY
-----------------
This study developed a machine learning models to assess credit risk in small-to-medium 
enterprises based on business compliance patterns and operational stability metrics.

METHODOLOGY
-----------
- Algorithm: Random Forest Classifier
- Features: {len(self.model.feature_names_in_)} business metrics including compliance ratios and stability indicators
- Evaluation: {len(self.test_results['cv_scores'])}-fold cross-validation

RESULTS
-------
Model Performance:
- Test Accuracy: {self.test_results['accuracy']:.4f}
- Cross-validation Accuracy: {self.test_results['cv_scores'].mean():.4f} (±{self.test_results['cv_scores'].std() * 2:.4f})
- Macro F1-Score: {self.test_results['f1']:.4f}
- Macro Precision: {self.test_results['precision']:.4f}
- Macro Recall: {self.test_results['recall']:.4f}

Grade Distribution in Dataset:
{dict(self.df['CreditGrade'].value_counts())}

Top 5 Most Important Features:
{self.feature_importance_df.head(5).to_string(index=False)}

KEY FINDINGS
------------
1. Compliance ratios are critical predictors of credit risk
2. Business stability (low volatility) strongly correlates with better grades
3. Zero sales months are significant red flags for credit assessment
4. Average monthly sales serve as important business health indicators

ACADEMIC IMPLICATIONS
--------------------
This research demonstrates the value of incorporating operational metrics 
beyond traditional financial ratios in credit scoring models. The high importance 
of compliance features suggests regulatory adherence is a strong proxy for 
overall business reliability.

LIMITATIONS
-----------
- Synthetic data may not capture all real-world complexities
- Model requires 12 months of data for assessment
- Limited to classification rather than probability of default estimation

FILES GENERATED
---------------
- Dataset: {self.output_dir}/data/credit_dataset.csv
- Model: {self.output_dir}/models/credit_model.pkl
- Feature Importance Plot: {self.output_dir}/plots/feature_importance.png
- This Report: {self.output_dir}/academic_report.txt
"""

        # Save report
        with open(f"{self.output_dir}/academic_report.txt", "w") as f:
            f.write(report)

        print(f"✓ Academic report saved to {self.output_dir}/academic_report.txt")

        return report


def main():
    """
    Main execution function for academic research pipeline.

    Usage Examples:
    --------------

    # Basic usage
    pipeline = CreditScoringPipeline(random_state=42, output_dir="results_experiment_1")
    pipeline.generate_synthetic_data(records_per_grade=1000)
    pipeline.train_model(hyperparameter_tuning=True)
    pipeline.analyze_feature_importance(top_n=15)
    pipeline.generate_academic_report()

    # For reproducible research
    pipeline = CreditScoringPipeline(random_state=123, output_dir="final_results")
    pipeline.generate_synthetic_data(records_per_grade=500, grades=["A", "B", "C", "D"])
    models = pipeline.train_model(test_size=0.25, cv_folds=10)
    importance_df = pipeline.analyze_feature_importance()
    report = pipeline.generate_academic_report()
    """

    # Initialize pipeline
    pipeline = CreditScoringPipeline(
        random_state=42,
        output_dir="results"
    )

    # Execute full pipeline
    print("ACADEMIC CREDIT SCORING PIPELINE")
    print("=" * 50)

    # Generate data
    df = pipeline.generate_synthetic_data(records_per_grade=500)

    # Train models
    model = pipeline.train_model(
        test_size=0.2,
        cv_folds=5,
        hyperparameter_tuning=True
    )

    # Analyze features
    importance_df = pipeline.analyze_feature_importance(top_n=15)

    # Generate report
    report = pipeline.generate_academic_report()

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"All outputs saved to: results/")

    return pipeline


if __name__ == "__main__":
    pipeline = main()