import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# ---------------------------
# 1. Generate Simulated Data
# ---------------------------

# Number of samples (SMEs)
n_samples = 200

# Simulate internal sales and purchase data (e.g., in Rwandan francs)
sales_internal = np.random.normal(loc=500000, scale=100000, size=n_samples)
purchase_internal = np.random.normal(loc=300000, scale=75000, size=n_samples)

# Simulate tax declaration data (ideally close to internal records with some noise)
sales_tax = sales_internal * np.random.normal(loc=1.0, scale=0.05, size=n_samples)
purchase_tax = purchase_internal * np.random.normal(loc=1.0, scale=0.05, size=n_samples)

# Create a consistency score: smaller differences imply higher consistency.
sales_consistency = 1 - np.abs(sales_internal - sales_tax) / sales_internal
purchase_consistency = 1 - np.abs(purchase_internal - purchase_tax) / purchase_internal

# Derive additional features: for example, gross margin, growth rate, etc.
# Here, a simple margin based on internal data.
gross_margin = sales_internal - purchase_internal

# For target credit limit, we assume it depends on sales, margin, and consistency.
# This is a synthetic formula for demonstration:
credit_limit = (0.2 * sales_internal + 0.5 * gross_margin) * (sales_consistency + purchase_consistency) / 2

# Create a DataFrame with our simulated data
data = pd.DataFrame({
    'TIN': [f"TIN_{i+1:03d}" for i in range(n_samples)],
    'sales_internal': sales_internal,
    'purchase_internal': purchase_internal,
    'sales_tax': sales_tax,
    'purchase_tax': purchase_tax,
    'sales_consistency': sales_consistency,
    'purchase_consistency': purchase_consistency,
    'gross_margin': gross_margin,
    'credit_limit': credit_limit
})

print("Simulated Data (first 5 rows):")
print(data.head())

# -----------------------------------
# 2. Feature Engineering and Splitting
# -----------------------------------

# We select our features and target.
# In a real scenario, you might engineer more features (e.g., ratios, growth trends, etc.)
features = ['sales_internal', 'purchase_internal', 'sales_tax', 'purchase_tax',
            'sales_consistency', 'purchase_consistency', 'gross_margin']
target = 'credit_limit'

X = data[features]
y = data[target]

# Split the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 3. Model Development and Training
# -------------------------------

# We'll use a Random Forest Regressor as an example
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 4. Model Evaluation
# -------------------------------

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Visualize the Actual vs. Predicted credit limits
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Credit Limit")
plt.ylabel("Predicted Credit Limit")
plt.title("Actual vs. Predicted Credit Limit")
plt.show()

# -------------------------------
# 5. Model Explainability (Feature Importance)
# -------------------------------

importances = model.feature_importances_
feature_importance = pd.Series(importances, index=features).sort_values(ascending=False)
print("\nFeature Importances:")
print(feature_importance)

plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.title("Feature Importance from RandomForestRegressor")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()
