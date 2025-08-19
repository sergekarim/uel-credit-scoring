# generate_dataset.py
import pandas as pd

def generate_client_data(client_id, grade_target):
    import numpy as np
    np.random.seed(client_id)

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

# Save to CSV (optional)
df.to_csv("dataset/credit_data-new.csv", index=False)

print("\nClass Distribution (CreditGrade):")
print(df["CreditGrade"].value_counts())
print("\nSample Data:")
print(df.head())