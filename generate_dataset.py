# generate_dataset.py
import pandas as pd

def generate_client_data(client_id, grade_target):
    import numpy as np
    np.random.seed(client_id)

    # Base business size (independent of credit grade)
    business_size = np.random.choice(['micro', 'small', 'medium'], p=[0.7, 0.25, 0.05])
    if business_size == 'micro':
        base_sales = np.random.uniform(50000, 500000)  # 50K-500K RWF monthly
    elif business_size == 'small':
        base_sales = np.random.uniform(300000, 2000000)  # 300K-2M RWF monthly
    else:  # medium
        base_sales = np.random.uniform(1500000, 4200000)  # 1.5M-4.2M RWF monthly

    # Generate data based on initial target grade
    if grade_target == "A":
        # base_sales = np.random.uniform(4000, 6000)
        sales = base_sales + np.random.normal(0, base_sales * 0.08, 12)
        purchases = sales * np.random.uniform(0.60, 0.75, 12)  # Good profit margins
        declared_sales = sales * np.random.uniform(0.96, 1.0, 12)
        declared_purchases = purchases * np.random.uniform(0.85, 1.0, 12)
    elif grade_target == "B":
        # base_sales = np.random.uniform(3000, 7000)
        sales = base_sales + np.random.normal(0, base_sales * 0.15, 12)
        purchases = sales * np.random.uniform(0.65, 0.80, 12)
        declared_sales = sales * np.random.uniform(0.91, 0.95, 12)
        declared_purchases = purchases * np.random.uniform(0.85, 1.0, 12)
    elif grade_target == "C":
        # base_sales = np.random.uniform(2000, 8000)
        sales = base_sales + np.random.normal(0, base_sales * 0.25, 12)
        purchases = sales * np.random.uniform(0.70, 0.85, 12)
        declared_sales = sales * np.random.uniform(0.86, 0.9, 12)
        declared_purchases = purchases * np.random.uniform(0.85, 1.0, 12)
    else:  # D
        sales = base_sales + np.random.normal(0, base_sales * 0.4, 12)
        purchases = sales * np.random.uniform(0.75, 1.0, 12)
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
        "Business_Size": business_size,
        **{f"Sales_M{i + 1}": round(sales[i]) for i in range(12)},
        **{f"Purchases_M{i + 1}": round(purchases[i]) for i in range(12)},
        **{f"Decl_Sales_M{i + 1}": round(declared_sales[i]) for i in range(12)},
        **{f"Decl_Purchases_M{i + 1}": round(declared_purchases[i]) for i in range(12)},
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
df.to_csv("dataset/credit_data.csv", index=False)

print("\nClass Distribution (CreditGrade):")
print(df["CreditGrade"].value_counts())
