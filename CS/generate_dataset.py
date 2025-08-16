import pandas as pd

def generate_client_datassss(client_id, grade_target):
    import numpy as np
    np.random.seed(client_id)  # reproducible per client

    # --- Step 1. Generate synthetic monthly data by target grade ---
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

    # --- Step 2. Derived features ---
    total_sales = sales.sum() if sales.sum() != 0 else 1
    total_purchases = purchases.sum() if purchases.sum() != 0 else 1

    compliance_sales = min(declared_sales.sum() / total_sales, 1.0)
    compliance_purchases = min(declared_purchases.sum() / total_purchases, 1.0)

    sales_volatility = np.std(sales) / np.mean(sales)
    purchases_volatility = np.std(purchases) / np.mean(purchases)

    profitability = (total_sales - total_purchases) / (total_sales + 1)
    sales_growth = (sales[-1] + 1) / (sales[0] + 1)
    sales_max_to_mean = sales.max() / (np.mean(sales) if np.mean(sales) != 0 else 1)

    # --- Step 3. Problematic months (extra signals) ---
    zero_sales_months = np.sum(sales == 0)
    zero_declared_months = np.sum(declared_sales == 0)
    high_mismatch_months = np.sum(np.abs(declared_sales - sales) > (sales * 0.2))

    # --- Step 4. Enhanced Grade Refinement ---
    if (
        compliance_sales > 0.95 and
        sales_volatility < 0.15 and
        profitability > 0.15 and
        sales_growth > 1.05 and
        sales_max_to_mean < 2 and
        zero_sales_months < 2 and
        zero_declared_months < 2 and
        high_mismatch_months < 2
    ):
        grade = "A"

    elif (
        compliance_sales > 0.9 and
        sales_volatility < 0.25 and
        profitability > 0.05 and
        sales_growth > 1.0 and
        sales_max_to_mean < 3 and
        zero_sales_months < 2
    ):
        grade = "B"

    elif (
        compliance_sales > 0.85 and
        sales_volatility < 0.4 and
        profitability > -0.05
    ):
        grade = "C"

    else:
        grade = "D"

    # --- Step 5. Return record ---
    return {
        "ClientID": f"C{client_id:03d}",
        **{f"Sales_M{i + 1}": sales[i] for i in range(12)},
        **{f"Purchases_M{i + 1}": purchases[i] for i in range(12)},
        **{f"Decl_Sales_M{i + 1}": declared_sales[i] for i in range(12)},
        **{f"Decl_Purchases_M{i + 1}": declared_purchases[i] for i in range(12)},
        "CreditGrade": grade,
        "Compliance_Sales": compliance_sales,
        "Compliance_Purchases": compliance_purchases,
        "Sales_Volatility": sales_volatility,
        "Profitability": profitability,
        "Sales_Growth": sales_growth,
        "Sales_Max_to_Mean": sales_max_to_mean
    }
