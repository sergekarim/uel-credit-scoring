import pandas as pd

def add_features(df_new):
    """
    Enhance dataset with financial behavior features for better credit grade prediction.
    """
    # --- 1. Sales Compliance Scores ---
    df_new["Compliance_Sales"] = np.minimum(
        df_new[[f"Decl_Sales_M{i + 1}" for i in range(12)]].sum(axis=1) /
        df_new[[f"Sales_M{i + 1}" for i in range(12)]].sum(axis=1), 1.0
    )
    # --- 2. Purchase Compliance Scores ---
    df_new["Compliance_Purchases"] = np.minimum(
        df_new[[f"Decl_Purchases_M{i + 1}" for i in range(12)]].sum(axis=1) /
        df_new[[f"Purchases_M{i + 1}" for i in range(12)]].sum(axis=1), 1.0
    )

    # --- 3. Stability (Volatility) ---
    df_new["Sales_Stability"] = df_new[[f"Sales_M{i + 1}" for i in range(12)]].std(axis=1) / \
                                df_new[[f"Sales_M{i + 1}" for i in range(12)]].mean(axis=1)
    # --- 4.
    df_new["Purchases_Stability"] = df_new[[f"Purchases_M{i + 1}" for i in range(12)]].std(axis=1) / \
                                    df_new[[f"Purchases_M{i + 1}" for i in range(12)]].mean(axis=1)

    # --- 5. Purchase-to-Sales Ratio ---
    df_new["Purchase_to_Sales_Ratio"] = df_new[[f"Purchases_M{i + 1}" for i in range(12)]].sum(axis=1) / \
                                        df_new[[f"Sales_M{i + 1}" for i in range(12)]].sum(axis=1)

    # --- 6. Growth & Trends --- (Distinguishes SMEs with strong upward trends from stagnant ones (helps separate A vs C/D).)
    df_new["Sales_Growth"] = (df_new["Sales_M12"] + 1) / (df_new["Sales_M1"] + 1)   # +1 avoids div by 0
    df_new["Purchases_Growth"] = (df_new["Purchases_M12"] + 1) / (df_new["Purchases_M1"] + 1)

    # --- 7. Seasonality --- (High seasonality + low compliance = riskier firms.)
    df_new["Sales_Seasonality"] = df_new[[f"Sales_M{i + 1}" for i in range(12)]].std(axis=1) / \
                                  df_new[[f"Sales_M{i + 1}" for i in range(12)]].mean(axis=1)
    # --- 8.
    df_new["Peak_Sales_Month"] = df_new[[f"Sales_M{i + 1}" for i in range(12)]].idxmax(axis=1).str.extract("(\d+)").astype(int)

    # --- 9. Profitability Proxy --- (Higher profitability stability = more creditworthy.)
    total_sales = df_new[[f"Sales_M{i + 1}" for i in range(12)]].sum(axis=1)
    total_purchases = df_new[[f"Purchases_M{i + 1}" for i in range(12)]].sum(axis=1)
    df_new["Profitability"] = (total_sales - total_purchases) / (total_sales + 1)

    # --- 10. Outlier Detection --- (If a business has one spike month and low other months, it’s riskier.)
    df_new["Sales_Max_to_Mean"] = df_new[[f"Sales_M{i + 1}" for i in range(12)]].max(axis=1) / \
                                  df_new[[f"Sales_M{i + 1}" for i in range(12)]].mean(axis=1)

    # --- 11. Interaction Features --- (A firm that is highly compliant and stable should rank higher than one that’s only compliant.)
    df_new["Compliance_Adjusted"] = df_new["Compliance_Sales"] / (1 + df_new["Sales_Stability"])

    # --- 12. Rolling / Window Features --- (Helps detect recent declines/improvements in performance.)
    df_new["Q4_to_Annual_Sales"] = df_new[[f"Sales_M{i + 1}" for i in range(9, 12)]].sum(axis=1) / \
                                   df_new[[f"Sales_M{i + 1}" for i in range(12)]].sum(axis=1)

    return df_new


import numpy as np

def calculate_derived_features(client_data):
    """
    Calculate derived features from raw monthly data for a single client,
    ensuring realistic compliance ratios and stable calculations.
    """

    # Extract monthly values
    sales = np.array([client_data[f"Sales_M{i+1}"] for i in range(12)], dtype=float)
    purchases = np.array([client_data[f"Purchases_M{i+1}"] for i in range(12)], dtype=float)
    decl_sales = np.array([client_data[f"Decl_Sales_M{i+1}"] for i in range(12)], dtype=float)
    decl_purchases = np.array([client_data[f"Decl_Purchases_M{i+1}"] for i in range(12)], dtype=float)

    # Avoid division by zero
    total_sales = sales.sum() if sales.sum() != 0 else 1
    total_purchases = purchases.sum() if purchases.sum() != 0 else 1

    # --- Compliance Ratios ---
    compliance_sales = np.clip(decl_sales.sum() / total_sales, 0, 1.0)  # Cap at 1
    compliance_purchases = np.clip(decl_purchases.sum() / total_purchases, 0, 1.0)  # Cap at 1

    # --- Stability (Coefficient of Variation) ---
    sales_stability = np.std(sales) / sales.mean() if sales.mean() != 0 else 0
    purchases_stability = np.std(purchases) / purchases.mean() if purchases.mean() != 0 else 0

    # --- Purchase-to-Sales Ratio ---
    purchase_to_sales_ratio = total_purchases / total_sales

    # --- Growth ---
    sales_growth = (sales[-1] + 1) / (sales[0] + 1)
    purchases_growth = (purchases[-1] + 1) / (purchases[0] + 1)

    # --- Seasonality ---
    sales_seasonality = np.std(sales) / sales.mean() if sales.mean() != 0 else 0
    peak_sales_month = int(np.argmax(sales)) + 1

    # --- Profitability Proxy ---
    profitability_proxy = (total_sales - total_purchases) / (total_sales + 1)

    # --- Outlier Detection ---
    sales_max_to_mean = sales.max() / sales.mean() if sales.mean() != 0 else 0

    # --- Interaction Features ---
    compliance_adjusted = compliance_sales / (1 + sales_stability)

    # --- Rolling / Window Features ---
    q4_to_annual_sales = sales[9:12].sum() / total_sales

    derived_features = {
        "Compliance_Sales": compliance_sales,
        "Compliance_Purchases": compliance_purchases,
        "Sales_Stability": sales_stability,
        "Purchases_Stability": purchases_stability,
        "Purchase_to_Sales_Ratio": purchase_to_sales_ratio,
        "Sales_Growth": sales_growth,
        "Purchases_Growth": purchases_growth,
        "Sales_Seasonality": sales_seasonality,
        "Peak_Sales_Month": peak_sales_month,
        "Profitability": profitability_proxy,
        "Sales_Max_to_Mean": sales_max_to_mean,
        "Compliance_Adjusted": compliance_adjusted,
        "Q4_to_Annual_Sales": q4_to_annual_sales
    }

    return derived_features
