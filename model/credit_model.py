import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_credit_model(file_path):
    df = pd.read_csv(file_path)

    # Create features
    df['sales_gap'] = df['sales_internal'] - df['sales_tax']
    df['purchase_gap'] = df['purchase_internal'] - df['purchase_tax']

    # Simulate credit limit as 20% of internal sales (you can replace with real targets)
    df['credit_limit'] = df['sales_internal'] * 0.2

    features = ['sales_internal', 'purchase_internal', 'sales_gap', 'purchase_gap', 'tax_compliance']
    target = 'credit_limit'

    # Split the dataset into training and testing sets (70/30 split)
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=42)

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
    predictions = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    print(f"Model trained. Test MSE: {mse:.2f}")

    return model, X_test, y_test, predictions
