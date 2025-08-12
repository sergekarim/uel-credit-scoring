from model.credit_model import train_credit_model

if __name__ == "__main__":
    model, X_test, y_test, predictions = train_credit_model("data/sample_data.csv")
    results = X_test.copy()
    results['actual_credit_limit'] = y_test
    results['predicted_credit_limit'] = predictions
    print("\nSample Predictions:")
    print(results.head())
