from src.data_loader import load_data
from src.train_model import train_models
from src.evaluate import evaluate
from src.predict import predict

print("\n" + "=" * 60)
print("   INTERN PERFORMANCE PREDICTION SYSTEM   ")
print("=" * 60)
print(" Please enter the intern's details below.")
print(" Format: time, feedback, attendance")
print(" Example: 5, 8, 90")
print("-" * 60)

user_input = input("  Input parameters: ")

# Load data
X_train, X_test, y_train, y_test = load_data("data/dataset.csv")

# Train
rf_model, xgb_model = train_models(X_train, y_train)

# Evaluate
rf_mse = evaluate(rf_model, X_test, y_test, "Random Forest")
xgb_mse = evaluate(xgb_model, X_test, y_test, "XGBoost")

# Compare
print("\nModel Comparison:")
better_model = "xgb"
if rf_mse < xgb_mse:
    print("Random Forest is better")
    better_model = "rf"
else:
    print("XGBoost is better")

# Predict
predict(user_input, better_model)