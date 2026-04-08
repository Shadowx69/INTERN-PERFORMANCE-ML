import joblib
import pandas as pd

def predict(user_input=None, better_model="xgb"):
    try:
        rf_model = joblib.load("models/random_forest.pkl")
        xgb_model = joblib.load("models/xgboost.pkl")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    if user_input is None:
        print("\n" + "=" * 60)
        print(" 🌟  INTERN PERFORMANCE PREDICTION SYSTEM  🌟 ")
        print("=" * 60)
        print(" Please enter the intern's details below.")
        print(" Format: time, feedback, attendance")
        print(" Example: 5, 8, 90")
        print("-" * 60)
        user_input = input(" 👉 Input parameters: ")
    
    try:
        # Split by comma, strip spaces, convert to float
        values = [float(v.strip()) for v in user_input.split(',')]
        if len(values) != 3:
            print("\n ❌ Error: Please provide exactly 3 comma-separated values.")
            print("=" * 60 + "\n")
            return
    except ValueError:
        print("\n ❌ Error: Invalid input! Please enter numbers only.")
        print("=" * 60 + "\n")
        return

    sample = pd.DataFrame([values],
                          columns=['completion_time', 'feedback_score', 'attendance'])

    rf_score = rf_model.predict(sample)[0]
    xgb_score = xgb_model.predict(sample)[0]

    best_score = rf_score if better_model == "rf" else xgb_score
    
    if best_score > 8:
        outcome = " EXCELLENT"
        outcome_color = "\033[92m" # Green
    elif best_score > 6:
        outcome = " AVERAGE"
        outcome_color = "\033[96m" # Cyan
    else:
        outcome = " STRUGGLING"
        outcome_color = "\033[91m" # Red

    print("\n" + "//" * 30)
    print(" "*18 + "🎯 PREDICTION RESULTS 🎯")
    print("//" * 30)
    print(f"   Random Forest Prediction : {rf_score:.2f}")
    print(f"   XGBoost Prediction       : {xgb_score:.2f}")
    print("=" * 60)
    print(f"  🔮 Predicted Outcome        : {outcome_color}{outcome}\033[0m")
    print("=" * 60 + "\n")