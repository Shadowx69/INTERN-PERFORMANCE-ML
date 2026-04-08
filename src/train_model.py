from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import os

def train_models(X_train, y_train):

    # Baseline Model
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)

    # Advanced Model
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)

    joblib.dump(rf_model, "models/random_forest.pkl")
    joblib.dump(xgb_model, "models/xgboost.pkl")

    return rf_model, xgb_model