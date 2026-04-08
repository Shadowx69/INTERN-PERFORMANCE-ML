from sklearn.metrics import mean_squared_error

def evaluate(model, X_test, y_test, name):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    print(f"{name} MSE:", mse)
    return mse