![RobinHoodFortuneTellerGIF](https://github.com/user-attachments/assets/39a2868a-ae57-4765-aeaa-f4df7239223e)

#   Intern Performance Prediction System 


A machine learning project designed to predict whether an intern is likely to excel, be strictly average, or struggle, based on three main parameters:
- **Completion Time**
- **Feedback Score**
- **Attendance**

## 🚀 Features
- **Dual Models:** Trains both **Random Forest** and **XGBoost** models on the dataset.
- **Dynamic Selection:** Automatically evaluates Mean Squared Error (MSE) on the test split and selects the best performing model.
- **Interactive UI:** Provides a beautiful ASCII interface for users to manually input values and get immediate, color-coded predictions.

## 🛠️ Usage
To run the prediction system, simply execute the main script:
```bash
python main.py
```
You will be prompted to enter the intern's details. Example: `5, 8, 90`.

## 📦 Requirements
Ensure you have the required packages installed:
```bash
pip install -r requirements.txt
```
