# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load the Dataset
data = pd.read_csv('car data.csv')  
print("Dataset loaded successfully.")
print(data.head())

# Data Cleaning and Exploratory Data Analysis (EDA)
print("\nData Information:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())

# Drop 'Car_Name' as it's not useful for the correlation plot or model
data = data.drop(columns=['Car_Name'])

# Encode categorical variables before calculating correlation
data = pd.get_dummies(data, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)

# Basic scatter plot (Present_Price vs. Selling_Price)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Present_Price', y='Selling_Price', data=data)
plt.title('Present Price vs Selling Price')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Feature Scaling
features_to_scale = ['Present_Price', 'Driven_kms', 'Year']
scaler = StandardScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Split Data into Training and Testing Sets
X = data.drop(columns=['Selling_Price'])  # Selling_Price is the target variable
y = data['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection and Training
# Initial Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predictions and Evaluation for Linear Regression
y_pred_lr = linear_model.predict(X_test)
print("\nLinear Regression Model Evaluation:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_lr))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_lr))
print("R² Score:", r2_score(y_test, y_pred_lr))

# Random Forest Model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions and Evaluation for Random Forest
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Model Evaluation:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_rf))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_rf))
print("R² Score:", r2_score(y_test, y_pred_rf))

# Hyperparameter Tuning for Random Forest Model
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='r2')
grid_search.fit(X_train, y_train)

print("\nBest Parameters from Grid Search:", grid_search.best_params_)
print("Best R² Score from Grid Search:", grid_search.best_score_)

# Feature Importance
importances = rf_model.feature_importances_
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()

# Save the Best Model
joblib.dump(rf_model, 'car price_prediction_model.pkl')
print("Model saved as 'car_price_prediction_model.pkl'.")
'''
# Entraînement du modèle (par exemple, un modèle RandomForest)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Sauvegarder le modèle entraîné en tant que fichier .pkl
import joblib 
joblib.dump(rf_model, 'car_price_prediction_model.pkl')
'''
