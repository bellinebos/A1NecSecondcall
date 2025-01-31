import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Load the training and testing data
train_data = pd.read_csv('traindata.csv')
test_data = pd.read_csv('testdata.csv')

# Separate the features (X) and target variable (y) for training and testing
X_train = train_data.drop(columns=['Life expectancy ']).values
y_train = train_data['Life expectancy '].values

X_test = test_data.drop(columns=['Life expectancy ']).values
y_test = test_data['Life expectancy '].values

# Initialize MinMaxScaler for scaling the features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Scale the features (X)
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scale the target variable (y)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Train the Multiple Linear Regression model
mlr_model = LinearRegression()
mlr_model.fit(X_train_scaled, y_train_scaled)

# Make predictions
y_pred_scaled = mlr_model.predict(X_test_scaled)

# Reverse the scaling for predictions
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")

# Scatter plot for MLR-F model
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Line of perfect prediction
plt.title('MLR-F Model: Predicted vs Actual Values')
plt.xlabel('Actual Values (zμ)')
plt.ylabel('Predicted Values (ŷμ)')
plt.grid(True)
plt.show()
