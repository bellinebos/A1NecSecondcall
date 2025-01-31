import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Load the training and testing data
train_data = pd.read_csv('traindata.csv')
test_data = pd.read_csv('testdata.csv')

# Separate features (X) and target variable (y)
X_train = train_data.drop(columns=['Life expectancy ']).values
y_train = train_data['Life expectancy '].values

X_test = test_data.drop(columns=['Life expectancy ']).values
y_test = test_data['Life expectancy '].values

# Scale the features and target using MinMaxScaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# Define Neural Network Model
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)  # Output layer for regression (no activation)
        )

    def forward(self, x):
        return self.model(x)

# Initialize model
input_size = X_train.shape[1]
model = NeuralNet(input_size)

# Set training parameters
learning_rate = 0.01
epochs = 1000
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Using pure SGD

# Training Loop
train_losses = []
for epoch in range(epochs):
    model.train()
    
    total_loss = 0
    for i in range(len(X_train_tensor)):  # Pure SGD (one sample at a time)
        optimizer.zero_grad()
        prediction = model(X_train_tensor[i].unsqueeze(0))  # Process one sample
        loss = criterion(prediction, y_train_tensor[i].unsqueeze(0))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    train_losses.append(total_loss / len(X_train_tensor))  # Average loss per epoch
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(X_train_tensor):.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy()

# Reverse scaling to original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Performance Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")

# Scatter plot for BP-F model
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Line of perfect prediction
plt.title('BP-F Model: Predicted vs Actual Values')
plt.xlabel('Actual Values (zμ)')
plt.ylabel('Predicted Values (ŷμ)')
plt.grid(True)
plt.show()

