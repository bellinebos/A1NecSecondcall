# Import required libraries
import pandas as pd  # For data manipulation and reading CSV files
import tensorflow as tf  # Deep learning framework
from sklearn.preprocessing import MinMaxScaler  # For feature scaling
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error  # For model evaluation
import matplotlib.pyplot as plt  # For visualization

# Load and prepare training and test datasets
train_data = pd.read_csv('traindata.csv')  # Read training data from CSV
test_data = pd.read_csv('testdata.csv')    # Read test data from CSV

# Separate features (X) and target variable (y)
X_train = train_data.drop('Life expectancy ', axis=1).values  # Remove target column for training features
y_train = train_data['Life expectancy '].values              # Extract target variable for training
X_test = test_data.drop('Life expectancy ', axis=1).values   # Remove target column for test features
y_test = test_data['Life expectancy '].values                # Extract target variable for test

# Initialize scalers for feature scaling
scaler_X = MinMaxScaler()  # Scaler for input features
scaler_y = MinMaxScaler()  # Scaler for target variable

# Scale the input features and target variables
X_train_scaled = scaler_X.fit_transform(X_train)  # Fit and transform training features
X_test_scaled = scaler_X.transform(X_test)        # Transform test features using training fit
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))  # Fit and transform training targets
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))        # Transform test targets using training fit

# Custom callback class to monitor training progress
class CustomCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to print training progress every 100 epochs
    Inherits from Keras Callback class
    """
    def on_epoch_end(self, epoch, logs=None):
        # Print progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{self.params['epochs']}, Total Error: {logs['loss']:.6f}")

def main():
    # Print model configuration details
    print("\nTesting Configuration:")
    print("Description: 5-layer ReLU with TensorFlow")
    print("Architecture: [13, 24, 18, 12, 1]")  # Network architecture
    print("Learning Rate: 0.01, Momentum: 0.85")  # Training parameters
    print("Activation: relu, Epochs: 1600")  # Model hyperparameters

    # Create sequential model with specified architecture
    model = tf.keras.Sequential([
        # Input layer with 24 neurons, ReLU activation, expects 13 input features
        tf.keras.layers.Dense(24, activation='relu', input_shape=(13,)),
        # First hidden layer with 18 neurons and ReLU activation
        tf.keras.layers.Dense(18, activation='relu'),
        # Second hidden layer with 12 neurons and ReLU activation
        tf.keras.layers.Dense(12, activation='relu'),
        # Output layer with 1 neuron (regression task)
        tf.keras.layers.Dense(1)
    ])

    # Configure optimizer with SGD and momentum
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.85)
    
    # Compile model with mean squared error loss
    model.compile(optimizer=optimizer, loss='mse')

    # Train the model
    history = model.fit(
        X_train_scaled, y_train_scaled,  # Training data
        epochs=1600,                     # Number of training iterations
        batch_size=5,                    # Mini-batch size
        validation_split=0.2,            # 20% of data used for validation
        verbose=0,                       # Suppress default output
        callbacks=[CustomCallback()]      # Use custom progress monitoring
    )

    # Make predictions
    y_pred_train = model.predict(X_train_scaled, verbose=0)  # Training set predictions
    y_pred_test = model.predict(X_test_scaled, verbose=0)    # Test set predictions

    # Inverse transform predictions back to original scale
    y_pred_train = scaler_y.inverse_transform(y_pred_train)
    y_pred_test = scaler_y.inverse_transform(y_pred_test)

    # Calculate performance metrics
    # Training metrics
    mse_train = mean_squared_error(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
    
    # Test metrics
    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)

    # Print performance metrics
    print("\nTraining Metrics:")
    print(f"MSE: {mse_train:.4f}")
    print(f"MAE: {mae_train:.4f}")
    print(f"MAPE: {mape_train:.4f}")
    
    print("\nTest Metrics:")
    print(f"MSE: {mse_test:.4f}")
    print(f"MAE: {mae_test:.4f}")
    print(f"MAPE: {mape_test:.4f}")

    # Create visualization of predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5, color='blue')  # Plot predictions
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Perfect prediction line
    plt.xlabel('Actual Values (zμ)', fontsize=12)
    plt.ylabel('Predicted Values (ŷμ)', fontsize=12)
    plt.title('BP-F Model: Predicted vs Actual Values', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Program entry point
if __name__ == "__main__":
    main()
