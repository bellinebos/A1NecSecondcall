import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Load data
train_data = pd.read_csv('traindata.csv')
test_data = pd.read_csv('testdata.csv')

X_train = train_data.drop('Life expectancy ', axis=1).values
y_train = train_data['Life expectancy '].values
X_test = test_data.drop('Life expectancy ', axis=1).values
y_test = test_data['Life expectancy '].values

# Scale data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

class CustomCallback(tf.keras.callbacks.Callback):
   def on_epoch_end(self, epoch, logs=None):
       if epoch % 100 == 0:
           print(f"Epoch {epoch}/{self.params['epochs']}, Total Error: {logs['loss']:.6f}")

def main():
    print("\nTesting Configuration:")
    print("Description: 5-layer ReLU with TensorFlow")
    print("Architecture: [13, 24, 18, 12, 1]")
    print("Learning Rate: 0.01, Momentum: 0.85")
    print("Activation: relu, Epochs: 1600")

    model = tf.keras.Sequential([
       tf.keras.layers.Dense(24, activation='relu', input_shape=(13,)),
       tf.keras.layers.Dense(18, activation='relu'),
       tf.keras.layers.Dense(12, activation='relu'),
       tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.85)
    model.compile(optimizer=optimizer, loss='mse')

    history = model.fit(
       X_train_scaled, y_train_scaled,
       epochs=1600,
       batch_size=5,
       validation_split=0.2,
       verbose=0,
       callbacks=[CustomCallback()]
    )

    y_pred_train = model.predict(X_train_scaled, verbose=0)
    y_pred_test = model.predict(X_test_scaled, verbose=0)

    y_pred_train = scaler_y.inverse_transform(y_pred_train)
    y_pred_test = scaler_y.inverse_transform(y_pred_test)

    # Calculate metrics
    mse_train = mean_squared_error(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mape_train = mean_absolute_percentage_error(y_train, y_pred_train)

    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)

    print("\nTraining Metrics:")
    print(f"MSE: {mse_train:.4f}")
    print(f"MAE: {mae_train:.4f}")
    print(f"MAPE: {mape_train:.4f}")

    print("\nTest Metrics:")
    print(f"MSE: {mse_test:.4f}")
    print(f"MAE: {mae_test:.4f}")
    print(f"MAPE: {mape_test:.4f}")

    # Create predictions plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values (zμ)', fontsize=12)
    plt.ylabel('Predicted Values (ŷμ)', fontsize=12)
    plt.title('BP-F Model: Predicted vs Actual Values', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
