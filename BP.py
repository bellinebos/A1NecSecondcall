import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split

class NeuralNet:
    def __init__(self, layers, epochs, learning_rate, momentum, activation_function, validation_split=0.2):
        self.L = len(layers)  # Number of layers
        self.n = layers  # Number of neurons in each layer
        self.epochs = epochs
        self.learning_rate = learning_rate  # η (eta) in the document
        self.momentum = momentum  # α (alpha) in the document
        self.validation_split = validation_split
        self.fact = activation_function

        # Define activation functions and their derivatives
        self.activations = {
            "sigmoid": (self.sigmoid, self.sigmoid_derivative),
            "relu": (self.relu, self.relu_derivative),
            "linear": (self.linear, self.linear_derivative),
            "tanh": (self.tanh, self.tanh_derivative)
        }

        # Initialize arrays according to document notation
        # ξ (xi) - activations for each layer including input layer
        self.xi = [np.zeros((layer, 1)) for layer in layers]
        
        # h - fields for each layer (except input layer)
        self.h = [np.zeros((layer, 1)) for layer in layers]
        
        # ω (omega) - weights between layers
        self.w = [np.random.randn(layers[l], layers[l-1]) * 0.01 for l in range(1, self.L)]
        
        # θ (theta) - thresholds/biases for each layer except input
        self.theta = [np.zeros((layer, 1)) for layer in layers[1:]]
        
        # Δ (delta) - error terms for each layer except input
        self.delta = [np.zeros((layer, 1)) for layer in layers[1:]]
        
        # δω (delta_w) - weight changes
        self.d_w = [np.zeros_like(w) for w in self.w]
        
        # δθ (delta_theta) - threshold changes
        self.d_theta = [np.zeros_like(t) for t in self.theta]
        
        # Previous changes for momentum
        self.d_w_prev = [np.zeros_like(w) for w in self.w]
        self.d_theta_prev = [np.zeros_like(t) for t in self.theta]

    def forward(self, X):
        """
        Implementation of feed-forward propagation according to equations 6-9 in the document
        """
        # Equation 6: Input layer activation
        self.xi[0] = X
        
        # Equations 7-8: Calculate fields and activations for all other layers
        for l in range(1, self.L):
            # Equation 8: Calculate fields
            self.h[l] = np.dot(self.w[l-1], self.xi[l-1]) - self.theta[l-1]
            
            # Equation 7: Calculate activations using g(h)
            self.xi[l] = self.activation_function(self.h[l])
        
        # Note: output o(x) is automatically stored in self.xi[-1] (Equation 9)

    def backpropagate(self, X, y):
        """
        Implementation of error back-propagation according to equations 11-14 in the document
        """
        # Forward pass first
        self.forward(X)
        
        # Equation 11: Calculate delta for output layer
        self.delta[-1] = self.activation_derivative(self.h[-1]) * (self.xi[-1] - y)
        
        # Equation 12: Back-propagate deltas to hidden layers
        for l in range(self.L-2, 0, -1):
            self.delta[l-1] = self.activation_derivative(self.h[l]) * np.dot(self.w[l].T, self.delta[l])
        
        # Equation 14: Calculate weight and threshold changes with momentum
        for l in range(self.L-1):
            # Weight changes
            self.d_w[l] = -self.learning_rate * np.dot(self.delta[l], self.xi[l].T) + \
                         self.momentum * self.d_w_prev[l]
            
            # Threshold changes
            self.d_theta[l] = self.learning_rate * self.delta[l] + \
                            self.momentum * self.d_theta_prev[l]
            
            # Store current changes for next iteration's momentum
            self.d_w_prev[l] = self.d_w[l]
            self.d_theta_prev[l] = self.d_theta[l]
            
            # Equation 15: Update weights and thresholds
            self.w[l] += self.d_w[l]
            self.theta[l] += self.d_theta[l]
        
        # Return error for this pattern
        return np.mean((self.xi[-1] - y) ** 2)

    def sigmoid(self, z):
        """Equation 10: Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        """Equation 13: Derivative of sigmoid function"""
        sig = self.sigmoid(z)
        return sig * (1 - sig)
    
    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def linear(self, z):
        return z

    def linear_derivative(self, z):
        return np.ones_like(z)

    def tanh(self, z):
        return np.tanh(z)

    def tanh_derivative(self, z):
        return 1 - np.tanh(z) ** 2

    def activation_function(self, z, is_output_layer=False):
        activation_func, _ = self.activations[self.fact]
        return activation_func(z)

    def activation_derivative(self, z, is_output_layer=False):
        _, activation_deriv = self.activations[self.fact]
        return activation_deriv(z)

    def forward(self, X):
        self.xi[0] = X  # Input layer activations
        
        for l in range(1, self.L):
            # Calculate fields (h) for current layer
            self.h[l] = np.dot(self.w[l-1], self.xi[l-1]) + self.theta[l-1]
            # Calculate activations (xi) using the activation function
            self.xi[l] = self.activation_function(self.h[l], is_output_layer=(l == self.L-1))

    def backpropagate(self, X, y):
        # Forward pass
        self.forward(X)
        
        # Output layer error
        output_error = self.xi[-1] - y
        self.delta[-1] = output_error * self.activation_derivative(self.h[-1], is_output_layer=True)
        
        # Hidden layers error
        for l in range(self.L-2, 0, -1):
            self.delta[l-1] = np.dot(self.w[l].T, self.delta[l]) * self.activation_derivative(self.h[l])
        
        # Update weights and biases with momentum
        for l in range(self.L-1):
            # Calculate weight and bias changes
            self.d_w[l] = -self.learning_rate * np.dot(self.delta[l], self.xi[l].T)
            self.d_theta[l] = -self.learning_rate * np.sum(self.delta[l], axis=1, keepdims=True)
            
            # Apply momentum
            self.d_w[l] += self.momentum * self.d_w_prev[l]
            self.d_theta[l] += self.momentum * self.d_theta_prev[l]
            
            # Update weights and biases
            self.w[l] += self.d_w[l]
            self.theta[l] += self.d_theta[l]
            
            # Store current changes for next iteration's momentum
            self.d_w_prev[l] = self.d_w[l]
            self.d_theta_prev[l] = self.d_theta[l]
        
        return np.mean(output_error**2)  # Return MSE loss

    def fit(self, X, y):
        """Train the network using backpropagation and return training history."""
        train_losses = []
        
        for epoch in range(self.epochs):
            # Perform one round of backpropagation and get the loss
            loss = self.backpropagate(X, y)
            train_losses.append(loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {loss:.4f}")
        
        return train_losses

    def predict(self, X):
        """Perform a forward pass and return predictions."""
        self.forward(X)
        return self.xi[-1]

    def loss_epochs(self, X_train, y_train, X_val, y_val):
        """Track training and validation loss over epochs."""
        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            # Train one epoch and get training loss
            train_loss = self.backpropagate(X_train, y_train)
            train_losses.append(train_loss)
            
            # Calculate validation loss
            self.forward(X_val)
            val_loss = np.mean((self.xi[-1] - y_val) ** 2)
            val_losses.append(val_loss)

        return np.array(train_losses), np.array(val_losses)

    def split_data(self, X, y):
        """Split data into training and validation sets."""
        validation_size = int(len(X) * self.validation_split)
        X_train = X[validation_size:]
        y_train = y[validation_size:]
        X_val = X[:validation_size]
        y_val = y[:validation_size]
        return X_train, y_train, X_val, y_val

# Utility functions
def standardize(data):
    """Standardize the data using z-score normalization."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return standardized_data, mean, std

def destandardize(data, mean, std):
    """Convert standardized data back to original scale."""
    return data * std + mean

def load_data(train_data, test_data):
    """Load and preprocess the data from CSV files."""
    train = pd.read_csv(train_data)
    test = pd.read_csv(test_data)

    # Separate features and targets
    X_train_val = train.iloc[:, :-1].values
    y_train_val = train.iloc[:, -1].values
    X_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values

    # Standardize
    X_train_std, X_mean, X_std = standardize(X_train_val)
    X_test_std = (X_test - X_mean) / X_std

    y_train_std, y_mean, y_std = standardize(y_train_val)
    y_test_std = (y_test - y_mean) / y_std

    return X_train_std, y_train_std, X_test_std, y_test_std, y_mean, y_std

# Example usage and evaluation
def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics."""
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    return mse, mae, mape

def plot_training_history(train_losses, val_losses):
    """Plot training and validation loss over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    X_train, y_train, X_test, y_test, y_mean, y_std = load_data("traindata.csv", "testdata.csv")

    # Define the combinations to evaluate
    combinations = [
        {'layers': [14, 9, 1], 'epochs': 2000, 'learning_rate': 0.01, 'momentum': 0.9, 'activation': 'relu'},
        {'layers': [14, 16, 8, 1], 'epochs': 1000, 'learning_rate': 0.001, 'momentum': 0.8, 'activation': 'tanh'},
        {'layers': [14, 9, 5, 1], 'epochs': 2000, 'learning_rate': 0.01, 'momentum': 0.7, 'activation': 'sigmoid'},
        {'layers': [14, 10, 1], 'epochs': 1500, 'learning_rate': 0.005, 'momentum': 0.7, 'activation': 'relu'},
        {'layers': [14, 12, 8, 1], 'epochs': 1000, 'learning_rate': 0.001, 'momentum': 0.9, 'activation': 'tanh'},
        {'layers': [14, 16, 1], 'epochs': 1000, 'learning_rate': 0.0005, 'momentum': 0.9, 'activation': 'relu'},
        {'layers': [14, 20, 15, 5, 1], 'epochs': 1500, 'learning_rate': 0.01, 'momentum': 0.9, 'activation': 'sigmoid'},
        {'layers': [14, 8, 1], 'epochs': 1000, 'learning_rate': 0.005, 'momentum': 0.8, 'activation': 'relu'},
        {'layers': [14, 9, 1], 'epochs': 1500, 'learning_rate': 0.0001, 'momentum': 0.9, 'activation': 'tanh'},
        {'layers': [14, 18, 10, 1], 'epochs': 2000, 'learning_rate': 0.001, 'momentum': 0.8, 'activation': 'relu'}
        ]

 # Initialize an empty list to store performance results
    performance_results = []

    # Iterate over combinations and evaluate
    for comb in combinations:
        nn = NeuralNet(
            layers=comb['layers'],
            epochs=comb['epochs'],
            learning_rate=comb['learning_rate'],
            momentum=comb['momentum'],
            activation_function=comb['activation'],
            validation_split=0.2
        )

        # Split data and train the network
        X_train_split, y_train_split, X_val, y_val = nn.split_data(X_train, y_train)
        train_losses, val_losses = nn.loss_epochs(X_train_split.T, y_train_split.T, X_val.T, y_val.T)

        # Make predictions
        predictions_std = nn.predict(X_test.T)
        predictions = destandardize(predictions_std, y_mean, y_std)
        y_test_original = destandardize(y_test, y_mean, y_std)

        # Evaluate the model
        mse, mae, mape = evaluate_model(y_test_original, predictions)

        # Store the performance results along with the combination
        performance_results.append({
            'combination': comb,
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'predictions': predictions,
            'y_test_original': y_test_original
        })

    # Print evaluation metrics (MSE, MAE, MAPE) for all combinations
    print("Evaluation metrics for all combinations:")
    for i, result in enumerate(performance_results):
        print(f"\nCombination {i+1}: {result['combination']}")
        print(f"MSE: {result['mse']}, MAE: {result['mae']}, MAPE: {result['mape']}")

    # Sort the results by MSE (or another metric)
    sorted_results = sorted(performance_results, key=lambda x: x['mse'])

    # Select the top 2 combinations based on MSE (or another metric)
    best_two = sorted_results[:2]

    # Print evaluation metrics (MSE, MAE, MAPE) for each combination
    for i, best in enumerate(best_two):
        print(f"\nBest combination {i+1}: {best['combination']}")
        print(f"MSE: {best['mse']}, MAE: {best['mae']}, MAPE: {best['mape']}")

        # Plot training and validation losses
        plot_training_history(best['train_losses'], best['val_losses'])

        # Plot Predicted vs Actual for the two best models
        plt.figure(figsize=(10, 6))
        plt.scatter(best['y_test_original'], best['predictions'], color='blue', label='Predicted vs Actual')
        plt.plot([min(best['y_test_original']), max(best['y_test_original'])], 
                 [min(best['y_test_original']), max(best['y_test_original'])], color='red', label='Ideal Line')
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Predicted vs Actual for Best Combination {i+1}")
        plt.legend()
