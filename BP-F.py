import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

class NeuralNet:
    def __init__(self, num_layers, units_per_layer, num_epochs=1000, learning_rate=0.01, 
                 momentum=0.9, activation='sigmoid', validation_split=0.2, batch_size=5):
        """
        Initialize the Neural Network
        
        Parameters:
        num_layers (int): Number of layers in the network
        units_per_layer (list): Number of units in each layer (including input and output layers)
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate (η)
        momentum (float): Momentum coefficient (α)
        activation (str): Activation function ('sigmoid', 'relu', 'tanh', or 'linear')
        validation_split (float): Fraction of data to use for validation (0 to 1)
        batch_size (int): Size of mini-batches for training
        """
        self.L = num_layers
        self.n = units_per_layer
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.fact = activation
        self.validation_split = validation_split
        self.batch_size = batch_size
        
        # Initialize arrays following equations (7)-(9)
        self.h = [np.zeros(n) for n in self.n]  # Fields h(l)_i
        self.xi = [np.zeros(n) for n in self.n]  # Activations ξ(l)_i
        
        # Initialize weights with small random values
        self.w = [None]  # w[1] is not used as per document notation
        for l in range(1, self.L):
            # He initialization for weights
            self.w.append(np.random.randn(self.n[l], self.n[l-1]) * np.sqrt(2.0/self.n[l-1]))
            
        # Initialize thresholds
        self.theta = [None]  # theta[1] is not used
        for l in range(1, self.L):
            self.theta.append(np.random.randn(self.n[l]) * 0.1)
        
        # Arrays for backpropagation equations (11)-(12)
        self.delta = [np.zeros(n) for n in self.n]  # Δ(l)_i
        
        # Arrays for weight and threshold updates equation (14)
        self.d_w_prev = [None] + [np.zeros((self.n[l], self.n[l-1])) for l in range(1, self.L)]
        self.d_theta_prev = [None] + [np.zeros(self.n[l]) for l in range(1, self.L)]
        
        self.train_errors = []
        self.val_errors = []
    
    def activation(self, h, derivative=False):
        """Implementation of g(h) and g'(h) from equations (10) and (13)"""
        if self.fact == 'sigmoid':
            if not derivative:
                return 1 / (1 + np.exp(-h))  # Equation (10)
            else:
                g_h = self.activation(h)
                return g_h * (1 - g_h)  # Equation (13)
        elif self.fact == 'relu':
            if not derivative:
                return np.maximum(0, h)
            else:
                return (h > 0).astype(float)
        elif self.fact == 'tanh':
            if not derivative:
                return np.tanh(h)
            else:
                return 1 - np.tanh(h)**2
        elif self.fact == 'linear':
            if not derivative:
                return h
            else:
                return 1
        else:
            raise ValueError("Invalid activation function specified.")
    
    def feed_forward(self, x):
        """Implementation of equations (6)-(9)"""
        # Equation (6): Set input layer
        self.xi[0] = x
        
        # Equations (7)-(8): Calculate activations for each layer
        for l in range(1, self.L):
            # Equation (8): Calculate fields
            self.h[l] = np.dot(self.w[l], self.xi[l-1]) - self.theta[l]
            # Equation (7): Calculate activations
            self.xi[l] = self.activation(self.h[l])
        
        # Equation (9): Return output
        return self.xi[self.L-1]
    
    def update_weights_batch(self, batch_X, batch_y):
        """Implementation of equation (16) for partial batched BP"""
        # Initialize accumulated gradients
        d_w_batch = [None] + [np.zeros_like(w) for w in self.w[1:]]
        d_theta_batch = [None] + [np.zeros_like(t) for t in self.theta[1:]]
        
        # Accumulate gradients for each pattern in batch
        for x, z in zip(batch_X, batch_y):
            # Forward pass
            y = self.feed_forward(x)
            
            # Equation (11): Output layer deltas
            self.delta[self.L-1] = self.activation(self.h[self.L-1], derivative=True) * (y - z)
            
            # Equation (12): Hidden layer deltas
            for l in range(self.L-2, 0, -1):
                self.delta[l] = self.activation(self.h[l], derivative=True) * \
                               np.dot(self.w[l+1].T, self.delta[l+1])
            
            # Accumulate gradients
            for l in range(1, self.L):
                d_w_batch[l] += -self.delta[l][:, np.newaxis] @ self.xi[l-1][np.newaxis, :]
                d_theta_batch[l] += self.delta[l]
        
        # Apply updates with momentum (equation 16)
        for l in range(1, self.L):
            # Weight updates
            d_w = self.learning_rate * d_w_batch[l] + self.momentum * self.d_w_prev[l]
            self.w[l] += d_w
            self.d_w_prev[l] = d_w
            
            # Threshold updates
            d_theta = self.learning_rate * d_theta_batch[l] + self.momentum * self.d_theta_prev[l]
            self.theta[l] += d_theta
            self.d_theta_prev[l] = d_theta
    
    def calculate_error(self, X, y):
        """Implementation of equation (5) from document"""
        total_error = 0
        for x, z in zip(X, y):
            y_pred = self.feed_forward(x)
            total_error += np.sum((y_pred - z)**2)
        return total_error / 2  # No division by len(X) to match equation (5)

    def fit(self, X, y):
        """Train the neural network using partial batched BP"""
        if self.validation_split > 0:
            split_idx = int(len(X) * (1 - self.validation_split))
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        n_samples = len(X_train)
        
        for epoch in range(self.num_epochs):
            # Randomly shuffle patterns at each epoch
            indices = np.random.permutation(n_samples)
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            # Process mini-batches
            for i in range(0, n_samples, self.batch_size):
                batch_X = X_train[i:i+self.batch_size]
                batch_y = y_train[i:i+self.batch_size]
                self.update_weights_batch(batch_X, batch_y)
            
            # Calculate errors (total, not averaged)
            train_error = self.calculate_error(X_train, y_train)
            self.train_errors.append(train_error)
            
            if X_val is not None:
                val_error = self.calculate_error(X_val, y_val)
                self.val_errors.append(val_error)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.num_epochs}, Total Error: {train_error:.6f}")
        
        return self
    
    def predict(self, X):
        """Make predictions for input X"""
        return np.array([self.feed_forward(x) for x in X])
    
    def loss_epochs(self):
        """Return training and validation error history"""
        return np.array(self.train_errors), np.array(self.val_errors)


def scale_data(X, feature_range=(0.1, 0.9)):
    """Equation (1): Scale features to range [smin, smax]"""
    smin, smax = feature_range
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return smin + (smax - smin) * (X - X_min) / (X_max - X_min), (X_min, X_max)

def inverse_scale(X_scaled, X_min, X_max, feature_range=(0.1, 0.9)):
    """Equation (2): Inverse scaling of data"""
    smin, smax = feature_range
    return X_min + (X_max - X_min) * (X_scaled - smin) / (smax - smin)

def test_configurations(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_train, y_test, y_min, y_max):
    """Test different neural network configurations"""
    # Define configurations to test
    configs = [
    # 3-layer configurations
    {
        'layers': [13, 26, 1],
        'epochs': 1000,
        'lr': 0.001,
        'momentum': 0.8,
        'activation': 'tanh',
        'description': '3-layer tanh, wide'
    },
    {
        'layers': [13, 20, 1],
        'epochs': 1200,
        'lr': 0.015,
        'momentum': 0.85,
        'activation': 'relu',
        'description': '3-layer ReLU, moderate'
    },
    {
        'layers': [13, 32, 1],
        'epochs': 1000,
        'lr': 0.02,
        'momentum': 0.75,
        'activation': 'sigmoid',
        'description': '3-layer sigmoid, extra wide'
    },
    
    # 4-layer configurations
    {
        'layers': [13, 26, 13, 1],
        'epochs': 1200,
        'lr': 0.001,
        'momentum': 0.8,
        'activation': 'tanh',
        'description': '4-layer tanh, symmetric'
    },
    {
        'layers': [13, 20, 10, 1],
        'epochs': 1500,
        'lr': 0.015,
        'momentum': 0.85,
        'activation': 'relu',
        'description': '4-layer ReLU, gradual reduction'
    },
    {
        'layers': [13, 24, 16, 1],
        'epochs': 1300,
        'lr': 0.005,
        'momentum': 0.9,
        'activation': 'linear',
        'description': '4-layer linear, moderate'
    },
    {
        'layers': [13, 30, 15, 1],
        'epochs': 1400,
        'lr': 0.001,
        'momentum': 0.82,
        'activation': 'sigmoid',
        'description': '4-layer sigmoid, wide start'
    },
    
    # 5-layer configurations
    {
        'layers': [13, 26, 20, 13, 1],
        'epochs': 1500,
        'lr': 0.005,
        'momentum': 0.9,
        'activation': 'tanh',
        'description': '5-layer tanh, gradual reduction'
    },
    {
        'layers': [13, 24, 18, 12, 1],
        'epochs': 1600,
        'lr': 0.01,
        'momentum': 0.85,
        'activation': 'relu',
        'description': '5-layer ReLU, careful reduction'
    },
    {
        'layers': [13, 28, 21, 14, 1],
        'epochs': 1700,
        'lr': 0.001,
        'momentum': 0.88,
        'activation': 'sigmoid',
        'description': '5-layer sigmoid, wide reduction'
    }
    ]
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\nTesting Configuration {i}/10:")
        print(f"Description: {config['description']}")
        print(f"Architecture: {config['layers']}")
        print(f"Learning Rate: {config['lr']}, Momentum: {config['momentum']}")
        print(f"Activation: {config['activation']}, Epochs: {config['epochs']}")
        
        model = NeuralNet(
            num_layers=len(config['layers']),
            units_per_layer=config['layers'],
            num_epochs=config['epochs'],
            learning_rate=config['lr'],
            momentum=config['momentum'],
            activation=config['activation'],
            validation_split=0.2
        )
        
        # Train model
        model.fit(X_train_scaled, y_train_scaled)
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Inverse scale predictions
        y_pred_train = inverse_scale(y_pred_train.reshape(-1, 1), y_min, y_max).ravel()
        y_pred_test = inverse_scale(y_pred_test.reshape(-1, 1), y_min, y_max).ravel()
        
        # Calculate metrics
        mse_train = mean_squared_error(y_train, y_pred_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
        
        mse_test = mean_squared_error(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
        
        # Store results
        results.append({
            'config': config['description'],
            'train_mse': mse_train,
            'train_mae': mae_train,
            'train_mape': mape_train,
            'test_mse': mse_test,
            'test_mae': mae_test,
            'test_mape': mape_test,
            'train_error': model.train_errors[-1],
            'val_error': model.val_errors[-1]
        })
        
        # Print metrics
        print("\nTraining Metrics:")
        print(f"MSE: {mse_train:.4f}")
        print(f"MAE: {mae_train:.4f}")
        print(f"MAPE: {mape_train:.4f}")
        
        print("\nTest Metrics:")
        print(f"MSE: {mse_test:.4f}")
        print(f"MAE: {mae_test:.4f}")
        print(f"MAPE: {mape_test:.4f}")
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(model.train_errors, label='Training Error')
        plt.plot(model.val_errors, label='Validation Error')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.title(f'Training History - Configuration {i}\n{config["description"]}')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Plot predictions vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_test, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted Value')
        plt.title(f'Predictions vs Actual - Configuration {i}\n{config["description"]}')
        plt.tight_layout()
        plt.show()
    
    # Print summary table
    print("\nSummary of all configurations:")
    print("\nTest Metrics:")
    print("Configuration | Test MSE | Test MAE | Test MAPE")
    print("-" * 60)
    for r in results:
        print(f"{r['config'][:20]:<20} | {r['test_mse']:.4f} | {r['test_mae']:.4f} | {r['test_mape']:.4f}")
    
    return results

def main():
    try:
        # Load data
        train_data = pd.read_csv('traindata.csv')
        test_data = pd.read_csv('testdata.csv')
        
        # Assuming 'Life expectancy' is the target variable
        X_train = train_data.drop('Life expectancy ', axis=1).values
        y_train = train_data['Life expectancy '].values
        X_test = test_data.drop('Life expectancy ', axis=1).values
        y_test = test_data['Life expectancy '].values

        # Scaling input data
        X_train_scaled, (X_min, X_max) = scale_data(X_train)
        X_test_scaled = scale_data(X_test, feature_range=(0.1, 0.9))[0]

        # Scaling target data (y values)
        y_train_scaled, (y_min, y_max) = scale_data(y_train.reshape(-1, 1))
        y_test_scaled = scale_data(y_test.reshape(-1, 1), feature_range=(0.1, 0.9))[0]
        
        # Convert back to 1D arrays
        y_train_scaled = y_train_scaled.ravel()
        y_test_scaled = y_test_scaled.ravel()
        
        # Test different configurations
        results = test_configurations(
            X_train_scaled, y_train_scaled,
            X_test_scaled, y_test_scaled,
            y_train, y_test,
            y_min, y_max
        )
        
        # Find best configuration based on test MSE
        best_config = min(results, key=lambda x: x['test_mse'])
        print("\nBest Configuration:")
        print(f"Configuration: {best_config['config']}")
        print(f"Test MSE: {best_config['test_mse']:.4f}")
        print(f"Test MAE: {best_config['test_mae']:.4f}")
        print(f"Test MAPE: {best_config['test_mape']:.4f}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
