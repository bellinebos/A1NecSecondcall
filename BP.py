# Import required libraries
import numpy as np  # For numerical computations and array operations
import pandas as pd  # For data manipulation and reading CSV files
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error  # For evaluation metrics
import matplotlib.pyplot as plt  # For plotting results

class NeuralNet:
    def __init__(self, num_layers, units_per_layer, num_epochs=1000, learning_rate=0.01, 
                 momentum=0.9, activation='sigmoid', validation_split=0.2, batch_size=5):
        #Initialize the Neural Network with specified architecture and hyperparameters
        self.L = num_layers  # Total number of layers
        self.n = units_per_layer  # Units in each layer
        self.num_epochs = num_epochs  # Training epochs
        self.learning_rate = learning_rate  # Learning rate η
        self.momentum = momentum  # Momentum coefficient α
        self.fact = activation  # Activation function type
        self.validation_split = validation_split  # Validation data ratio
        self.batch_size = batch_size  # Mini-batch size
        
        # Initialize neuron activations and fields
        self.h = [np.zeros(n) for n in self.n]  # h(l)_i: Input fields for each layer
        self.xi = [np.zeros(n) for n in self.n]  # ξ(l)_i: Neuron activations
        
        # Initialize weights
        self.w = [None]  # w[0] not used (1-based indexing)
        for l in range(1, self.L):
            # initialization scales weights based on input size
            self.w.append(np.random.randn(self.n[l], self.n[l-1]) * np.sqrt(2.0/self.n[l-1]))
        
        # Initialize bias terms (thresholds)
        self.theta = [None]  # theta[0] not used
        for l in range(1, self.L):
            self.theta.append(np.random.randn(self.n[l]) * 0.1)  # Small random initial biases
        
        # Initialize backpropagation variables
        self.delta = [np.zeros(n) for n in self.n]  # Error terms for each layer
        
        # Initialize momentum terms
        self.d_w_prev = [None] + [np.zeros((self.n[l], self.n[l-1])) for l in range(1, self.L)]  # Previous weight changes
        self.d_theta_prev = [None] + [np.zeros(self.n[l]) for l in range(1, self.L)]  # Previous bias changes
        
        # Initialize error tracking
        self.train_errors = []  # Store training errors
        self.val_errors = []  # Store validation errors
    
    def activation(self, h, derivative=False):
        """
        Compute activation function or its derivative
        
        Parameters:
        h: Input value
        derivative: If True, compute derivative instead of function value
        """
        if self.fact == 'sigmoid':
            if not derivative:
                return 1 / (1 + np.exp(-h))  # Sigmoid function
            else:
                g_h = self.activation(h)
                return g_h * (1 - g_h)  # Derivative of sigmoid
        elif self.fact == 'relu':
            if not derivative:
                return np.maximum(0, h)  # ReLU function
            else:
                return (h > 0).astype(float)  # Derivative of ReLU
        elif self.fact == 'tanh':
            if not derivative:
                return np.tanh(h)  # Hyperbolic tangent
            else:
                return 1 - np.tanh(h)**2  # Derivative of tanh
        elif self.fact == 'linear':
            if not derivative:
                return h  # Linear function
            else:
                return 1  # Derivative of linear function
        else:
            raise ValueError("Invalid activation function specified.")
    
    def feed_forward(self, x):
        """
        Perform forward pass through the network
        
        Parameters:
        x: Input pattern
        """
        # Set input layer activations
        self.xi[0] = x
        
        # Propagate through hidden layers to output
        for l in range(1, self.L):
            # Compute weighted sum and subtract bias
            self.h[l] = np.dot(self.w[l], self.xi[l-1]) - self.theta[l]
            # Apply activation function
            self.xi[l] = self.activation(self.h[l])
        
        # Return output layer activations
        return self.xi[self.L-1]

    def update_weights_batch(self, batch_X, batch_y):
        """
        Update network weights using mini-batch gradient descent
        
        Parameters:
        batch_X: Input patterns in current batch
        batch_y: Target outputs for current batch
        """
        # Initialize gradient accumulators for batch
        d_w_batch = [None] + [np.zeros_like(w) for w in self.w[1:]]
        d_theta_batch = [None] + [np.zeros_like(t) for t in self.theta[1:]]
        
        # Process each pattern in batch
        for x, z in zip(batch_X, batch_y):
            # Forward pass
            y = self.feed_forward(x)
            
            # Compute output layer error
            self.delta[self.L-1] = self.activation(self.h[self.L-1], derivative=True) * (y - z)
            
            # Backpropagate error
            for l in range(self.L-2, 0, -1):
                self.delta[l] = self.activation(self.h[l], derivative=True) * \
                               np.dot(self.w[l+1].T, self.delta[l+1])
            
            # Accumulate gradients
            for l in range(1, self.L):
                # Weight gradients
                d_w_batch[l] += -self.delta[l][:, np.newaxis] @ self.xi[l-1][np.newaxis, :]
                # Bias gradients
                d_theta_batch[l] += self.delta[l]
        
        # Update weights and biases with momentum
        for l in range(1, self.L):
            # Update weights
            d_w = self.learning_rate * d_w_batch[l] + self.momentum * self.d_w_prev[l]
            self.w[l] += d_w
            self.d_w_prev[l] = d_w
            
            # Update biases
            d_theta = self.learning_rate * d_theta_batch[l] + self.momentum * self.d_theta_prev[l]
            self.theta[l] += d_theta
            self.d_theta_prev[l] = d_theta
    
    def calculate_error(self, X, y):
        """
        Calculate total error for a set of patterns
        
        Parameters:
        X: Input patterns
        y: Target outputs
        """
        total_error = 0
        # Process each pattern
        for x, z in zip(X, y):
            # Get network output
            y_pred = self.feed_forward(x)
            # Add squared error to total
            total_error += np.sum((y_pred - z)**2)
        # Return total error (not averaged)
        return total_error / 2

    def fit(self, X, y):
        """
        Train the neural network
        
        Parameters:
        X: Training input patterns
        y: Training target outputs
        """
        # Split data into training and validation sets if specified
        if self.validation_split > 0:
            # Calculate split index
            split_idx = int(len(X) * (1 - self.validation_split))
            # Randomly shuffle data
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
            # Perform split
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            # Use all data for training if no validation
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        n_samples = len(X_train)
        
        # Training loop for specified number of epochs
        for epoch in range(self.num_epochs):
            # Shuffle training data at start of each epoch
            indices = np.random.permutation(n_samples)
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            # Process mini-batches
            for i in range(0, n_samples, self.batch_size):
                # Extract current batch
                batch_X = X_train[i:i+self.batch_size]
                batch_y = y_train[i:i+self.batch_size]
                # Update weights using current batch
                self.update_weights_batch(batch_X, batch_y)
            
            # Calculate and store training error
            train_error = self.calculate_error(X_train, y_train)
            self.train_errors.append(train_error)
            
            # Calculate and store validation error if using validation set
            if X_val is not None:
                val_error = self.calculate_error(X_val, y_val)
                self.val_errors.append(val_error)
            
            # Print progress every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.num_epochs}, Total Error: {train_error:.6f}")
        
        return self

    def predict(self, X):
        """
        Make predictions for new input patterns
        
        Parameters:
        X: Input patterns to predict
        """
        # Return predictions for all input patterns
        return np.array([self.feed_forward(x) for x in X])

def scale_data(X, feature_range=(0.1, 0.9)):
    """
    Scale features to specified range
    
    Parameters:
    X: Input data
    feature_range: Target range for scaled data
    """
    # Extract range bounds
    smin, smax = feature_range
    # Get data bounds
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    # Scale data to target range
    return smin + (smax - smin) * (X - X_min) / (X_max - X_min), (X_min, X_max)

def inverse_scale(X_scaled, X_min, X_max, feature_range=(0.1, 0.9)):
    """
    Reverse scaling transformation
    
    Parameters:
    X_scaled: Scaled data
    X_min, X_max: Original data bounds
    feature_range: Range used for scaling
    """
    # Extract range bounds
    smin, smax = feature_range
    # Reverse scaling transformation
    return X_min + (X_max - X_min) * (X_scaled - smin) / (smax - smin)

def test_configurations(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_train, y_test, y_min, y_max):
    """
    Test different neural network architectures and configurations
    
    Parameters:
    X_train_scaled, y_train_scaled: Scaled training data
    X_test_scaled, y_test_scaled: Scaled test data
    y_train, y_test: Original (unscaled) target values
    y_min, y_max: Target value bounds for inverse scaling
    """
    # Define different network configurations to test
    configs = [
        # 3-layer configurations
        {
        'layers': [13, 26, 1],  # Input layer, hidden layer, output layer
        'epochs': 1000,
        'lr': 0.001,
        'momentum': 0.8,
        'activation': 'tanh',
        'description': '3-layer tanh'
        },
    {
        'layers': [13, 20, 1],
        'epochs': 1200,
        'lr': 0.015,
        'momentum': 0.85,
        'activation': 'relu',
        'description': '3-layer ReLU'
    },
    {
        'layers': [13, 32, 1],
        'epochs': 1000,
        'lr': 0.02,
        'momentum': 0.75,
        'activation': 'sigmoid',
        'description': '3-layer sigmoid'
    },
    
    # 4-layer configurations
    {
        'layers': [13, 26, 13, 1],
        'epochs': 1200,
        'lr': 0.001,
        'momentum': 0.8,
        'activation': 'tanh',
        'description': '4-layer tanh'
    },
    {
        'layers': [13, 20, 10, 1],
        'epochs': 1500,
        'lr': 0.015,
        'momentum': 0.85,
        'activation': 'relu',
        'description': '4-layer ReLU'
    },
    {
        'layers': [13, 24, 16, 1],
        'epochs': 1300,
        'lr': 0.005,
        'momentum': 0.9,
        'activation': 'linear',
        'description': '4-layer linear'
    },
    {
        'layers': [13, 30, 15, 1],
        'epochs': 1400,
        'lr': 0.001,
        'momentum': 0.82,
        'activation': 'sigmoid',
        'description': '4-layer sigmoid'
    },
    
    # 5-layer configurations
    {
        'layers': [13, 26, 20, 13, 1],
        'epochs': 1500,
        'lr': 0.005,
        'momentum': 0.9,
        'activation': 'tanh',
        'description': '5-layer tanh'
    },
    {
        'layers': [13, 24, 18, 12, 1],
        'epochs': 1600,
        'lr': 0.01,
        'momentum': 0.85,
        'activation': 'relu',
        'description': '5-layer ReLU'
    },
    {
        'layers': [13, 28, 21, 14, 1],
        'epochs': 1700,
        'lr': 0.001,
        'momentum': 0.88,
        'activation': 'sigmoid',
        'description': '5-layer sigmoid'
    }
    ]
    
    results = []  # Store results for each configuration
    
    # Test each configuration
    for i, config in enumerate(configs, 1):
        # Print configuration details
        print(f"\nTesting Configuration {i}/10:")
        print(f"Description: {config['description']}")
        print(f"Architecture: {config['layers']}")
        print(f"Learning Rate: {config['lr']}, Momentum: {config['momentum']}")
        print(f"Activation: {config['activation']}, Epochs: {config['epochs']}")
        
        # Create and configure model
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
        
        # Inverse scale predictions back to original range
        y_pred_train = inverse_scale(y_pred_train.reshape(-1, 1), y_min, y_max).ravel()
        y_pred_test = inverse_scale(y_pred_test.reshape(-1, 1), y_min, y_max).ravel()
        
        # Calculate performance metrics
        mse_train = mean_squared_error(y_train, y_pred_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
        
        mse_test = mean_squared_error(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
        
        # Store results for this configuration
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
        
        # Print performance metrics
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
        
        # Plot predictions vs actual values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_test, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted Value')
        plt.title(f'Predictions vs Actual - Configuration {i}\n{config["description"]}')
        plt.tight_layout()
        plt.show()
    
    # Print summary table of all configurations
    print("\nSummary of all configurations:")
    print("\nTest Metrics:")
    print("Configuration | Test MSE | Test MAE | Test MAPE")
    print("-" * 60)
    for r in results:
        print(f"{r['config'][:20]:<20} | {r['test_mse']:.4f} | {r['test_mae']:.4f} | {r['test_mape']:.4f}")
    
    return results

def main():
    """Main function to run the neural network experiments"""
    try:
        # Load training and test data from CSV files
        train_data = pd.read_csv('traindata.csv')
        test_data = pd.read_csv('testdata.csv')
        
        # Separate features and target variable
        X_train = train_data.drop('Life expectancy ', axis=1).values
        y_train = train_data['Life expectancy '].values
        X_test = test_data.drop('Life expectancy ', axis=1).values
        y_test = test_data['Life expectancy '].values

        # Scale input features
        X_train_scaled, (X_min, X_max) = scale_data(X_train)
        X_test_scaled = scale_data(X_test, feature_range=(0.1, 0.9))[0]

        # Scale target values
        y_train_scaled, (y_min, y_max) = scale_data(y_train.reshape(-1, 1))
        y_test_scaled = scale_data(y_test.reshape(-1, 1), feature_range=(0.1, 0.9))[0]
        
        # Convert target arrays to 1D
        y_train_scaled = y_train_scaled.ravel()
        y_test_scaled = y_test_scaled.ravel()
        
        # Test different neural network configurations
        results = test_configurations(
            X_train_scaled, y_train_scaled,
            X_test_scaled, y_test_scaled,
            y_train, y_test,
            y_min, y_max
        )
        
        # Find and print best configuration based on test MSE
        best_config = min(results, key=lambda x: x['test_mse'])
        print("\nBest Configuration:")
        print(f"Configuration: {best_config['config']}")
        print(f"Test MSE: {best_config['test_mse']:.4f}")
        print(f"Test MAE: {best_config['test_mae']:.4f}")
        print(f"Test MAPE: {best_config['test_mape']:.4f}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Entry point of the program
if __name__ == "__main__":
    main()
