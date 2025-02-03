import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

class NeuralNet:
    def __init__(self, num_layers, units_per_layer, num_epochs=1000, learning_rate=0.01, 
                 momentum=0.9, activation='sigmoid', validation_split=0.2):
        self.L = num_layers
        self.n = units_per_layer
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.fact = activation
        self.validation_split = validation_split
        
        # Initialize arrays following equations (7)-(9)
        self.h = [np.zeros(n) for n in self.n]  # Fields h(l)_i
        self.xi = [np.zeros(n) for n in self.n]  # Activations ξ(l)_i
        
        # Initialize weights with small random values
        self.w = [None]  # w[1] is not used as per document notation
        for l in range(1, self.L):
            self.w.append(np.random.randn(self.n[l], self.n[l-1]) * np.sqrt(2.0/self.n[l-1]))
            
        # Initialize thresholds
        self.theta = [None]  # theta[1] is not used
        for l in range(1, self.L):
            self.theta.append(np.random.randn(self.n[l]) * 0.1)
        
        # Arrays for backpropagation equations (11)-(12)
        self.delta = [np.zeros(n) for n in self.n]  # Δ(l)_i
        
        # Arrays for weight and threshold updates equation (14)
        self.d_w = [None] + [np.zeros((self.n[l], self.n[l-1])) for l in range(1, self.L)]
        self.d_theta = [None] + [np.zeros(self.n[l]) for l in range(1, self.L)]
        
        # Arrays for momentum
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
        else:  # linear
            if not derivative:
                return h
            else:
                return 1
    
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
    
    def process_mini_batch(self, X_batch, y_batch):
        """Implementation of equation (16) for partial batched BP"""
        # Initialize accumulators for equation (16)
        delta_w = [None] + [np.zeros_like(w) for w in self.w[1:]]
        delta_theta = [None] + [np.zeros_like(theta) for theta in self.theta[1:]]
        
        # For each pattern in batch P
        for x, z in zip(X_batch, y_batch):
            # Forward pass
            y = self.feed_forward(x)
            
            # Equation (11): Output layer deltas
            self.delta[self.L-1] = self.activation(self.h[self.L-1], derivative=True) * (y - z)
            
            # Equation (12): Hidden layer deltas
            for l in range(self.L-2, 0, -1):
                self.delta[l] = self.activation(self.h[l], derivative=True) * \
                               np.dot(self.w[l+1].T, self.delta[l+1])
            
            # Accumulate gradients for batch update (equation 16)
            for l in range(1, self.L):
                delta_w[l] += np.outer(self.delta[l], self.xi[l-1])
                delta_theta[l] += self.delta[l]
        
        # Update weights and thresholds (equation 16)
        for l in range(1, self.L):
            self.d_w[l] = -self.learning_rate * delta_w[l] + self.momentum * self.d_w_prev[l]
            self.w[l] += self.d_w[l]
            self.d_w_prev[l] = self.d_w[l].copy()
            
            self.d_theta[l] = self.learning_rate * delta_theta[l] + \
                             self.momentum * self.d_theta_prev[l]
            self.theta[l] += self.d_theta[l]
            self.d_theta_prev[l] = self.d_theta[l].copy()
    
    def calculate_error(self, X, y):
        """Implementation of equation (5)"""
        total_error = 0
        for x, z in zip(X, y):
            y_pred = self.feed_forward(x)
            total_error += np.sum((y_pred - z)**2)
        return total_error / (2 * len(X))
    
    def fit(self, X, y):
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
        
        batch_size = 5  # As specified in the document
        
        for epoch in range(self.num_epochs):
            indices = np.random.permutation(len(X_train))
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                self.process_mini_batch(X_batch, y_batch)
            
            train_error = self.calculate_error(X_train, y_train)
            self.train_errors.append(train_error)
            
            if X_val is not None:
                val_error = self.calculate_error(X_val, y_val)
                self.val_errors.append(val_error)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.num_epochs}, Error: {train_error:.6f}")
    
    def predict(self, X):
        return np.array([self.feed_forward(x) for x in X])
    
    def loss_epochs(self):
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

def main():
    try:
        # Load data
        train_data = pd.read_csv('traindata.csv')
        test_data = pd.read_csv('testdata.csv')
        
        X_train = train_data.drop('Life expectancy ', axis=1).values
        y_train = train_data['Life expectancy '].values
        X_test = test_data.drop('Life expectancy ', axis=1).values
        y_test = test_data['Life expectancy '].values
        
        # Scale data using equations (1)-(2)
        X_train_scaled, (X_min, X_max) = scale_data(X_train)
        X_test_scaled = scale_data(X_test)[0]
        y_train_scaled, (y_min, y_max) = scale_data(y_train.reshape(-1, 1))
        y_test_scaled = scale_data(y_test.reshape(-1, 1))[0]
        
        y_train_scaled = y_train_scaled.ravel()
        y_test_scaled = y_test_scaled.ravel()
        
        # Create and train model
        input_size = X_train.shape[1]
        architecture = [input_size, 32, 16, 1]
        
        model = NeuralNet(
            num_layers=len(architecture),
            units_per_layer=architecture,
            num_epochs=1000,
            learning_rate=0.01,
            momentum=0.9,
            activation='sigmoid', 
            validation_split=0.2
        )
        
        model.fit(X_train_scaled, y_train_scaled)
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Inverse scale predictions
        y_pred_train = inverse_scale(y_pred_train.reshape(-1, 1), y_min, y_max).ravel()
        y_pred_test = inverse_scale(y_pred_test.reshape(-1, 1), y_min, y_max).ravel()
        
        # Calculate metrics
        print("\nTraining Metrics:")
        mse_train = mean_squared_error(y_train, y_pred_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
        print(f"MSE: {mse_train:.4f}")
        print(f"MAE: {mae_train:.4f}")
        print(f"MAPE: {mape_train:.4f}")
        
        print("\nTest Metrics:")
        mse_test = mean_squared_error(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
        print(f"MSE: {mse_test:.4f}")
        print(f"MAE: {mae_test:.4f}")
        print(f"MAPE: {mape_test:.4f}")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_test, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Life Expectancy')
        plt.ylabel('Predicted Life Expectancy')
        plt.title('Life Expectancy: Predicted vs Actual')
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(10, 6))
        plt.plot(model.train_errors, label='Training Error')
        plt.plot(model.val_errors, label='Validation Error')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.title('Training History')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
