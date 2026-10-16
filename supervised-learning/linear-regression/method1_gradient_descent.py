"""
Method 1: Linear Regression with Gradient Descent (from scratch)
"""

import numpy as np

# Training data
X = np.array([1, 2, 3, 4, 5], dtype=float)
Y = np.array([2, 3, 4, 5, 6], dtype=float)

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# Start with slope (m) and intercept (b) as 0
m = 0.0
b = 0.0
n = len(X)

# Gradient Descent Loop
for i in range(epochs):
    # Predict Y
    Y_pred = m * X + b

    # Mean Squared Error (Cost)
    cost = ((Y_pred - Y) ** 2).mean()

    # Derivatives
    derivative_m = (-2 / n) * np.sum(X * (Y - Y_pred))
    derivative_b = (-2 / n) * np.sum(Y - Y_pred)

    # Update parameters
    m -= learning_rate * derivative_m
    b -= learning_rate * derivative_b

    # Show progress every 100 epochs
    if (i + 1) % 100 == 0:
        print(f"Epoch {i + 1}/{epochs}, Cost: {cost:.4f}, m: {m:.4f}, b: {b:.4f}")

print(f"\nFinal learned line: y = {m:.4f} * x + {b:.4f}")

# Predictions for new values
new_x = 6
prediction = m * new_x + b
print(f"Prediction for x={new_x}: {prediction:.4f}")

new_x1 = 7
prediction1 = m * new_x1 + b
print(f"Prediction for x={new_x1}: {prediction1:.4f}")
