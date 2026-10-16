"""
Linear Regression: Gradient Descent (from scratch) & Scikit-Learn
Simple implementation of both methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. DATA
# ==========================================
X = np.array([1, 2, 3, 4, 5], dtype=float)
Y = np.array([2, 3, 4, 5, 6], dtype=float)

# ==========================================
# 2. METHOD 1: Gradient Descent (From Scratch)
# ==========================================
print("--- METHOD 1: Gradient Descent ---")
learning_rate = 0.01
epochs = 1000

m = 0.0
b = 0.0
n = len(X)

for i in range(epochs):
    Y_pred = m * X + b
    cost = ((Y_pred - Y) ** 2).mean()

    derivative_m = (-2 / n) * np.sum(X * (Y - Y_pred))
    derivative_b = (-2 / n) * np.sum(Y - Y_pred)

    m -= learning_rate * derivative_m
    b -= learning_rate * derivative_b

    if (i + 1) % 200 == 0:
        print(f"Epoch {i + 1}/{epochs}, Cost: {cost:.4f}, m: {m:.4f}, b: {b:.4f}")

print(f"Gradient Descent line: y = {m:.4f} * x + {b:.4f}")
print(f"Prediction for x=6: {m * 6 + b:.4f}")
print(f"Prediction for x=7: {m * 7 + b:.4f}")

# ==========================================
# 3. METHOD 2: Scikit-Learn
# ==========================================
print("\n--- METHOD 2: Scikit-Learn ---")
X_reshaped = X.reshape(-1, 1)
model = LinearRegression()
model.fit(X_reshaped, Y)

m_sk = model.coef_[0]
b_sk = model.intercept_
print(f"Scikit-Learn line: y = {m_sk:.4f} * x + {b_sk:.4f}")
print(f"Prediction for x=6: {model.predict([[6]])[0]:.4f}")
print(f"Prediction for x=7: {model.predict([[7]])[0]:.4f}")

# ==========================================
# 4. PLOT: Best Fit Line (Scikit-Learn Only)
# ==========================================
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X_reshaped), color='red', label=f'Best Fit Line: y = {m_sk:.2f}x + {b_sk:.2f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression Best Fit Line (Scikit-Learn)')
plt.legend()
plt.grid(True)
plt.savefig('best_fit_line.png')
print("\nPlot saved as 'best_fit_line.png'")
plt.show()
