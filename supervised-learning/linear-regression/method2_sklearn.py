"""
Method 2: Linear Regression using Scikit-Learn with Matplotlib Best Fit Line
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Training data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([2, 3, 4, 5, 6])

# Train Scikit-Learn Model
model = LinearRegression()
model.fit(X, Y)

# Get slope (m) and intercept (b)
m = model.coef_[0]
b = model.intercept_
print(f"Slope (m): {m:.4f}")
print(f"Intercept (b): {b:.4f}")
print(f"Learned line: y = {m:.4f} * x + {b:.4f}")

# Predictions for x=6 and x=7
new_x = np.array([[6], [7]])
predictions = model.predict(new_x)
print(f"Prediction for x=6: {predictions[0]:.4f}")
print(f"Prediction for x=7: {predictions[1]:.4f}")

# Plot best fit line for Scikit-Learn only
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label=f'Best Fit Line: y = {m:.2f}x + {b:.2f}')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression (Scikit-Learn)')
plt.legend()
plt.grid(True)
plt.savefig('sklearn_best_fit_line.png')
print("\nPlot saved as 'sklearn_best_fit_line.png'")
plt.show()
