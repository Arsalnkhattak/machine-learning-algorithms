# Linear Regression

Linear regression learns a line that predicts a numeric target:

```text
y = m * x + b
```

`m` is the slope and `b` is the intercept. This folder demonstrates the same
small example in two separate Python scripts: one implements gradient descent
with NumPy, and the other trains scikit-learn's `LinearRegression` model and
plots its fitted line with Matplotlib.

## Files

- `method1_gradient_descent.py`: trains the line with a gradient-descent loop.
- `method2_sklearn.py`: trains with scikit-learn and displays a plot.
- `linear_regression_gradient_descent.ipynb`: notebook with both approaches
  split into separate cells, so they can be studied independently.
- `linear_regression_combined_demo.py`: preserved combined example that runs
  both methods in one script. Use one of the `method*.py` scripts to run only
  one method.
- `requirements.txt`: Python packages used by the examples.

## Run the scripts

Open PowerShell in this folder, then install the dependencies:

```powershell
python -m pip install -r .\requirements.txt
```

Run either method separately:

```powershell
python .\method1_gradient_descent.py
```

```powershell
python .\method2_sklearn.py
```

The scikit-learn script opens a Matplotlib plot. To open the notebook instead:

```powershell
jupyter notebook .\linear_regression_gradient_descent.ipynb
```

## Training and prediction

The example data in each script is used for training. The predictions for
`x=6` and `x=7` demonstrate using the fitted model on new inputs; they are not
test-set evaluation because no actual target values are provided for those
inputs.
