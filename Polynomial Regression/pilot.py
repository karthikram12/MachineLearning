import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from numpy import random
from sklearn.linear_model import LinearRegression

X = 4 * random.rand(100, 1) - 2
y = 5 + 4 * X + 3 * X ** 2 + 2 * random.randn(100, 1)

poly_features = PolynomialFeatures(degree=2, include_bias=False)

X_poly = poly_features.fit_transform(X)
print(X_poly)

model = LinearRegression()
model.fit(X_poly, y)

X_vals = np.linspace(-2, 2, 100).reshape(-1, 1)
X_vals_poly = poly_features.transform(X_vals)
y_vals = model.predict(X_vals_poly)

plt.scatter(X, y, color='red')
plt.plot(X_vals, y_vals, color='blue' )

plt.show()
