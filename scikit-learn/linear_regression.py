import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

X, y = fetch_california_housing(return_X_y=True)
data = fetch_california_housing(as_frame=True).frame

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
output = model.predict(X_test_scaled)

r_two = r2_score(y_test, output)
mse = mean_squared_error(y_test, output)
mae = mean_absolute_error(y_test, output)
rmse = root_mean_squared_error(y_test, output)

print(r_two, mse, mae, rmse)
print(y_test.min(), y_test.max())
print(model.score(X_test_scaled, y_test))

plt.scatter(y_test, output, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)

plt.show()


