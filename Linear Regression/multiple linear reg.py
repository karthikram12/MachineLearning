import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('CarPrice_Assignment.csv')
data = pd.DataFrame(data)
columns_to_drop = ['car_ID','CarName']
data = data.drop(columns=columns_to_drop)

columns_to_ohe = ['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'fuelsystem']
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_values = ohe.fit_transform(data[columns_to_ohe])
encoded_column_names = ohe.get_feature_names_out(columns_to_ohe)
encoded_df = pd.DataFrame(encoded_values, columns=encoded_column_names)

data = data.drop(columns=columns_to_ohe)
data = pd.concat([data, encoded_df], axis=1)

doornumber_mapping = {"two": 2, "four": 4}
cylindernumber_mapping = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8}
data['doornumber'] = data['doornumber'].map(doornumber_mapping)
data['cylindernumber'] = data['cylindernumber'].map(cylindernumber_mapping)

data.dropna(inplace=True)

corr_with_target = data.corr()['price'].sort_values(ascending=False)
feature_columns = corr_with_target[1:11].index

X = data[feature_columns]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linear = LinearRegression()
linear.fit(X_train_scaled, y_train)
y_pred = linear.predict(X_test_scaled)

mse_linear = mean_squared_error(y_test, y_pred)
mae_linear = mean_absolute_error(y_test, y_pred)
rmse_linear = root_mean_squared_error(y_test, y_pred)
print(f"Mean square error is {mse_linear} and Mean Absolute Error is {mae_linear} and Root Mean square error is {rmse_linear}")
print("Linear Reg R2(train) score:", linear.score(X_train_scaled, y_train))
print("Linear Reg R2(test) score:", linear.score(X_test_scaled, y_test))

plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue')

ridge = Ridge(alpha=1, max_iter=1000)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = root_mean_squared_error(y_test, y_pred_ridge)
print(f"Mean square error is {mse_ridge} and Mean Absolute Error is {mae_ridge} and Root Mean square error is {rmse_ridge}")

print("Ridge Reg R2(train) score:", ridge.score(X_train_scaled, y_train))
print("Ridge Reg R2(test) score:", ridge.score(X_test_scaled, y_test))

plt.subplot(2, 2, 2)
plt.scatter(y_test, y_pred_ridge, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue')

lasso = Lasso(alpha=1, max_iter=1000)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
rmse_lasso = root_mean_squared_error(y_test, y_pred_lasso)
print(f"Mean square error is {mse_lasso} and Mean Absolute Error is {mae_lasso} and Root Mean square error is {rmse_lasso}")

print("Lasso Reg R2(train) score:", lasso.score(X_train_scaled, y_train))
print("Lasso Reg R2(test) score:", lasso.score(X_test_scaled, y_test))

plt.subplot(2, 2, 3)
plt.scatter(y_test, y_pred_lasso, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue')

plt.show()


