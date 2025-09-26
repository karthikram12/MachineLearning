import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error

data = pd.read_csv('insurance.csv')
data = pd.DataFrame(data)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_values = ohe.fit_transform(X[['sex', 'smoker', 'region']])
encoded_columns = ohe.get_feature_names_out(['sex', 'smoker', 'region'])
encoded_df = pd.DataFrame(encoded_values, columns=encoded_columns)
X = data.drop(columns=['sex', 'smoker', 'region'])
X = pd.concat([X, encoded_df], axis=1)

corr_matrix = X.corr()

#Required X_features are picked
X = X[['age', 'bmi', 'smoker_no', 'smoker_yes']]
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear = LinearRegression()
linear.fit(X_train, y_train)
y_pred = linear.predict(X_test)

plt.scatter(y_test, y_pred, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue')

r2_score = linear.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R2 score:", r2_score)

#sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
#sns.pairplot(data)

plt.show()




