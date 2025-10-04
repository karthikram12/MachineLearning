import pandas as pd
from NewDecisionTreeRegressor import LinearRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import KFold

data = pd.read_csv('winequality-red.csv')
data = pd.DataFrame(data)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
tree = LinearRegressor(max_depth=5, min_number_of_splits=2)
output_data = pd.DataFrame()

def compare_results(y_test, output, i, output_data):
    y_test = pd.Series(y_test).reset_index(drop=True)
    output = pd.Series(output).round().astype(int)
    df = pd.DataFrame({ f"y_test{i}":y_test, f"output{i}":output})
    df[f'result{i}'] = df[f'y_test{i}'] == df[f'output{i}']
    output_data = pd.concat([output_data, df], axis=1)
    return output_data

i = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    tree.fit(np.array(X_train), np.array(y_train))
    output = tree.predict(np.array(X_test))
    output_data = compare_results(y_test, output, i, output_data)
    i += 1

print(output_data.to_string())
print(output_data.mean())