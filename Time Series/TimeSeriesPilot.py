import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def data_prep_for_df(df):
    data = pd.DataFrame(df)
    data = data.set_index('Period')
    print(data.shape)
    all_cols = data.columns
    feature_columns = all_cols[1:]
    target_col = all_cols[0]
    final_df = create_lag_features(data, feature_columns, target_col, n_lags=5)
    final_df = final_df.drop(columns=data.columns)
    return final_df

def create_lag_features(df, feature_cols, target, n_lags, drop_na=True):
    df = df.copy().sort_index()
    lagged = df.copy()
    for col in feature_cols:
        for lag in range(1, n_lags+1):
            lagged[f"{col}_lag{lag}"] = df[col].shift(lag)
    for lag in range(1, n_lags+1):
        lagged[f"{target}_lag{lag}"] = df[target].shift(lag)
    lagged[f"{target}_t+1"] = df[target].shift(-1) - df[target]
    if drop_na:
        lagged = lagged.dropna()
    return lagged

def train_test_split(data, test_size):
    data_length = len(data)
    train_size = int(data_length * (1 - test_size))
    X = data.drop(columns=['Revenue_t+1'])
    y = data['Revenue_t+1']
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    return X_train, X_test, y_train, y_test, train_size

def fit_and_predict(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def reconstruct_abs_target(last_train_value, diffs):
    values = [last_train_value + diffs[0]]
    for diff in diffs[1:]:
        values.append(values[-1] + diff)
    return values

def calculate_scores(model,  X_train, y_train, X_test, y_test, y_pred):
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    r_score = r2_score(y_test, y_pred)
    return train_score, test_score, r_score

data_raw = pd.read_csv('Month_Value_1.csv', parse_dates=['Period'])
data = data_prep_for_df(data_raw)
X_train, X_test, y_train, y_test, train_size = train_test_split(data, test_size=0.2)
model = Ridge(max_iter=1000, alpha=0.8)
y_predictions = fit_and_predict(model, X_train, y_train, X_test, y_test)
last_train_value = data_raw.set_index('Period').loc[X_train.index[-1], 'Revenue']
reconstructed_y_predictions = reconstruct_abs_target(last_train_value, y_predictions)
train_score, test_score, r_score = calculate_scores(model, X_train, y_train, X_test, y_test, y_predictions)
print(train_score)
print(test_score)
print(r_score)
actual_abs_y = data_raw.set_index('Period').loc[y_test.index, 'Revenue']
plt.plot(actual_abs_y.index, actual_abs_y.values, label='Test')
plt.plot(actual_abs_y.index, reconstructed_y_predictions, label='Predictions')
plt.legend()
plt.show()



