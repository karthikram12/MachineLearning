import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures

def data_prep_for_df(df):
    data = pd.DataFrame(df)
    data = data.set_index('Period')
    print(data.shape)
    all_cols = data.columns
    feature_columns = all_cols[1:]
    target_col = all_cols[0]
    final_df = create_lag_features(data, feature_columns, target_col, n_lags=4)
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
    # lagged['month'] = lagged.index.month
    # lagged['day_of_week'] = lagged.index.dayofweek
    lagged[f"{target}_t+1"] = df[target].shift(-1) - df[target]
    if drop_na:
        lagged = lagged.dropna()
    return lagged

def train_test_split(data, test_size):
    data_length = len(data)
    train_size = int(data_length * (1 - test_size))
    X = data.drop(columns=['Revenue_t+1'])
    y = data['Revenue_t+1']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
    return X_train, X_test, y_train, y_test, train_size, scaler_y

def model_selection(model_grid_params, X_train, y_train):
    best_models = {}
    tscv = TimeSeriesSplit(n_splits=3)
    for name, cfg in model_grid_params.items():
        random_cv = RandomizedSearchCV(estimator=cfg["model"], param_distributions=cfg["params"], scoring='r2', cv=tscv,
                                       random_state=42, n_iter=10)
        random_cv.fit(X_train, y_train)
        best_models[name] = {
            "best_model": random_cv.best_estimator_,
            "best_params": random_cv.best_params_,
            "best_score": random_cv.best_score_
        }
    return best_models

def fit_and_predict(model, X_train, y_train, X_test, use_poly=False, poly_degree=2):
    if use_poly:
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X_train_t = poly.fit_transform(X_train)
        X_test_t = poly.transform(X_test)
    else:
        X_train_t = X_train
        X_test_t = X_test

    model.fit(X_train_t, y_train)
    y_pred = model.predict(X_test_t)
    return y_pred, model, (poly if use_poly else None)

def reconstruct_abs_target(last_train_value, diffs):
    values = [last_train_value + diffs[0]]
    for diff in diffs[1:]:
        values.append(values[-1] + diff)
    return values

def calculate_scores(model,  X_train, y_train, X_test, y_test, y_pred, scaler_y, poly_used=None):
    if poly_used is not None:
        X_train_t = poly_used.transform(X_train)
        X_test_t = poly_used.transform(X_test)
    else:
        X_train_t = X_train
        X_test_t = X_test
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    train_score = model.score(X_train_t, y_train)
    test_score = model.score(X_test_t, y_test)
    y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    r_score_test = r2_score(y_test_original, y_pred_original)
    return train_score, test_score, r_score_test

data_raw = pd.read_csv('Month_Value_1.csv', parse_dates=['Period'])
data = data_prep_for_df(data_raw)
X_train, X_test, y_train, y_test, train_size, scaler = train_test_split(data, test_size=0.2)
model_param_grid = {
                    "LinearRegression": {
                        "model":LinearRegression(),
                        "params": {
                        }
                    },
                    "Ridge": {
                        "model":Ridge(),
                        "params": {
                            "alpha": [0.75, 0.8, 0.9],
                            "max_iter": [100, 200]
                        }
                    },
                    "Lasso": {
                        "model":Lasso(),
                        "params": {
                            "alpha": [0.75, 0.9],
                            "max_iter": [200, 400]
                        }
                    },
                    "DecisionTree": {
                        "model":DecisionTreeRegressor(),
                        "params": {
                            "max_depth": [5, 6, 7],
                            "min_samples_split": [2, 3, 4]
                        }
                    },
                    "RandomForestRegressor": {
                        "model":RandomForestRegressor(random_state=42),
                        "params": {
                            "n_estimators": [10, 30],
                            "max_depth": [4, 5, 6],
                            "min_samples_split": [2, 3],
                        }
                    }
                    }
best_models = model_selection(model_param_grid, X_train, y_train)
print(best_models)
best_model_name = max(best_models, key=lambda x: best_models[x]['best_score'])
best_model = best_models[best_model_name]['best_model']
print(best_model)
use_poly = isinstance(best_model, (LinearRegression, Ridge, Lasso))
poly = PolynomialFeatures(degree=2, include_bias=False) if use_poly else None
y_predictions_scaled, fitted_model, poly_used = fit_and_predict(best_model, X_train, y_train, X_test, use_poly=use_poly, poly_degree=2)
y_predictions = scaler.inverse_transform(y_predictions_scaled.reshape(-1, 1)).flatten()
train_index = data.index[:train_size]
last_train_value = data_raw.set_index('Period').loc[train_index[-1], 'Revenue']
reconstructed_y_predictions = reconstruct_abs_target(last_train_value, y_predictions)
train_score, test_score, r_score_test = calculate_scores(best_model, X_train, y_train, X_test, y_test, y_predictions_scaled,
                                                    scaler, poly_used)
print(train_score)
print(test_score)
print(r_score_test)
test_index = data.index[train_size:]
actual_abs_y = data_raw.set_index('Period').loc[test_index, 'Revenue']
plt.plot(actual_abs_y.index, actual_abs_y.values, label='Test')
plt.plot(actual_abs_y.index, reconstructed_y_predictions, label='Predictions')
plt.legend()
plt.show()



