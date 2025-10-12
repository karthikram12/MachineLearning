import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

data = pd.read_csv('winequality-red.csv')
data = pd.DataFrame(data).dropna()
print(data.head())
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
tree = DecisionTreeRegressor(min_samples_split=2, max_depth=5)
output = pd.Series(index=data.index, dtype=float)
r2_scores = []
scores = []
for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    tree.fit(X_train_scaled,y_train)
    predictions = tree.predict(X_test_scaled)
    predictions = np.round(predictions).astype(int)
    output.iloc[test_index] = predictions
    score = tree.score(X_test_scaled, y_test)
    r2 = r2_score(y_test, predictions)
    scores.append(score)
    r2_scores.append(r2)

data['predicted_quality'] = output
print(data.to_string())
print(r2_scores)
print(scores)