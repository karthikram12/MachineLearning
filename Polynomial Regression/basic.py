import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

data = pd.read_csv('Fish.csv')
data = pd.DataFrame(data)

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_values = ohe.fit_transform(data[['Species']])
encoded_columns = ohe.get_feature_names_out(['Species'])
encoded_df = pd.DataFrame(encoded_values, columns=encoded_columns)
data = data.drop(columns='Species')
data = pd.concat([data, encoded_df], axis=1)

data_corr_matrix = data.corr()
#sns.heatmap(data_corr_matrix, cmap='coolwarm', annot=True)

X = data.drop(columns='Weight')
y = data['Weight']

kf = KFold(n_splits=5, shuffle=True, random_state=42)
poly_features = PolynomialFeatures(degree=2, include_bias=False)

def return_cross_val_scores(model, X, y, poly=None):
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if poly:
            X_train = poly.fit_transform(X_train)
            X_test = poly.transform(X_test)

        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))

    return np.mean(scores)

avg_linear_score = return_cross_val_scores(LinearRegression(), X, y)
avg_poly_score = return_cross_val_scores(LinearRegression(), X, y, poly_features)

print("Polynomial Regression score:", avg_poly_score)
print("Linear Regression score:", avg_linear_score)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

ridge = Ridge(alpha=1, max_iter=1000)
lasso = Lasso(alpha=1, max_iter=1000)

if avg_poly_score > avg_linear_score:
    print("polynomial regression is the best fit for the model")
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)
    ridge.fit(X_train_poly, y_train)
    y_pred_ridge = ridge.predict(X_test_poly)
    lasso.fit(X_train_poly, y_train)
    y_pred_lasso = lasso.predict(X_test_poly)
    print("Polynomial Regression with Lasso Reg:", lasso.score(X_test_poly, y_test))
    print("Polynomial Regression with Ridge Reg:", ridge.score(X_test_poly, y_test))
    plt.subplot(2,1,1)
    plt.scatter(y_test, y_pred, color='red')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue', marker='o')

else:
    print("Linear Regression is the best fit")
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    print("Linear Regression with Lasso Reg:", lasso.score(X_test, y_test))
    print("Linear Regression with Ridge Reg:", ridge.score(X_test, y_test))
    plt.subplot(2, 1, 1)
    plt.scatter(y_test, y_pred, color='red')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue', marker='o')

residual = y_test - y_pred
plt.subplot(2,1,2)
plt.scatter(y_pred, residual, color='red')
plt.show()



