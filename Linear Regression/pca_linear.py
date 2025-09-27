import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

data = pd.read_csv('song_data.csv')
data = pd.DataFrame(data)
data = data.drop(columns='song_name')
data_corr_matrix = data.corr()
X = data.drop(columns='song_popularity')
y = data['song_popularity']
X_scaled = (X - X.mean()) / X.std(ddof=0)
cov_matrix = np.cov(X_scaled.T)
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
sorted_idx = np.argsort(eig_vals)[::-1]
sorted_eig_vals = eig_vals[sorted_idx][:2]
sorted_eig_vecs = eig_vecs[:, sorted_idx][:, :2]
X_pca = X_scaled.dot(sorted_eig_vecs)
X_pca = pd.DataFrame(X_pca).rename(columns={0:'PC1', 1:'PC2'})
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
linear = LinearRegression()
linear.fit(X_train, y_train)
y_pred = linear.predict(X_test)

#plt.scatter(y_test, y_pred, color='red')
#plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue')
sns.heatmap(data_corr_matrix, cmap='coolwarm', annot=True)

plt.show()
