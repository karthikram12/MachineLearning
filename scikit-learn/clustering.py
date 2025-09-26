import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

plt.subplot(2, 2, 1)
X, y = make_blobs(n_samples=5000, centers=5, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5)
kmeans.fit(X_scaled)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y)

plt.subplot(2, 2, 2)

X1, _ = make_moons(n_samples=5000, noise=0.05)
X1_scaled = scaler.fit_transform(X1)
kmeans.fit(X1_scaled)

plt.scatter(X1_scaled[:, 0], X1_scaled[:, 1], c=kmeans.labels_)

plt.subplot(2, 2, 3)

dbscan = DBSCAN(eps=0.5)
dbscan.fit(X1_scaled)
plt.scatter(X1_scaled[:, 0], X1_scaled[:, 1], c=dbscan.labels_)

plt.show()