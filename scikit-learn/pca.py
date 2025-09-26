from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X, y = fetch_openml('iris', return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pca = PCA(n_components=2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled_reduced = pca.fit_transform(X_train_scaled)
X_test_scaled_reduced = pca.transform(X_test_scaled)

log = KNeighborsClassifier()
log.fit(X_train_scaled_reduced, y_train)
output = log.predict(X_test_scaled_reduced)

accuracy = accuracy_score(y_test, output)
precision = precision_score(y_test, output, average='macro')
recall = recall_score(y_test, output, average='macro')
f_one = f1_score(y_test, output, average='macro')

print(accuracy, precision, recall, f_one)