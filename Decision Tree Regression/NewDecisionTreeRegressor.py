import numpy as np
import TreeNode

class LinearRegressor:
    def __init__(self, max_depth, min_number_of_splits):
        self.root = None
        self.max_depth = max_depth
        self.min_number_of_splits = min_number_of_splits

    def fit(self, X, y):
        self.root = self.build_tree(X, y, depth=0)

    def predict(self, X):
        return [self.predict_sample(np.array(x), self.root) for x in X]

    def build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_number_of_splits:
            leaf_value =  np.mean(y)
            return TreeNode.TreeNode(value=leaf_value)

        best_feature, best_threshold = self.best_split(X, y, n_features)

        if best_feature is None:
            return TreeNode.TreeNode(np.mean(y))

        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold

        left_child = self.build_tree(X[left_idx], y[left_idx], depth+1)
        right_child = self.build_tree(X[right_idx], y[right_idx], depth+1)

        return TreeNode.TreeNode(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def best_split(self, X, y, n_features):
        best_mse = float("inf")
        best_feature, best_threshold = None, None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_idx = X[:, feature] <= t
                right_idx = X[:, feature] > t

                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                mse = self.compute_mse(y[left_idx], y[right_idx])

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = t

        return best_feature, best_threshold

    def compute_mse(self, y_left, y_right):
        n_left, n_right = len(y_left), len(y_right)
        n_total = n_left + n_right

        def mse(y):
            return np.mean((y - np.mean(y)) ** 2) if len(y) > 0 else 0

        return (n_left / n_total) * mse(y_left) + (n_right / n_total) * mse(y_right)

    def predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)

