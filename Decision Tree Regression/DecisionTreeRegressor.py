import numpy as np
import TreeNode

class DecisionTreeRegressorScratch:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        # stopping conditions
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            leaf_value = np.mean(y)
            return TreeNode.TreeNode(value=leaf_value)

        # find best split
        best_feat, best_thresh = self._best_split(X, y, n_features)

        if best_feat is None:  # no split found
            return TreeNode.TreeNode(value=np.mean(y))

        # split dataset
        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh

        left_child = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return TreeNode.TreeNode(feature=best_feat, threshold=best_thresh,
                        left=left_child, right=right_child)

    def _best_split(self, X, y, n_features):
        best_mse = float("inf")
        best_feat, best_thresh = None, None

        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left_idx = X[:, feat] <= t
                right_idx = X[:, feat] > t

                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                # compute MSE of this split
                mse = self._split_mse(y[left_idx], y[right_idx])

                if mse < best_mse:
                    best_mse = mse
                    best_feat = feat
                    best_thresh = t

        return best_feat, best_thresh

    def _split_mse(self, y_left, y_right):
        """Compute weighted MSE of a split"""
        n_left, n_right = len(y_left), len(y_right)
        n_total = n_left + n_right

        def mse(y):
            return np.mean((y - np.mean(y)) ** 2) if len(y) > 0 else 0

        return (n_left / n_total) * mse(y_left) + (n_right / n_total) * mse(y_right)

    def _predict_sample(self, x, node):
        # if leaf node, return stored value
        if node.value is not None:
            return node.value

        # otherwise, decide left or right
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
