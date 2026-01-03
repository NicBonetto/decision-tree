import numpy as np
from src.node import Node
from src.utils import gini_impurity, split_data, most_common_class

class DecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, x, y):
        self.root = self._build_tree(x, y, depth=0)
        return self
    
    def predict(self, x):
        return np.array([self._traverse_tree(n, self.root) for n in x])

    def _build_tree(self, x, y, depth):
        n_samples, n_features = x.shape
        n_classes = len(np.unique(y))

        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1):
            leaf_value = most_common_class(y)
            return Node(val=leaf_value)

        best_feature, best_threshold = self._best_split(x, y)

        if best_feature is None:
            leaf_value = most_common_class(y)
            return Node(val=leaf_value)

        x_left, x_right, y_left, y_right = split_data(x, y, best_feature, best_threshold)

        left_child = self._build_tree(x_left, y_left, depth + 1)
        right_child = self._build_tree(x_right, y_right, depth + 1)

        return Node(feature_idx=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def _best_split(self, x, y):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        n_samples, n_features = x.shape

        for feature_idx in range(n_features):
            thresholds = np.unique(x[:, feature_idx])

            for threshold in thresholds:
                x_left, x_right, y_left, y_right = split_data(x, y, feature_idx, threshold)

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                n_left, n_right = len(y_left), len(y_right)
                gini_left = gini_impurity(y_left)
                gini_right = gini_impurity(y_right)

                weighted_gini = (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.val

        if x[node.feature_idx] < node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
