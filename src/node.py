class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, val=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.val = val

    def is_leaf(self):
        return self.val is not None
