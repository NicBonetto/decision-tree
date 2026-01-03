import numpy as np 

def gini_impurity(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    gini = 1.0 - np.sum(probabilities ** 2)
    return gini

def split_data(x, y, feature_idx, threshold):
    left_mask = x[:, feature_idx] < threshold
    right_mask = ~left_mask

    x_left = x[left_mask]
    y_left = y[left_mask]

    x_right = x[right_mask]
    y_right = y[right_mask]

    return x_left, x_right, y_left, y_right

def most_common_class(y):
    classes, counts = np.unique(y, return_counts=True)
    return classes[np.argmax(counts)]

