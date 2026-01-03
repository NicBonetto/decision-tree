import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import sys
from src.tree import DecisionTreeClassifier

sys.path.append('..')

print('Loading Iris dataset...')
iris = load_iris()
x, y = iris.data, iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f'Training sampels: {len(x_train)}')
print(f'Test samples: {len(x_test)}')
print(f'Features: {iris.feature_names}')
print(f'Classes: {iris.target_names}')

print('\nTraining decision tree from scratch...')
tree = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
tree.fit(x_train, y_train)

y_pred = tree.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'\nAccuracy: {accuracy:.3f}')
print('\nConfusion Matrix:')
print(conf_matrix)

