# decision-tree
A decision tree from scratch

## Description
This implementation demonstrates the core concepts for decision tree classification.
- Recursive tree building using Gini impurity
- Binary splitting on features and thresholds
- Configurable stopping criteria (max depth, min samples)
- Prediction via tree traversal

## API
```python
from decision_tree.tree import DecisionTreeClassifier
import numpy as np

x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 0, 1, 1])

tree = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
tree.fit(x, y)

predictions = tree.predict(new_vals)
```

## Running examples
```bash
python -m examples.iris
```
