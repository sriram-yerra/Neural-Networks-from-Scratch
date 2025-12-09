import numpy as np

# ----------------------------------- --- #
# FIRST LAYER (from the first image)
# ---------------------------------------#

inputs = [
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

biases = [2.0, 3.0, 0.5]

layer1 = np.dot(inputs, np.array(weights).T) + biases
print("Layer 1 output:")
print(layer1)
print()


# ----------------------------------- --- #
# SECOND LAYER (from the second image)
# ---------------------------------------#

weights2 = [
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13]
]

biases2 = [-1.0, 2.0, -0.5]

layer2 = np.dot(layer1, np.array(weights2).T) + biases2
print("Layer 2 output:")
print(layer2)


# ----------------------------------- --- #
# SAME NETWORK — COMBINED VERSION
# ---------------------------------------#

inputs = [
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

weights2 = [
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13]
]

biases = [2.0, 3.0, 0.5]
biases2 = [-1.0, 2.0, -0.5]

layer1 = np.dot(inputs, np.array(weights).T) + biases
layer2 = np.dot(layer1, np.array(weights2).T) + biases2

print("Layer 1 output:")
print(layer1)
print()
print("Layer 2 output:")
print(layer2)


# ----------------------------------- --- #
# NNFS — THEORY BACKGROUND
# ---------------------------------------#

'''
## NNFS
### Linear vs. Non-linear Data

Linear data can be represented (fit) using a straight line —
for example, when y = f(x) forms a clear linear trend.

Non-linear data cannot be represented well by a straight line.

Simple ML models (like linear regression) handle linear data easily.

Neural networks are powerful because they can model non-linear relationships.
'''


# ----------------------------------- --- #
# WHY NNFS WAS CREATED
# ---------------------------------------#

'''
### Why nnfs was created

To simplify learning and testing,
the authors created a helper Python package called nnfs
(Neural Networks From Scratch).
'''


# ----------------------------------- --- #
# WHAT NNFS PROVIDES
# ---------------------------------------#

'''
### What nnfs provides

spiral_data() — creates a non-linear spiral dataset for
testing classification models.

from nnfs.datasets import spiral_data

nnfs.init() — ensures reproducibility by:

- Setting the random seed to 0
- Forcing NumPy to use float32 precision
- Overriding NumPy’s dot product to ensure consistent results
'''


# ----------------------------------- --- #
# SPIRAL DATASET DEMO
# ---------------------------------------#

import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()
X, y = spiral_data(samples=100, classes=3)

# plt.scatter(X[:, 0], X[:, 1])
# plt.show()

# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
# plt.show()
