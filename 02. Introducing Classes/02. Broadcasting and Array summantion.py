import numpy as np

# ----------------------------------- --- #
# ARRAY SUMMATION
# ---------------------------------------#

'''
NumPy allows summation across all elements or specific axes.

axis = None → sum over all elements
axis = 0    → column-wise sum
axis = 1    → row-wise sum

keepdims=True preserves dimensionality
(useful for broadcasting later)
'''

A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

print(np.sum(A))
print(np.sum(A, axis=None))
print(np.sum(A, axis=0))                 # column-wise sum
print(np.sum(A, axis=1))                 # row-wise sum
print(np.sum(A, axis=0, keepdims=True))  # column-wise with shape preserved
print(np.sum(A, axis=1, keepdims=True))  # row-wise with shape preserved


# ----------------------------------- --- #
# BROADCASTING + SUMMATION IN NEURAL NETWORKS
# ---------------------------------------#

'''
### Broadcasting Rules

Rule 1 — Compare shapes from RIGHT to LEFT

A.shape = (3, 1, 4)
B.shape = (    5, 4)

Aligned:
(3, 1, 4)
(1, 5, 4)

Rule 2 — Dimensions must be either:
- Equal, OR
- One of them is 1, OR
- One dimension is missing (left-padded with 1)

If these fail → broadcasting error

Rule 3 — Dimension of size 1 is "stretched"

A dimension with size 1 expands automatically.

Example:
X shape = (3, 1)
Y shape = (3, 4)

Broadcast result = (3, 4)
'''


# ----------------------------------- --- #
# BROADCASTING EXAMPLES
# ---------------------------------------#

'''
Example 1 — Scalar + Array

a = [1, 2, 3]
b = 5

b is broadcast to [5, 5, 5]
a + b → valid

Example 2 — (3,4) + (1,4)

(1,4) expands to (3,4) → valid

Example 3 — (3,1) + (3,4)

(3,1) expands to (3,4) → valid

Example 4 — Invalid Broadcasting

A = (3,4)
B = (2,4)

Compare last dim → 4 = 4 ✅
Compare next dim → 3 vs 2 ❌

→ Broadcasting error
'''


# ----------------------------------- --- #
# BROADCASTING IN NEURAL NETWORKS
# ---------------------------------------#

'''
### Weights × Inputs + Biases

output = np.dot(X, W) + b

X shape → (n_samples, n_features)
W shape → (n_features, n_neurons)
b shape → (1, n_neurons)

Bias b is broadcast across all rows automatically.

Final output shape:
(n_samples, n_neurons)
'''


# ----------------------------------- --- #
# PRACTICAL NUMPY EXAMPLE
# ---------------------------------------#

import numpy as np

X = np.array([
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8]
])  # shape (3,4)

W = np.array([
    [0.2, 0.8, -0.5],
    [0.5, -0.91, 0.26],
    [-0.26, -0.27, 0.17],
    [1.0, -0.5, 0.87]
])  # shape (4,3)

b = np.array([[2, 3, 0.5]])  # shape (1,3)

Z = np.dot(X, W)

'''
Shapes after dot product:

X = (3,4)
W = (4,3)
Z = (3,3)

Example Z:
[
 [4.8,   1.21,  2.385],
 [8.9,  -1.81,  0.2  ],
 [1.41,  1.051, 0.026]
]
'''


# ----------------------------------- --- #
# ADDING BIAS (BROADCASTING IN ACTION)
# ---------------------------------------#

'''
Z = (3,3)
b = (1,3)

b is expanded to:
[2, 3, 0.5]
[2, 3, 0.5]
[2, 3, 0.5]
'''

output = Z + b

'''
So the final computation becomes:

[
 [4.8+2,   1.21+3,   2.385+0.5],
 [8.9+2,  -1.81+3,   0.2+0.5 ],
 [1.41+2,  1.051+3,  0.026+0.5]
]
'''
