import numpy as np

# ----------------------------------- --- #
# WHAT ARE “TENSORS”?
# ---------------------------------------#

'''
### What are “tensors?”
Tensors are closely-related to arrays. 

If you interchange tensor/array/matrix when it comes to
machine learning, people probably won’t give you too hard of a time. But there are subtle
differences, and they are primarily either the context or attributes of the tensor object. 

To understand a tensor, let’s compare and describe some of the other data containers
in Python (things that hold data). 

Let’s start with a list. A Python list is defined by comma-separated
objects contained in brackets. So far, we’ve been using lists.

A tensor object is an object that can be represented as an array.
'''

# ----------------------------------- --- #
# VECTOR × VECTOR
# ---------------------------------------#

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Elementwise multiplication
print(a * b)   # → [ 4 10 18]

# Dot product
print(np.dot(a, b))  # → 32  (1*4 + 2*5 + 3*6)


# ----------------------------------- --- #
# VECTOR × MATRIX
# ---------------------------------------#

v = np.array([1, 2, 3])  # 1×3
M = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])  # 3×3

# Vector × Matrix (v treated as row vector)
print(v @ M)
# → [30 36 42]

# Matrix × Vector (v treated as column vector)
print(M @ v)
# → [14 32 50]


# ----------------------------------- --- #
# MATRIX × MATRIX
# ---------------------------------------#

A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

print(A @ B)
# → [[19 22]
#    [43 50]]


# ----------------------------------- --- #
# OUTPUT OF A NEURON USING DOT PRODUCT
# ---------------------------------------#

inputs = [
    1,
    2,
    3,
    2.5
]  # This is Column Vector.

weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

biases = [2, 3, 0.5]

# Dot product of weights and inputs + biases
neuron_outputs = np.dot(weights, inputs) + biases

# layer_outputs = layer_outputs + biases
print(neuron_outputs)


# ----------------------------------- --- #
# BATCH INPUTS + TRANSPOSE LOGIC
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

'''
Doing Transpose is very Important to get the correct calculation of neuron
through Matrix Multiplication.
'''

layer_outputs = np.dot(inputs, np.array(weights).T) + biases
print(layer_outputs)
