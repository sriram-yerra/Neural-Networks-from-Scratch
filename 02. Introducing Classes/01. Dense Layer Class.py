import nnfs
from nnfs.datasets import spiral_data
import numpy as np


# ----------------------------------- --- #
# WEIGHTS & BIASES — SHAPE CONFUSION
# ---------------------------------------#

'''
The Weights Dimension is the ***(n_inputs × n_neurons)***

The Bias Dimension is the ***(n_neurons)***

The common confusion:

Some tutorials define weights as:

***W = (neurons, inputs)***

meaning "each row corresponds to one neuron".

Others define weights as:

***W = (inputs, neurons)***

meaning "each column corresponds to one neuron".

Both are valid — but they change how you perform the dot product.
'''


# ----------------------------------- --- #
# DENSE LAYER IMPLEMENTATION
# ---------------------------------------#

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # weights shape: (n_inputs, n_neurons)
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

        # biases shape: (1, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def foreward_pass(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# ----------------------------------- --- #
# INPUT MATRIX (X) AND TARGET VECTOR (y)
# ---------------------------------------#

'''
In machine learning:

X represents the input feature matrix
y represents the output labels or targets

Uppercase X = matrix (many samples)
Lowercase y = vector (targets)

X = many xs (inputs)
y = outputs (targets)
'''

X, y = spiral_data(samples=100, classes=3)


# ----------------------------------- --- #
# FORWARD PASS DEMONSTRATION
# ---------------------------------------#

# Create Dense layer with 2 input features and 3 neurons (output values)
dense1 = Layer_Dense(2, 3)

# Perform a forward pass of our training data through this layer
dense1.foreward_pass(X)

print(dense1.output[:5])
# print(np.array(dense1.output[:5]).T)


# ----------------------------------- --- #
# INTERPRETING THE OUTPUT SHAPE
# ---------------------------------------#

'''
You’re seeing multiple rows because you have multiple input samples,
and 3 columns because you have 3 neurons.

The layer outputs one vector (3 values)
for each input sample.
'''

# ----------------------------------- --- #
# FINAL CODE (POLISHED VERSION)
# ---------------------------------------#

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# Initialize NNFS (sets random seed, dtype, etc.)
# nnfs.init()

# Dense layer
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Display first 5 outputs
print(dense1.output[:5])
