import sys
print(sys.executable)

# ----------------------------------- --- #
# SIMPLE NEURON (3 inputs)
# ---------------------------------------#

### SIMPLE NEURON:
inputs = [1.0, 2.0, 3.0]
weights = [0.2, 0.8, -0.5]
bias = 2.0

output = bias
for i in range(len(inputs)):
    output += (inputs[i]) * (weights[i])

print(output)


# ----------------------------------- --- #
# SIMPLE NEURON (5 inputs)
# ---------------------------------------#

inputs = [1.0, 2.0, 3.0, 4.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0, 1.0]
bias = 2.0

output = bias
for i in range(len(inputs)):
    output += (inputs[i]) * (weights[i])

print(output)


# ----------------------------------- --- #
# LAYER OF NEURONS (manual computation)
# ---------------------------------------#

### LAYER OF NEURONS:

# We have layer of neurons which has 3 output neurons with 4 inputs and weights..!
# and each output neuron has thrie respective weights..!
# Alright..!

inputs = [1, 2, 3, 2.5]

weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

n = len(inputs)

outputs = [
    # Neuron 1:
    inputs[0]*weights1[0] +
    inputs[1]*weights1[1] +
    inputs[2]*weights1[2] +
    inputs[3]*weights1[3] + bias1,

    # Neuron 2:
    inputs[0]*weights2[0] +
    inputs[1]*weights2[1] +
    inputs[2]*weights2[2] +
    inputs[3]*weights2[3] + bias2,

    # Neuron 3:
    inputs[0]*weights3[0] +
    inputs[1]*weights3[1] +
    inputs[2]*weights3[2] +
    inputs[3]*weights3[3] + bias3
]

print(outputs)


# ----------------------------------- --- #
# LAYER OF NEURONS (generic loop-based)
# ---------------------------------------#

# Neural networks typically have **layers** that consist of more than one neuron.
# Layers are nothing more than groups of neurons.
# Each neuron in a layer takes exactly the same input — the input given to the layer
# (which can be either the training data or the output from the previous layer),
# but contains its own set of weights and its own bias producing its own unique output.
#
# The layer’s output is a set of each of these outputs — one per each neuron.
# Let’s say we have a scenario with 3 neurons in a layer and 4 inputs:

inputs = [1, 2, 3, 2.5]

weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

biases = [2, 3, 0.5]

# Output of current layer
layer_outputs = []

# For each neuron
for neuron_weights, neuron_bias in zip(weights, biases):
    # Zeroed output of given neuron
    neuron_output = 0

    # For each input and weight to the neuron
    for neuron_input, weight in zip(inputs, neuron_weights):
        # Multiply this input by associated weight
        # and add to the neuron’s output variable
        neuron_output += neuron_input * weight  # (w1*x1 + w2*x2 + w3*x3 + w4*x4)

    # Add bias
    neuron_output += neuron_bias

    # Put neuron’s result to the layer’s output list
    layer_outputs.append(neuron_output)

print(layer_outputs)
