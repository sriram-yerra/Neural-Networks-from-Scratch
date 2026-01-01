import numpy as np

class Layer:
    """
    A dense (fully connected) layer for a neural network.
    This layer performs a linear transformation: output = inputs @ weights + biases.
    It supports L1 and L2 regularization on weights and biases.
    """
    '''
    Shapes
    inputs: (batch_size, num_inputs)
    weights: (num_inputs, num_neurons)
    biases: (1, num_neurons)
    derivatives (from next layer): (batch_size, num_neurons)
    '''
    def __init__(self, num_inputs, num_neurons, *, w_l1, b_l1, w_l2, b_l2):

        # Store layer dimensions
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        
        # Initialize weights randomly with small values to break symmetry
        self.weights = np.random.randn(num_inputs, num_neurons) * 0.01
        
        # Initialize biases to zeros
        self.biases = np.zeros((1, num_neurons))
        
        # Store regularization parameters
        self.w_l1 = w_l1
        self.b_l1 = b_l1
        self.w_l2 = w_l2
        self.b_l2 = b_l2

        # Attributes to store intermediate values during forward/backward passes
        self.inputs = None  # Input data for backpropagation
        self.output = None  # Output of the layer
        self.dw = None      # Gradient of loss w.r.t. weights
        self.db = None      # Gradient of loss w.r.t. biases
        self.derivatives = None  # Derivatives to pass to previous layer

    def forward(self, inputs):
        """
        Perform the forward pass through the layer.
        Args: inputs (np.ndarray): Input data of shape (batch_size, num_inputs).
        Returns: np.ndarray: Output of shape (batch_size, num_neurons).
        """
        # Store inputs for use in backward pass
        self.inputs = inputs
        
        # Compute linear transformation: output = inputs @ weights + biases
        self.output = np.dot(inputs, self.weights) + self.biases
        
        return self.output

    def backward(self, derivatives: np.ndarray):
        """
        Perform the backward pass through the layer, computing gradients.
        ** Args: derivatives (np.ndarray): "Derivatives from the next layer", shape (batch_size, num_neurons).
        """
        # Compute gradient w.r.t. weights: dw = inputs.T @ derivatives
        self.dw = self.inputs.T @ derivatives
        
        # Compute gradient w.r.t. biases: db = sum of derivatives over batch dimension
        self.db = derivatives.sum(axis=0, keepdims=True)
        
        # Compute derivatives for previous layer: derivatives = derivatives @ weights.T
        self.derivatives = derivatives @ self.weights.T

        # Adding the Elastic Net regularization gradients (lasso + ridge)..!

        # Add L1 regularization gradients (Lasso)
        if self.w_l1 > 0:
            # L1 gradient is sign of weights (1 for positive, -1 for negative)
            d_w_l1 = np.ones_like(self.weights)
            d_w_l1[self.weights < 0] = -1
            self.dw += self.w_l1 * d_w_l1

        if self.b_l1 > 0:
            d_b_l1 = np.ones_like(self.biases)
            d_b_l1[self.biases < 0] = -1
            self.db += self.b_l1 * d_b_l1

        # Add L2 regularization gradients (Ridge)
        if self.w_l2 > 0:
            # L2 gradient is 2 * lambda * weights
            d_w_l2 = 2 * self.w_l2 * self.weights
            self.dw += d_w_l2

        if self.b_l2 > 0:
            d_b_l2 = 2 * self.b_l2 * self.biases
            self.db += d_b_l2