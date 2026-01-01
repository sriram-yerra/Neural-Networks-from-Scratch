import abc
import numpy as np

class ActivationFunction(abc.ABCMeta):
    # Abstract Base Class metaclass for activation functions
    def __init__(self) -> None:
        self.inputs = None
        self.output = None
        self.derivatives = None

    @abc.abstractmethod
    def forward(self, inputs: np.ndarray) -> None:
        """
        Forward pass of the activation function.
        """
        pass

    @abc.abstractmethod
    def backward(self, derivatives: np.ndarray) -> None:
        """
        Backward pass of the activation function.
        """
        pass


class ReLU(Activation):
    """
    Rectified Linear Unit activation function.
    Forward: `x > 0` -> `x` otherwise `0`
    Backward: `x > 0` -> `1` otherwise `0`
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: np.ndarray) -> None:
        """
        Forward pass of the activation function.
        Forward: `x > 0` -> `x` otherwise `0`
        """
        # Remember the inputs for the backward pass
        self.inputs = inputs
        # Apply the activation function. ReLU is applied element-wise.
        self.output = np.maximum(0, inputs)

    def backward(self, derivatives: np.ndarray) -> None:
        """
        Backward pass of the activation function.
        Backward: `x > 0` -> `1` otherwise `0`
        """
        self.derivatives = derivatives.copy()
        # Apply the derivatives of the activation function.
        self.derivatives[self.inputs < 0] = 0