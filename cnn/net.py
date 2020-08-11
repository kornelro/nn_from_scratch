from typing import Tuple
from .layers.layer import Layer
import numpy as np


class Net():

    def __init__(
        self,
        n_inputs: int,
        batch_size: int,
        layers: Tuple[Layer, ...],
        lr: float = 0.01
    ):
        self.n_inputs = n_inputs
        self.batch_size = batch_size
        self.layers = layers
        self.lr = lr
        self.output = None

    def forward(
        self,
        inputs: np.array
    ):

        if not inputs.shape == (self.n_inputs, 1, self.batch_size):
            raise Exception(
                F"Wrong net input size {inputs.shape},\
                expected ({self.n_inputs, 1, self.batch_size})"
            )

        output = self.layers[0].forward(inputs)
        for layer in self.layers[1:]:
            output = layer.forward(output)

        self.output = output

        return output

    def backward(
        self,
        y_true: np.array
    ):

        if self.output is None:
            raise Exception(
                'Backward must be proceded after forward in net'
            )
        if not y_true.shape == (1, 1, self.batch_size):
            raise Exception(
                F"Wrong y_true size {y_true.shape},\
                expected ({1, 1, self.batch_size})"
            )

        error_wrt_output = self.output - y_true

        error_wrt_layer_output = error_wrt_output
        for layer in self.layers[::-1]:
            error_wrt_layer_output = layer.backward(
                error_wrt_layer_output,
                self.lr
             )
