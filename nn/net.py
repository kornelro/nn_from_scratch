from typing import Tuple

import numpy as np

from .layers.layer import Layer
from .utils import input_is_proper_size


class Net():

    def __init__(
        self,
        layers: Tuple[Layer, ...],
    ):
        self.n_inputs = layers[0].n_inputs
        self.layers = layers
        self.output = None

    def forward(
        self,
        inputs: np.array
    ):

        if not input_is_proper_size(inputs, self.n_inputs):
            raise Exception(
                F"Wrong net input size {inputs.shape},\
                expected ({self.n_inputs}, 1, batch_size)"
            )

        output = self.layers[0].forward(inputs)
        for layer in self.layers[1:]:
            output = layer.forward(output)

        self.output = output

        return output

    def backward(
        self,
        y_true: np.array,
        lr: float
    ):

        if self.output is None:
            raise Exception(
                'Backward must be proceded after forward in net'
            )
        if not input_is_proper_size(y_true, self.layers[-1].n_outputs):
            raise Exception(
                F"Wrong y_true size {y_true.shape},\
                expected ({self.layers[-1].n_outputs}, 1, batch_size)"
            )

        error_wrt_output = self.output - y_true

        error_wrt_layer_output = error_wrt_output
        for layer in self.layers[::-1]:
            error_wrt_layer_output = layer.backward(
                error_wrt_layer_output,
                lr
             )
