import numpy as np

from .layer import Layer


class Sigmoid(Layer):

    def __init__(
        self,
        layer_id: str,
        n_inputs: int,
    ):
        super(Sigmoid, self).__init__(
            layer_id=layer_id,
            n_inputs=n_inputs,
            n_outputs=n_inputs,
        )

    def _run_forward(
        self,
        inputs: np.array
    ) -> np.array:

        return self._sigmoid(inputs)

    def _run_backward(
        self,
        error_wrt_output: np.array,
        lr: float
    ) -> np.array:

        y = self._sigmoid(self.inputs)
        outputs_wrt_input = y * (1-y)

        return error_wrt_output * outputs_wrt_input

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
