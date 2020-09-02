import numpy as np

from .layer import Layer


class Softmax(Layer):

    def __init__(
        self,
        layer_id: str,
        n_inputs: int,
    ):
        super(Softmax, self).__init__(
            layer_id=layer_id,
            n_inputs=n_inputs,
            n_outputs=n_inputs,
        )
        self.outputs = None

    def _run_forward(
        self,
        inputs: np.array
    ) -> np.array:

        exp = np.exp(inputs)
        exp_sums = np.sum(exp, axis=0)[0]

        for i in range(len(exp_sums)):
            exp[:, :, i] = exp[:, :, i] / exp_sums[i]

        self.outputs = exp

        return exp

    def _run_backward(
        self,
        error_wrt_output: np.array,
        lr: float
    ) -> np.array:

        error_wrt_input = np.zeros_like(self.inputs)

        for batch in range(error_wrt_output.shape[2]):
            for i in range(len(self.outputs)):
                for j in range(len(self.inputs)):
                    if i == j:
                        error_wrt_input[i, 0, batch] += self.outputs[i, 0, batch] * (1 - self.inputs[j, 0, batch])
                    else:
                        error_wrt_input[i, 0, batch] += -self.outputs[i, 0, batch] * self.inputs[j, 0, batch]

        for batch in range(error_wrt_output.shape[2]):
            for i in range(len(error_wrt_input)):
                error_wrt_input[i, 0, batch] = error_wrt_input[i, 0, batch] * error_wrt_output[i, 0, batch]

        return error_wrt_input
