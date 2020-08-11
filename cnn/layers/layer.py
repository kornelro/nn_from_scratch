import numpy as np
import abc


class Layer(abc.ABC):

    def __init__(
        self,
        layer_id: str,
        n_inputs: int,
        n_outputs: int,
        batch_size: int
    ):
        self.layer_id = layer_id
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.batch_size = batch_size
        self.inputs = None

    def forward(
        self,
        inputs: np.array
    ) -> np.array:

        if not inputs.shape == (self.n_inputs, 1, self.batch_size):
            raise Exception(
                F"Wrong input size {inputs.shape} in {self.layer_id},\
                expected ({self.n_inputs, 1, self.batch_size})"
            )

        self.inputs = inputs

        return self._run_forward(inputs)

    @abc.abstractmethod
    def _run_forward(
        self,
        inputs: np.array
    ) -> np.array:
        pass

    def backward(
        self,
        error_wrt_output: np.array,
        lr: float = 0.1
    ) -> np.array:

        if self.inputs is None:
            raise Exception(
                F'Backward must be proceded after forward in {self.layer_id}'
            )
        if not error_wrt_output.shape == (self.n_outputs, 1, self.batch_size):
            raise Exception(
                F"Wrong error_wrt_output size \
                {error_wrt_output.shape} in {self.layer_id}, \
                expected ({self.n_outputs}, 1, {self.batch_size})"
            )

        return self._run_backward(error_wrt_output, lr)

    @abc.abstractmethod
    def _run_backward(
        self,
        error_wrt_output: np.array,
        lr: float
    ) -> np.array:
        pass
