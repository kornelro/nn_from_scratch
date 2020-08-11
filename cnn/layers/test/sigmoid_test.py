import numpy as np

from ..sigmoid import Sigmoid


def test_sigmoid_forward_size(sigmoid: Sigmoid):
    n_inputs = sigmoid.n_inputs
    batch_size = sigmoid.batch_size

    output = sigmoid.forward(
        np.random.uniform(
            size=(n_inputs, 1, batch_size)
        )
    )

    assert output.shape == (n_inputs, 1, batch_size)


def test_sigmoid_forward_values(sigmoid: Sigmoid):

    inputs = np.array(
        [
            [[1, 2]],
            [[2, 1]],
            [[1, 2]]
        ]
    )

    output = sigmoid.forward(inputs)

    assert np.array_equal(
        np.around(output, 3),
        np.array(
            [
                [[0.731, 0.881]],
                [[0.881, 0.731]],
                [[0.731, 0.881]]
            ]
        )
    )


def test_sigmoid_backward_size(sigmoid: Sigmoid):
    n_inputs = sigmoid.n_inputs
    batch_size = sigmoid.batch_size

    sigmoid.forward(
        np.random.uniform(
            size=(n_inputs, 1, batch_size)
        )
    )

    output = sigmoid.backward(
        np.random.uniform(
            size=(n_inputs, 1, batch_size)
        )
    )

    assert output.shape == (n_inputs, 1, batch_size)
