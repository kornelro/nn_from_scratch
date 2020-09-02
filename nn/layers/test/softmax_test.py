import numpy as np

from ..softmax import Softmax


def test_softmax_forward_size(softmax: Softmax):
    n_inputs = softmax.n_inputs
    batch_size = 2

    output = softmax.forward(
        np.random.uniform(
            size=(n_inputs, 1, batch_size)
        )
    )

    assert output.shape == (n_inputs, 1, batch_size)


def test_softmax_forward_values(softmax: Softmax):

    inputs = np.array(
        [
            [[1, 2]],
            [[2, 1]],
            [[1, 2]]
        ]
    )

    output = softmax.forward(inputs)

    assert np.array_equal(
        np.around(output, 3),
        np.array(
            [
                [[0.212, 0.422]],
                [[0.576, 0.155]],
                [[0.212, 0.422]]
            ]
        )
    )


def test_softmax_backward_size(softmax: Softmax):
    n_inputs = softmax.n_inputs
    batch_size = 2

    softmax.forward(
        np.random.uniform(
            size=(n_inputs, 1, batch_size)
        )
    )

    output = softmax.backward(
        np.random.uniform(
            size=(n_inputs, 1, batch_size)
        )
    )

    assert output.shape == (n_inputs, 1, batch_size)
