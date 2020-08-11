import numpy as np

from ..net import Net


def test_net_forward(net: Net):

    batch_size = 2

    inputs = np.array(
        [
            [[1, 2]],
            [[2, 1]]
        ]
    )

    output = net.forward(inputs)

    assert output.shape == (1, 1, batch_size)
    assert output[0][0][0] >= 0 and output[0][0][0] <= 1
    assert output[0][0][1] >= 0 and output[0][0][1] <= 1


def test_net_backward(net: Net):

    inputs = np.array(
        [
            [[1, 2]],
            [[2, 1]]
        ]
    )
    y_true = np.array([[[1, 0]]])

    weights_hidden = np.copy(net.layers[0].weights)
    weights_output = np.copy(net.layers[2].weights)

    net.forward(inputs)
    net.backward(y_true)

    assert not np.array_equal(
        net.layers[0].weights,
        weights_hidden
    )
    assert not np.array_equal(
        net.layers[2].weights,
        weights_output
    )
