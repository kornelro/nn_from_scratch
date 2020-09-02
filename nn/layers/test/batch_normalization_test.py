import numpy as np

from ..batch_normalization import BatchNormalization


def test_batch_normalization_forward_size(
    batch_normalization: BatchNormalization
):
    inputs = np.array([
        [[1, 2]],
        [[2, 3]],
        [[3, 4]]
    ])

    output = batch_normalization.forward(inputs)

    assert output.shape == inputs.shape


def test_batch_normalization_forward_value(
    batch_normalization: BatchNormalization
):
    inputs = np.array([
        [[1, 2]],
        [[2, 5]],
        [[3, 4]]
    ])

    batch_normalization.gamma = 0.5
    batch_normalization.beta = 0.5

    output = batch_normalization.forward(inputs)

    assert np.array_equal(
        np.around(output, 2),
        np.array([
            [[-0.29, 0.22]],
            [[0.14, 1.22]],
            [[0.57, 0.89]]
        ])
    )
    assert np.mean((output - 0.5)/0.5).round() == 0
    assert np.std((output - 0.5)/0.5).round() == 1


def test_batch_normalization_backward(
    batch_normalization: BatchNormalization
):
    inputs = np.array([
        [[1, 2]],
        [[2, 3]],
        [[3, 4]]
    ])

    error_wrt_output = np.array([
        [[0.1, 0.2]],
        [[0.2, 0.3]],
        [[0.3, 0.4]]
    ])

    gamma = batch_normalization.gamma
    beta = batch_normalization.beta

    batch_normalization.forward(inputs)
    output = batch_normalization.backward(error_wrt_output)

    assert output.shape == inputs.shape
    assert gamma != batch_normalization.gamma
    assert beta != batch_normalization.beta
