from pytest import fixture

from .layers.fully_connected import FullyConnected
from .layers.sigmoid import Sigmoid
from .net import Net


@fixture(scope='session')
def fully_connected() -> FullyConnected:
    return FullyConnected(
        'fc',
        n_inputs=2,
        n_neurons=4
    )


@fixture(scope='session')
def sigmoid() -> Sigmoid:
    return Sigmoid(
        'sigmoid',
        n_inputs=3
    )


@fixture(scope='session')
def net() -> Net:
    return Net(
        n_inputs=2,
        layers=(
            FullyConnected(
                'fc_hidden',
                n_inputs=2,
                n_neurons=3,
            ),
            Sigmoid(
                'sigmoid_hidden',
                n_inputs=3,
            ),
            FullyConnected(
                'fc_output',
                n_inputs=3,
                n_neurons=1,
            ),
            Sigmoid(
                'sigmoid_output',
                n_inputs=1,
            )
        ),
        lr=0.1
    )
