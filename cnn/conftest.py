from pytest import fixture

from .layers.fully_connected import FullyConnected
from .layers.sigmoid import Sigmoid
from .net import Net


@fixture(scope='session')
def fully_connected() -> FullyConnected:
    return FullyConnected(
        'fc',
        n_inputs=2,
        n_neurons=4,
        batch_size=2
    )


@fixture(scope='session')
def sigmoid() -> Sigmoid:
    return Sigmoid(
        'sigmoid',
        n_inputs=3,
        batch_size=2
    )


@fixture(scope='session')
def net() -> Net:
    batch_size = 2
    return Net(
        n_inputs=2,
        batch_size=batch_size,
        layers=(
            FullyConnected(
                'fc_hidden',
                n_inputs=2,
                n_neurons=3,
                batch_size=batch_size
            ),
            Sigmoid(
                'sigmoid_hidden',
                n_inputs=3,
                batch_size=batch_size
            ),
            FullyConnected(
                'fc_output',
                n_inputs=3,
                n_neurons=1,
                batch_size=batch_size
            ),
            Sigmoid(
                'sigmoid_output',
                n_inputs=1,
                batch_size=batch_size
            )
        ),
        lr=0.1
    )
