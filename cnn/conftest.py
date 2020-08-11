from pytest import fixture

from .layers.fully_connected import FullyConnected
from .layers.sigmoid import Sigmoid


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
