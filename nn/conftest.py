from pytest import fixture

from .layers.fully_connected import FullyConnected
from .layers.sigmoid import Sigmoid
from .layers.softmax import Softmax
from .layers.batch_normalization import BatchNormalization
from .net import Net
from .trainer import Trainer


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
def softmax() -> Softmax:
    return Softmax(
        'softmax',
        n_inputs=3
    )


@fixture(scope='session')
def batch_normalization() -> BatchNormalization:
    return BatchNormalization(
        'sigmoid',
        n_inputs=3,
        epsilon=0
    )


@fixture(scope='session')
def net() -> Net:
    return Net(
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
        )
    )


@fixture(scope='session')
def trainer() -> Trainer:
    net = Net(
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
        )
    )
    return Trainer(
        net=net
    )
