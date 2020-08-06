from pytest import fixture
from .layers.fully_connected import FullyConnected


@fixture(scope='session')
def fully_connected() -> FullyConnected:
    return FullyConnected(
        'fc',
        n_inputs=2,
        n_neurons=4,
        batch_size=2
    )
