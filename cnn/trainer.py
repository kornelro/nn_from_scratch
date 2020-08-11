import numpy as np

from .net import Net


class trainer():

    def __init__(
        self,
        net: Net
    ):
        self.net = Net

    def train(
        self,
        X_train: np.array,
        y_train: np.array,
        batch_size: int,
        epochs: int
    ):
        pass
