from typing import List

import numpy as np
from tqdm import tqdm

from .net import Net


class Trainer():

    def __init__(
        self,
        net: Net
    ):
        self.net = net

    def train(
        self,
        X_train: np.array,
        y_train: np.array,
        batch_size: int,
        epochs: int,
        lr: float
    ):
        if len(X_train.shape) != 2:
            raise Exception('X_train must be 2-dim array')
        if (len(y_train.shape) != 2
                or y_train.shape[1] != 1):
            raise Exception('y_train must be 2-dim array \
                with second dim equals 1')
        if X_train.shape[0] != y_train.shape[0]:
            raise Exception('X-train and y_train first dims must be equal')
        if X_train.shape[1] != self.net.n_inputs:
            raise Exception('X_train second dim must equals net n_inputs')

        batches_X = self._get_batches(X_train, batch_size)
        batches_X = self._resize_batches(batches_X)

        batches_y = self._get_batches(y_train, batch_size)
        batches_y = self._resize_batches(batches_y)

        batches = list(zip(batches_X, batches_y))

        for epoch_num in range(epochs):
            for batch_num in tqdm(
                range(len(batches)),
                desc="Epoch "+str(epoch_num+1)
            ):
                batch_X, batch_y = batches[batch_num]
                self.net.forward(batch_X)
                self.net.backward(batch_y, lr)

    def predict(
        self,
        X_test: np.array
    ):
        if len(X_test.shape) != 2:
            raise Exception('X_test must be 2-dim array')
        if X_test.shape[1] != self.net.n_inputs:
            raise Exception('X_test second dim must equals net n_inputs')

        X_test = self._resize_batches([X_test])[0]
        y_pred = self.net.forward(X_test)

        return np.resize(y_pred, (y_pred.shape[2], 1))

    def _get_batches(
        self,
        data,
        batch_size
    ) -> List[np.array]:

        if data.shape[0] % batch_size != 0:
            raise Exception('data first dim must be multiple of batch size')

        batches = [
            data[
                batch_num*batch_size:
                batch_num*batch_size+batch_size
            ]
            for batch_num in range(int(data.shape[0] / batch_size))
        ]

        return batches

    def _resize_batches(
        self,
        batches: List[np.array]
    ) -> List[np.array]:
        for i in range(len(batches)):
            batches[i] = np.resize(
                batches[i].T,
                (batches[i].shape[1], 1, batches[i].shape[0])
            )

        return batches
