import numpy as np

from ..trainer import Trainer


def test_trainer_get_batches_X(trainer: Trainer):
    X_train = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 1],
        [2, 3]
    ])
    batch_size = 2

    batches = trainer._get_batches(X_train, batch_size)

    assert len(batches) == 3
    assert np.array_equal(
        batches[0],
        np.array([
            [1, 2],
            [3, 4]
        ])
    )
    assert np.array_equal(
        batches[1],
        np.array([
            [5, 6],
            [7, 8]
        ])
    )
    assert np.array_equal(
        batches[2],
        np.array([
            [9, 1],
            [2, 3]
        ])
    )


def test_trainer_get_batches_y(trainer: Trainer):
    y_train = np.array([
        [1],
        [2],
        [3],
        [4],
        [5],
        [6]
    ])
    batch_size = 2

    batches = trainer._get_batches(y_train, batch_size)

    assert len(batches) == 3
    assert np.array_equal(
        batches[0],
        np.array([
            [1],
            [2]
        ])
    )
    assert np.array_equal(
        batches[1],
        np.array([
            [3],
            [4]
        ])
    )
    assert np.array_equal(
        batches[2],
        np.array([
            [5],
            [6]
        ])
    )


def test_trainer_resize_batches_X(trainer: Trainer):
    batches = [
        np.array([
            [1, 2],
            [3, 4]
        ]),
        np.array([
            [5, 6],
            [7, 8]
        ])
    ]

    batches = trainer._resize_batches(batches)

    assert len(batches) == 2
    assert np.array_equal(
        batches[0],
        np.array([
            [[1, 3]],
            [[2, 4]]
        ])
    )
    assert np.array_equal(
        batches[1],
        np.array([
            [[5, 7]],
            [[6, 8]]
        ])
    )


def test_trainer_resize_batches_y(trainer: Trainer):
    batches = [
        np.array([
            [1],
            [2]
        ]),
        np.array([
            [3],
            [4]
        ])
    ]

    batches = trainer._resize_batches(batches)

    assert len(batches) == 2
    assert np.array_equal(
        batches[0],
        np.array([
            [[1, 2]]
        ])
    )
    assert np.array_equal(
        batches[1],
        np.array([
            [[3, 4]]
        ])
    )


def test_trainer(trainer: Trainer):
    X_train = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 1],
        [2, 3]
    ])
    y_train = np.array([
        [1],
        [0],
        [1],
        [0],
        [1],
        [0]
    ])
    X_test = np.array([
        [1, 4],
        [3, 6]
    ])
    y_test = np.array([
        [1],
        [0]
    ])

    trainer.train(
        X_train=X_train,
        y_train=y_train,
        batch_size=2,
        epochs=2
    )

    y_pred = trainer.predict(X_test)

    assert y_pred.shape == y_test.shape
