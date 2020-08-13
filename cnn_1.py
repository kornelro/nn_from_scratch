# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys; sys.path.append('../')


# %%
from cnn.trainer import Trainer
from cnn.net import Net
from cnn.layers.fully_connected import  FullyConnected
from cnn.layers.batch_normalization import BatchNormalization
from cnn.layers.sigmoid import Sigmoid


# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
import keras
import matplotlib.pyplot as plt

# %% [markdown]
# Data preparation

# %%
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


# %%
X_train.shape


# %%
X_test.shape


# %%
y_train.shape


# %%
y_test.shape


# %%
# plt.figure()

# plt.subplot(1, 3, 1)
# plt.imshow(X_train[0])

# plt.subplot(1, 3, 2)
# plt.imshow(X_train[1])

# plt.subplot(1, 3, 3)
# plt.imshow(X_train[2])

# plt.subplots_adjust(right=2)
# plt.show()


# %%
X_train = X_train.reshape((X_train.shape[0], 28*28))
X_test = X_test.reshape((X_test.shape[0], 28*28))
y_train.resize((y_train.shape[0],1))
y_test.resize((y_test.shape[0],1))


# %%
X_train = X_train[:1000]
X_train.shape


# %%
X_test = X_test[:100]
X_test.shape


# %%
y_train = y_train[:1000]
y_train.shape


# %%
y_test = y_test[:100]
y_test.shape


# %%
# for i in range(y_train.shape[0]):
#     if y_train[i][0] == 4:
#         y_train[i][0] = 1
#     else:
#         y_train[i][0] = 0

# for i in range(y_test.shape[0]):
#     if y_test[i][0] == 4:
#         y_test[i][0] = 1
#     else:
#         y_test[i][0] = 0


# %%
pd.Series(y_train.T[0]).value_counts()


# %%
pd.Series(y_test.T[0]).value_counts()


# %%
X_train = X_train / 255
X_test = X_test / 255

# %% [markdown]
# Net 

# %%
net = Net(
    layers=(
        FullyConnected(
            'fc_hidden',
            n_inputs=28*28,
            n_neurons=128,
        ),
        BatchNormalization(
            'bn_hidden',
            128
        ),
        Sigmoid(
            'sigmoid_hidden',
            n_inputs=128,
        ),
        FullyConnected(
            'fc_hidden',
            n_inputs=128,
            n_neurons=64,
        ),
        BatchNormalization(
            'bn_hidden',
            64
        ),
        Sigmoid(
            'sigmoid_hidden',
            n_inputs=64,
        ),
        FullyConnected(
            'fc_hidden',
            n_inputs=64,
            n_neurons=32,
        ),
        BatchNormalization(
            'bn_hidden',
            32
        ),
        Sigmoid(
            'sigmoid_hidden',
            n_inputs=32,
        ),
        FullyConnected(
            'fc_output',
            n_inputs=32,
            n_neurons=10,
        ),
        BatchNormalization(
            'bn_hidden',
            10
        ),
        Sigmoid(
            'sigmoid_output',
            n_inputs=10,
        ),
        FullyConnected(
            'fc_output',
            n_inputs=10,
            n_neurons=5,
        ),
        BatchNormalization(
            'bn_hidden',
            5
        ),
        Sigmoid(
            'sigmoid_output',
            n_inputs=5,
        )
    )
)

# %% [markdown]
# Trainig

# %%
trainer = Trainer(net=net)


# %%
trainer.train(
    X_train=X_train,
    y_train=y_train,
    batch_size=100,
    epochs=1,
    lr=0.1
)

# %% [markdown]
# Predict

# %%
y_pred = trainer.predict(
    X_test=X_test
)


# %%
y_pred


# %%
y_test


# %%
np.sum((y_pred-y_test)**2)


