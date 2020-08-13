import numpy as np

from .layer import Layer


class BatchNormalization(Layer):

    def __init__(
        self,
        layer_id: str,
        n_inputs: int,
        epsilon: float = 0.001
    ):
        super(BatchNormalization, self).__init__(
            layer_id=layer_id,
            n_inputs=n_inputs,
            n_outputs=n_inputs,
        )

        self.gamma = np.random.uniform()
        self.beta = np.random.uniform()
        self.epsilon = epsilon
        self.cache = None

    def _run_forward(
        self,
        inputs: np.array
    ) -> np.array:
        x = inputs

        mu = np.mean(np.mean(x, axis=2))
        xmu = x-mu

        sq = xmu**2
        var = np.mean(sq, axis=0)
        sqrtvar = np.sqrt(var + self.epsilon)
        ivar = 1/sqrtvar

        xn = xmu * ivar
        xngamma = xn * self.gamma
        xout = xngamma + self.beta

        self.cache = (x, mu, xmu, sq, var, sqrtvar, ivar, xn, xngamma, xout)

        return xout

    def _run_backward(
        self,
        error_wrt_output: np.array,
        lr: float
    ) -> np.array:

        dout = error_wrt_output
        (x, mu, xmu, sq, var, sqrtvar,
            ivar, xn, xngamma, xout) = self.cache
        N = x.shape[0]

        dbeta = np.sum(np.mean(dout, axis=2))
        self.beta = self.beta - lr * dbeta

        dxngamma = dout

        dgamma = np.sum(np.mean(xn * dxngamma, axis=2))
        dxn = dxngamma * self.gamma
        self.gamma = self.gamma - lr * dgamma

        dxmu1 = dxn * ivar

        divar = np.sum(np.mean(dxn * xmu, axis=2))
        dsqrtvar = divar * -1/(sqrtvar)**2
        dvar = 0.5 * 1/np.sqrt(var + self.epsilon) * dsqrtvar
        dsq = 1/N * np.ones_like(x) * dvar
        dxmu2 = 2 * xmu * dsq

        dx1 = dxmu1 + dxmu2

        dmu = -np.sum(np.mean(dxmu1 + dxmu2, axis=2))
        dx2 = 1/N * np.ones_like(x) * dmu

        dx = dx1 + dx2

        return dx
