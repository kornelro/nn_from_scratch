import numpy as np


def input_is_proper_size(
    array: np.array,
    first_dim: int
) -> bool:
    proper_size = True
    shape = array.shape

    if (len(shape) != 3
            or shape[0] != first_dim
            or shape[1] != 1
            or shape[2] < 1):

        proper_size = False

    return proper_size
