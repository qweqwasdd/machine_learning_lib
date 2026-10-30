import numpy as np


def mean_squared_error(y_true, y_pred):
    """Среднеквадратичная ошибка."""
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
