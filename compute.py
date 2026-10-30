import numpy as np


def mean_squared_error(y_true, y_pred):
    """Среднеквадратичная ошибка."""
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)


def mean_absolute_error(y_true, y_pred):
    """Средняя абсолютная ошибка."""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def r2_score(y_true, y_pred):
    """Коэффициент детерминации R^2."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot
