from sklearn.linear_model import LinearRegression


def train_regressor(X, y):
    """Обучает модель линейной регрессии."""
    model = LinearRegression()
    model.fit(X, y)
    return model
