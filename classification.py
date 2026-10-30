from sklearn.linear_model import LogisticRegression


def train_classifier(X, y):
    """Обучает модель классификации."""
    model = LogisticRegression()
    model.fit(X, y)
    return model
