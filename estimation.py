from sklearn.model_selection import cross_val_score


def evaluate(model, X, y, cv=5):
    """Оценка модели кросс-валидацией."""
    return cross_val_score(model, X, y, cv=cv).mean()
