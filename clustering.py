from sklearn.cluster import KMeans


def train_clustering(X, n_clusters=3):
    """Кластеризация методом k-средних."""
    model = KMeans(n_clusters=n_clusters, n_init=10)
    model.fit(X)
    return model
