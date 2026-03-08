import numpy as np
from sklearn.mixture import GaussianMixture


def perform_clustering(embeddings, n_clusters=20):

    print("Running fuzzy clustering...")

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=42
    )

    gmm.fit(embeddings)

    probabilities = gmm.predict_proba(embeddings)

    return gmm, probabilities