import numpy as np
from typing import List

from sklearn.cluster import SpectralClustering
from sklearn.manifold import spectral_embedding
from sklearn.cluster._kmeans import k_means

random_state = 42


def get_clusters(
    dist_matrix: np.ndarray,
    n_clusters: int = 2,
    delta=1.0,
):
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=random_state,
    )
    sim_matrix = np.exp(-(dist_matrix**2) / (2.0 * delta**2))
    sc_out = sc.fit(sim_matrix)
    clusters = {}
    for model_no, cluster in enumerate(sc_out.labels_):
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(model_no)
    return list(clusters.values())


def get_spectral_embed(
    dist_matrix: np.ndarray,
    n_clusters: int = 2,
    delta=1.0,
):
    sim_matrix = np.exp(-(dist_matrix**2) / (2.0 * delta**2))
    se = spectral_embedding(
        sim_matrix,
        random_state=random_state,
        n_components=n_clusters - 1,
    )
    return se.reshape(-1)


def get_sc_centroid_dists(
    dist_matrix: np.ndarray,
    n_clusters: int = 2,
    delta=1.0,
) -> List[List[float]]:
    sim_matrix = np.exp(-(dist_matrix**2) / (2.0 * delta**2))
    se = spectral_embedding(
        sim_matrix,
        n_components=n_clusters,
        random_state=random_state,
        drop_first=False,
    )
    cluster_centers, labels, inertia = k_means(
        se,
        n_clusters,
        random_state=random_state,
    )

    dists = []
    print(se.shape, cluster_centers.shape)
    for center in cluster_centers:
        dists.append([float(np.linalg.norm(embed - center)) for embed in se])

    return dists
