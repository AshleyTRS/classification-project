import numpy as np
from implementation import kmeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist


def cs_index(X, labels, centroids):
    k = len(centroids)

    # Distancias entre centroides
    centroid_distances = cdist(centroids, centroids)
    np.fill_diagonal(centroid_distances, np.inf)
    min_centroid_dist = np.min(centroid_distances, axis=1)

    # Dispersión intra-cluster
    intra_dists = []
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) == 0:
            intra_dists.append(0)
        else:
            dists = np.linalg.norm(cluster_points - centroids[i], axis=1)
            intra_dists.append(np.mean(dists))

    intra_dists = np.array(intra_dists)

    # CS Index
    cs = np.sum(intra_dists / min_centroid_dist)
    return cs


if __name__ == "__main__":
    try:
        X = np.loadtxt("../../../data/preprocessed/X_scaled.csv", delimiter=",")
    except Exception as e:
        print("Error cargando datos:", e)
        exit()

    k = int(input("Ingresa el valor de k: "))

    resultado = kmeans(X, k)

    labels = resultado["labels"]
    centroids = resultado["centroids"]

    print("\n=== ÍNDICES DE VALIDEZ ===")

    # Calinski-Harabasz
    ch = calinski_harabasz_score(X, labels)
    print(f"Calinski-Harabasz (Kalinsky): {ch:.4f}")

    # Davies-Bouldin
    db = davies_bouldin_score(X, labels)
    print(f"Davies-Bouldin: {db:.4f}")

    # CS Index
    cs = cs_index(X, labels, centroids)
    print(f"CS Index: {cs:.4f}")
