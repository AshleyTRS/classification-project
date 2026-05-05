import numpy as np


def kmeans(X, k, max_iters=100, tol=1e-4):
    np.random.seed(42)

    indices = np.random.choice(len(X), k, replace=False)
    centroids = X[indices]

    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else centroids[i]
            for i in range(k)
        ])

        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    distribution = {
        f"Grupo {i}": int(np.sum(labels == i))
        for i in range(k)
    }

    return {
        "k": k,
        "centroids": centroids,
        "labels": labels,
        "distribution": distribution
    }


if __name__ == "__main__":
    try:
        X = np.loadtxt("../../../data/preprocessed/X_scaled.csv", delimiter=",")
    except Exception as e:
        print("Error cargando datos:", e)
        exit()

    k = int(input("Ingresa el valor de k: "))

    resultado = kmeans(X, k)

    print("\n=== RESULTADOS K-MEANS ===")
    print("K usado:", resultado["k"])

    print("\nDistribución de clusters:")
    for grupo, cantidad in resultado["distribution"].items():
        print(f"{grupo}: {cantidad} objetos")

    print("\nCentroides:")
    print(resultado["centroids"])
